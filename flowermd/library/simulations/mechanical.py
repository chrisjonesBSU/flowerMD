"""Tensile simulation class."""

import hoomd
import numpy as np
import unyt as u

from flowermd.base.simulation import Simulation
from flowermd.utils import HOOMDThermostats, PullParticles


class Shear(Simulation):
    """Shear test simulation class.

    Parameters
    ----------
    shear_axis : tupe of int, required
        The axis along which to apply the shear strain.
    shear_axis_normal : tupe of int, required
        The direction normal to the shear axis. This is used along
        with fix_ratio or fix_depth to set the fixed wall of
        particles that have their positions updated.
    fix_ratio : float, optional, default None
        The ratio of the box length to fix particles at each end
        of the direciton normal to the shear axis.
    fix_depth : float, optional, default None
        The depth of the box length to fix particles at each end
        of the direction normal to the shear axis.

    """

    def __init__(
        self,
        initial_state,
        forcefield,
        shear_axis,
        shear_axis_normal,
        fix_ratio=None,
        fix_length=None,
        reference_values=dict(),
        dt=0.0001,
        device=hoomd.device.auto_select(),
        seed=42,
        gsd_write_freq=1e4,
        gsd_file_name="trajectory.gsd",
        log_write_freq=1e3,
        log_file_name="log.txt",
        thermostat=HOOMDThermostats.MTTK,
    ):
        super(Shear, self).__init__(
            initial_state=initial_state,
            forcefield=forcefield,
            reference_values=reference_values,
            dt=dt,
            device=device,
            seed=seed,
            gsd_write_freq=gsd_write_freq,
            gsd_file_name=gsd_file_name,
            log_write_freq=log_write_freq,
            log_file_name=log_file_name,
            thermostat=thermostat,
        )
        if not any([fix_ratio, fix_length]) or all([fix_ratio, fix_length]):
            raise ValueError("Specify only one of fix_ratio or fix_length.")
        self.shear_axis = np.asarray(shear_axis)
        self.normal_axis = np.asarray(shear_axis_normal)
        self._shear_axis_index = np.where(self.shear_axis != 0)[0]
        self._normal_axis_index = np.where(self.normal_axis != 0)[0]
        # Use fix_ratio to get fixed length in reduced units
        if fix_ratio:
            box_length_normal = self.box_lengths_reduced[
                self._normal_axis_index
            ]
            self.fix_length = box_length_normal * fix_ratio
        else:
            # Check fix_length for units, convert to reduced units
            if isinstance(fix_length, u.unyt_array):
                fix_length = fix_length.to(self.reference_length.unit)
                fix_length /= self.reference_length
            self.fix_length = fix_length
        # Set up walls of fixed particles:
        snapshot = self.state.get_snapshot()
        positions = snapshot.particles.position[:, self._normal_axis_index]
        box_length_normal = self.box_lengths_reduced[self._normal_axis_index]
        box_max = box_length_normal / 2
        box_min = -box_max
        left_tags = np.where(positions < (box_min + self.fix_length))[0]
        right_tags = np.where(positions > (box_max - self.fix_length))[0]
        self.fix_left = hoomd.filter.Tags(left_tags.astype(np.uint32))
        self.fix_right = hoomd.filter.Tags(right_tags.astype(np.uint32))
        all_fixed = hoomd.filter.Union(self.fix_left, self.fix_right)
        # Set the group of particles to be integrated over
        self.integrate_group = hoomd.filter.SetDifference(
            hoomd.filter.All(), all_fixed
        )

    def run_shear(
        self,
        n_steps,
        kT,
        tau_kt,
        period,
        strain=None,
        shear_length=None,
        ensemble="NVT",
    ):
        """Run a shear simulation.

        Parameters
        ----------
        strain : float, required
            The strain used to calculate shearing distance.
        n_steps : int, required
            The number of steps to run the simulation for.
        tau_kt : float, required
            Thermostat coupling period (in simulation time units).
        period : int, required
            The period number of simulation steps between the strain updates.

        """
        if all([strain, shear_length]) or not any([strain, shear_length]):
            raise ValueError("Specify only one of strain or shear_length.")
        if strain:
            current_length = self.box_lengths_reduced[self._shear_axis_index]
            final_length = current_length * (1 + strain)
            shift_by = (final_length - current_length) / (n_steps // period)
        else:
            if isinstance(shear_length, u.unyt_array):
                shear_length = shear_length.to(self.reference_length.unit)
                shear_length /= self.reference_length
            shift_by = shear_length / (n_steps // period)
        resize_trigger = hoomd.trigger.Periodic(int(period))
        particle_puller = PullParticles(
            shift_by=shift_by / 2,
            axis=self.shear_axis,
            neg_filter=self.fix_left,
            pos_filter=self.fix_right,
            box=self.box_lengths_reduced,
            apply_pbc=True,
        )
        particle_updater = hoomd.update.CustomUpdater(
            trigger=resize_trigger, action=particle_puller
        )
        self.operations.updaters.append(particle_updater)
        if ensemble.lower() == "nvt":
            self.run_NVT(n_steps=n_steps + 1, kT=kT, tau_kt=tau_kt)
        if ensemble.lower() == "nve":
            self.run_NVE(n_steps=n_steps + 1, kT=kT)
        self.operations.updaters.remove(particle_updater)


class Tensile(Simulation):
    """Tensile test simulation class.

    Parameters
    ----------
    tensile_axis : tuple of int, required
        The axis along which to apply the tensile strain.
    fix_ratio : float, required
        The ratio of the box length to fix particles at each end
        of the tensile axis.

    """

    def __init__(
        self,
        initial_state,
        forcefield,
        tensile_axis,
        fix_ratio=None,
        fix_length=None,
        reference_values=dict(),
        dt=0.0001,
        device=hoomd.device.auto_select(),
        seed=42,
        gsd_write_freq=1e4,
        gsd_file_name="trajectory.gsd",
        log_write_freq=1e3,
        log_file_name="log.txt",
        thermostat=HOOMDThermostats.MTTK,
    ):
        super(Tensile, self).__init__(
            initial_state=initial_state,
            forcefield=forcefield,
            reference_values=reference_values,
            dt=dt,
            device=device,
            seed=seed,
            gsd_write_freq=gsd_write_freq,
            gsd_file_name=gsd_file_name,
            log_write_freq=log_write_freq,
            log_file_name=log_file_name,
            thermostat=thermostat,
        )
        if not any([fix_ratio, fix_length]) or all([fix_ratio, fix_length]):
            raise ValueError("Specify only one of fix_ratio or fix_length.")
        self.tensile_axis = np.asarray(tensile_axis)
        self.fix_ratio = fix_ratio
        self._axis_index = np.where(self.tensile_axis != 0)[0]
        self.initial_box = self.box_lengths_reduced
        self.initial_length = self.initial_box[self._axis_index]
        if fix_ratio:
            self.fix_length = self.initial_length * fix_ratio
        else:
            # Check fix_length for units, convert to reduced units
            if isinstance(fix_length, u.unyt_array):
                fix_length = fix_length.to(self.reference_length.unit)
                fix_length /= self.reference_length
            self.fix_length = fix_length
        # Set up walls of fixed particles:
        snapshot = self.state.get_snapshot()
        positions = snapshot.particles.position[:, self._axis_index]
        box_max = self.initial_length / 2
        box_min = -box_max
        left_tags = np.where(positions < (box_min + self.fix_length))[0]
        right_tags = np.where(positions > (box_max - self.fix_length))[0]
        self.fix_left = hoomd.filter.Tags(left_tags.astype(np.uint32))
        self.fix_right = hoomd.filter.Tags(right_tags.astype(np.uint32))
        all_fixed = hoomd.filter.Union(self.fix_left, self.fix_right)
        # Set the group of particles to be integrated over
        self.integrate_group = hoomd.filter.SetDifference(
            hoomd.filter.All(), all_fixed
        )

    @property
    def strain(self):
        """The current strain of the simulation."""
        delta_L = (
            self.box_lengths_reduced[self._axis_index] - self.initial_length
        )
        return delta_L / self.initial_length

    def run_tensile(
        self, n_steps, kT, tau_kt, period, strain=None, tensile_length=None
    ):
        """Run a tensile test simulation.

        Parameters
        ----------
        strain : float, required
            The strain to apply to the simulation.
        n_steps : int, required
            The number of steps to run the simulation for.
        tau_kt : float, required
            Thermostat coupling period (in simulation time units).
        period : int, required
            The period of the strain application.

        """
        if all([strain, tensile_length]) or not any([strain, tensile_length]):
            raise ValueError("Specify only one of strain or shear_length.")
        if strain:
            current_length = self.box_lengths_reduced[self._axis_index]
            final_length = current_length * (1 + strain)
            final_box = np.copy(self.box_lengths_reduced)
            final_box[self._axis_index] = final_length
            shift_by = (final_length - current_length) / (n_steps // period)
        else:
            if isinstance(tensile_length, u.unyt_array):
                tensile_length = tensile_length.to(self.reference_length.unit)
                tensile_length /= self.reference_length
            shift_by = tensile_length / (n_steps // period)

        resize_trigger = hoomd.trigger.Periodic(period)
        box_ramp = hoomd.variant.Ramp(
            A=0, B=1, t_start=self.timestep, t_ramp=int(n_steps)
        )
        box_resizer = hoomd.update.BoxResize(
            box1=self.box_lengths_reduced,
            box2=final_box,
            variant=box_ramp,
            trigger=resize_trigger,
            filter=hoomd.filter.Null(),
        )
        particle_puller = PullParticles(
            shift_by=shift_by / 2,
            axis=self.tensile_axis,
            neg_filter=self.fix_left,
            pos_filter=self.fix_right,
            box=self.box_lengths_reduced,
            apply_pbc=False,
        )
        particle_updater = hoomd.update.CustomUpdater(
            trigger=resize_trigger, action=particle_puller
        )
        self.operations.updaters.append(box_resizer)
        self.operations.updaters.append(particle_updater)
        self.run_NVT(n_steps=n_steps + 1, kT=kT, tau_kt=tau_kt)
        self.operations.updates.remove(box_resizer)
        self.operations.updaters.remove(particle_updater)
