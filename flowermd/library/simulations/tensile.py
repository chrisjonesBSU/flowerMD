"""Tensile simulation class."""

import hoomd
import numpy as np

from flowermd.base.simulation import Simulation
from flowermd.utils import HOOMDThermostats, PullParticles, StressStrainLogger


class Tensile(Simulation):
    """Tensile simulation class.

    Parameters
    ----------
    tensile_axis : tuple of int, required
        The axis along which to apply the tensile strain.
    fix_ratio : float, default=0.20
        The ratio of the box length to fix particles at each end
        of the tensile axis.

    """

    def __init__(
        self,
        initial_state,
        forcefield,
        tensile_axis,
        fix_ratio=0.20,
        reference_values=dict(),
        dt=0.0001,
        device=hoomd.device.auto_select(),
        seed=42,
        gsd_write_freq=1e4,
        gsd_file_name="trajectory.gsd",
        log_write_freq=1e3,
        log_file_name="log.txt",
        log_stress_real_time=True,
        log_particles=False,
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
            log_particles=log_particles,
            thermostat=thermostat,
        )
        self.tensile_axis = np.asarray(tensile_axis)
        self.fix_ratio = fix_ratio
        self._axis_index = np.where(self.tensile_axis != 0)[0].astype(int)
        self.initial_box = self.box_lengths_reduced
        self.initial_length = self.initial_box[self._axis_index]
        self.fix_length = self.initial_length * fix_ratio
        self.log_stress_real_time = log_stress_real_time
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
        # Set up logger and data structures
        # Make a list of arrays, one for each time Tensile.run() is called.
        self._initial_timestep = np.copy(self.timestep)
        self._run_strain = None
        self._run_stress = None
        self._strain_logs = []
        self._stress_logs = []
        # Set up custom action
        if self.log_stress_real_time:
            tensile_log = StressStrainLogger(
                sim=self, pull_axis=self._axis_index
            )
            tensile_logger = hoomd.update.CustomUpdater(
                trigger=hoomd.trigger.Periodic(int(log_write_freq)),
                action=tensile_log,
            )
            self.operations.updaters.append(tensile_logger)

    @property
    def strain(self):
        """The current strain of the simulation."""
        delta_L = (
            self.box_lengths_reduced[self._axis_index] - self.initial_length
        )
        return delta_L / self.initial_length

    @property
    def strain_data(self):
        """Combines strain data collected for all tensile runs."""
        if not self.log_stress_real_time:
            raise ValueError(
                "Strain data is not available. "
                "Set `log_stress_real_time` to `True` to "
                "log strain and stress data in real time. "
            )
        return np.concatenate(self._strain_logs)

    @property
    def stress_data(self):
        """Combines stress data collected for all tensile runs."""
        if not self.log_stress_real_time:
            raise ValueError(
                "Stress data is not available. "
                "Set `log_stress_real_time` to `True` to "
                "log strain and stress data in real time. "
            )
        return np.concatenate(self._stress_logs)

    def compile_stress_strain_data(self):
        """Perofmrs averaging with errors for the
        saved strain and stress run logs.

        Uses numpy.mean() and numpy.std() to
        calculate averages and errors.

        Returns
        -------
        tuple of np.ndarray
            (strain, stress averages, stress errors)

        """
        if not self.log_stress_real_time:
            raise ValueError(
                "Stress data is not available. "
                "Set `log_stress_real_time` to `True` to "
                "log strain and stress data in real time. "
            )
        strains = np.unique(self.strain_data)
        stress_means = np.zeros_like(strains)
        stress_stds = np.zeros_like(strains)
        for idx, strain in enumerate(strains):
            indices = np.where(self.strain_data == strain)[0]
            stress_values = self.stress_data[indices]
            stress_means[idx] = np.mean(stress_values)
            stress_stds[idx] = np.std(stress_values)
        return (strains, stress_means, stress_stds)

    def save_stress_strain_data(self, filename):
        """Save the compiled stress vs strain data to file.

        filename : str, required
            Filepath to save the numpy array to.
            Must be a file type compatible with numpy.save()

        """
        strain, stress_avg, stress_std = self.compile_stress_strain_data()
        data = np.vstack([strain, stress_avg, stress_std]).T
        np.save(file=filename, arr=data)

    def run_tensile(self, strain, n_steps, kT, tau_kt, period):
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
        if self.log_stress_real_time:
            log_array_size = int(n_steps // self.log_write_freq) + 1
            self._run_strain = np.zeros(log_array_size)
            self._run_stress = np.zeros(log_array_size)
        current_length = self.box_lengths_reduced[self._axis_index]
        final_length = current_length * (1 + strain)
        final_box = np.copy(self.box_lengths_reduced)
        final_box[self._axis_index] = final_length
        shift_by = (final_length - current_length) / (n_steps // period)
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
        )
        particle_updater = hoomd.update.CustomUpdater(
            trigger=resize_trigger, action=particle_puller
        )
        self.operations.updaters.append(box_resizer)
        self.operations.updaters.append(particle_updater)
        self.run_NVT(n_steps=n_steps + 1, kT=kT, tau_kt=tau_kt)
        if self.log_stress_real_time:
            self._strain_logs.append(self._run_strain)
            self._stress_logs.append(self._run_stress)
