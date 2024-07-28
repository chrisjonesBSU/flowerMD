"""Tensile simulation class."""

import hoomd
import numpy as np
import unyt as u

from flowermd.base.simulation import Simulation
from flowermd.utils import HOOMDThermostats


class Shear(Simulation):
    """Shear simulation class.

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
        self._shear_left_force = None
        self._shear_right_force = None
        self._shear_forces = []

    @property
    def shear_forces_reduced(self):
        """The shear force in reduced units."""
        return (self._shear_left_force, self._shear_right_force)

    @property
    def shear_forces(self):
        """The shear force in reduced units."""
        if self.shear_forces_reduced == (None, None):
            return (None, None)
        if self.reference_values:
            conv_factor = (
                self.reference_mass.to("kg") * self.reference_length.to("m")
            ) / u.Unit("s**2")
        else:
            conv_factor = 1
        left_shear = self.shear_force_reduced[0] * conv_factor
        right_shear = self.shear_force_reduced[1] * conv_factor
        return (left_shear, right_shear)

    def add_shear_forces(
        self,
        shear_force,
        shear_normal,
        shear_depth,
    ):
        """"""
        if isinstance(shear_depth, u.unyt_array):
            length_units = self.reference_length.units
            shear_depth = shear_depth.to(length_units)
            # This is the shear depth in reduced units
            shear_depth /= self.reference_length
        if isinstance(shear_force, u.unyt_array):
            shear_force = shear_force.to("N")
            conv_factor = (
                self.reference_mass.to("kg") * self.reference_length.to("m")
            ) / u.Unit("s**2")
            # This is the shear force in reduced units
            shear_force /= conv_factor
            self._shear_left_force = shear_force
            self._shear_right_force = -shear_force
        # Create set of particles to apply shear force
        normal_index = np.where(shear_normal != 0)[0]
        snapshot = self.state.get_snapshot()
        positions = snapshot.particles.position[:, normal_index]
        box_max = self.box_lengths_reduced / 2
        box_min = -box_max
        left_tags = np.where(positions < (box_min + shear_depth))[0]
        right_tags = np.where(positions > (box_max - shear_depth))[0]
        shear_left = hoomd.filter.Tags(left_tags.astype(np.uint32))
        shear_right = hoomd.filter.Tags(right_tags.astype(np.uint32))
        # Create 2 force objects, set equal and opposite shear forces
        shear_left_force = hoomd.md.force.Constant(filter=shear_left)
        shear_right_force = hoomd.md.force.Constant(filter=shear_right)
        shear_left_force[self.state.particle_types] = shear_force
        shear_right_force[self.state.particle_types] = -shear_force
        self.add_force(shear_left_force)
        self.add_force(shear_right_force)
        self._shear_forces.extend([shear_left_force, shear_right_force])

    def remove_shear_forces(self):
        """"""
        for force in self._shear_forces:
            self.remove_force(force)
        self._shear_right_force = None
        self._shear_left_force = None
