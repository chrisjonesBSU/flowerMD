import gsd.hoomd
import hoomd
import numpy as np

from flowermd.base.simulation import Simulation
from flowermd.utils import HOOMDThermostats


class Projectile:
    def __init__(self, snapshot):
        self.snapshot = snapshot

    @classmethod
    def from_compound(cls, compound):
        top = compound.to_gmso()
        snapshot = top.to_gsd_snapshot()
        return cls(snapshot=snapshot)

    @classmethod
    def LJ_bead(cls, mass=1, initial_pos=(0.0, 0.0, 0.0)):
        snapshot = gsd.hoomd.Frame()
        snapshot.particles.N = 1
        snapshot.particles.position = np.array(initial_pos)
        snapshot.particles.mass = mass
        snapshot.particles.types = ["PROJ"]
        snapshot.particles.typeid = [0]
        return cls(snapshot=snapshot)

    @classmethod
    def from_gsd(cls, gsd_file, initial_frame=-1):
        with gsd.hoomd.open(gsd_file, "r") as traj:
            snapshot = traj[initial_frame]
            return cls(snapshot=snapshot)


class ImpactSystem:
    def __init__(
        self,
        target,
        projectile,
        impact_axis,
        projectile_velocity,
        box_expand=2,
        starting_frame=-1,
        wall_sigma=0.25,
    ):
        self.target = target
        self.projectile = projectile
        self.projectile_velocity = float(projectile_velocity)
        self.impact_axis = np.asarray(impact_axis)
        self.box_expand = box_expand
        self.wall_sigma = wall_sigma
        if isinstance(target, str) and target.split(".")[1] == "gsd":
            with gsd.hoomd.open(target) as traj:
                self.target_snap = traj[starting_frame]
        elif isinstance(target, gsd.hoomd.Frame):
            self.target_snap = target

        if isinstance(projectile, str) and target.split(".")[1] == "gsd":
            with gsd.hoomd.open(projectile) as traj:
                self.projectile_snap = traj[starting_frame]
        elif isinstance(projectile, gsd.hoomd.Frame):
            self.projectile_snap = projectile

        self.target_indices = np.arange(0, self.target_snap.particles.N)
        self.projectile_indices = np.arange(
            self.target_snap.particles.N,
            self.target_snap.particles.N + self.projectile_snap.particles.N,
        )
        self.hoomd_snapshot = self._build()

        self.target_filter = hoomd.filter.Tags(
            range(self.target_snap.particles.N)
        )
        self.projectile_filter = hoomd.filter.Tags(
            range(
                self.target_snap.particles.N,
                self.target_snap.particles.N + self.projectile_snap.particles.N,
            )
        )

    def _build(self):
        system = gsd.hoomd.Frame()
        box_expand_index = np.where(self.impact_axis != 0)[0]
        system.configuration.box = self.target_snap.configuration.box
        system.configuration.box[box_expand_index] *= self.box_expand
        shift = (
            system.configuration.box[box_expand_index] - self.wall_sigma
        ) / 4
        target_xyz = np.copy(self.target_snap.particles.position)
        projectile_xyz = np.copy(self.projectile.particles.position)
        # Shift target coordinates to the left
        target_xyz[:, box_expand_index] -= shift
        if projectile_xyz.shape == (3,):
            projectile_xyz[box_expand_index] += float(shift[0])
        else:
            projectile_xyz[:, box_expand_index] += float(shift)

        system.particles.N = (
            self.target_snap.particles.N + self.projectile_snap.particles.N
        )
        system.bonds.N = self.target_snap.bonds.N + self.projectile_snap.bonds.N
        system.angles.N = (
            self.target_snap.angles.N + self.projectile_snap.angles.N
        )
        system.dihedrals.N = (
            self.target_snap.dihedrals.N + self.projectile_snap.dihedrals.N
        )
        system.pairs.N = self.target_snap.pairs.N + self.projectile_snap.pairs.N
        projectile_velocities = np.zeros((self.projectile_snap.particles.N, 3))
        projectile_velocities -= self.impact_axis * self.projectile_velocity

        pos = np.concatenate((target_xyz, projectile_xyz), axis=None)
        velocity = np.concatenate(
            (self.target_snap.particles.velocity, projectile_velocities), axis=0
        )
        mass = np.concatenate(
            (
                self.target_snap.particles.mass,
                self.projectile_snap.particles.mass,
            ),
            axis=None,
        )
        charges = np.concatenate(
            (
                self.target_snap.particles.charge,
                self.projectile_snap.particles.charge,
            ),
            axis=None,
        )
        system.particles.position = pos
        system.particles.velocity = velocity
        system.particles.mass = mass
        system.particles.charge = charges
        system.particles.types = list(
            set(
                self.target_snap.particles.types
                + self.projectile_snap.particles.types
            )
        )

        type_ids = []
        for type_id in self.target_snap.particles.typeid:
            ptype = self.target_snap.particles.types[type_id]
            new_type_id = system.particles.types.index(ptype)
            type_ids.append(new_type_id)

        for type_id in self.projectile_snap.particles.typeid:
            ptype = self.projectile_snap.particles.types[type_id]
            new_type_id = system.particles.types.index(ptype)
            type_ids.append(new_type_id)

        system.particles.typeid = type_ids

        # Set up bonds:
        system.bonds.group = self.target_snap.bonds.group
        system.bonds.types = self.target_snap.bonds.types
        system.bonds.type_id = self.target_snap.bonds.typeid

        system.angles.group = self.target_snap.angles.group
        system.angles.typeid = self.target_snap.angles.typeid
        system.angles.types = self.target_snap.angles.types

        # Set up dihedrals:
        system.dihedrals.group = self.target_snap.dihedrals.group
        system.dihedrals.typeid = self.target_snap.dihedrals.typeid
        system.dihedrals.types = self.target_snap.dihedrals.types

        # Set up pairs:
        if self.target_snap.pairs.N > 0:
            system.pairs.group = self.target_snap.pairs.group
            system.pairs.typeid = self.target_snap.pairs.typeid
            system.pairs.types = self.target_snap.pairs.types
        return system


class ImpactSimulation(Simulation):
    def __init__(
        self,
        initial_state,
        forcefield,
        impact_axis,
        target_filter,
        projectile_filter,
        wall_sigma,
        wall_epsilon,
        wall_r_cut,
        wall_r_extrap=0,
        reference_values=dict(),
        dt=0.0001,
        device=hoomd.device.auto_select(),
        seed=42,
        gsd_write_freq=1e4,
        gsd_file_name="impact.gsd",
        log_write_freq=1e3,
        log_file_name="impact_sim_data.txt",
        thermostat=HOOMDThermostats.MTTK,
    ):
        super(ImpactSimulation, self).__init__(
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
        self.impact_axis = impact_axis
        self.target_filter = target_filter
        self.projectile_filter = projectile_filter

        self.add_walls(
            self.impact_axis,
            wall_sigma,
            wall_epsilon,
            wall_r_cut,
            wall_r_extrap,
        )

    def _thermalize_system(self, kT):
        """Assign random velocities to all particles.

        Parameters
        ----------
        kT : float or hoomd.variant.Ramp, required
            The temperature to use during the thermalization.

        """
        if isinstance(kT, hoomd.variant.Ramp):
            self.state.thermalize_particle_momenta(
                filter=self.target_filter, kT=kT.range[0]
            )
        else:
            self.state.thermalize_particle_momenta(
                filter=self.target_filter, kT=kT
            )
