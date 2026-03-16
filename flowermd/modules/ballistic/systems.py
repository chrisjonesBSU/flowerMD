import gsd.hoomd
import numpy as np


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
        self.projectile_velocity = np.asarray(projectile_velocity)
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

        pos = np.concatenate((target_xyz, projectile_xyz), axis=None)
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
