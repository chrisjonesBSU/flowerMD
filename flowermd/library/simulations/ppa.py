import itertools

import hoomd
import numpy as np
from cmeutils.gsd_utils import get_molecule_cluster

from flowermd.base import Simulation


class PrimitivePathAnalysis(Simulation):
    def __init__(
        self,
        initial_state,
        head_particle_index,
        tail_particle_index,
        device=hoomd.device.auto_select(),
        seed=42,
        gsd_write_freq=1e4,
        gsd_file_name="ppa.gsd",
        log_write_freq=1e3,
        log_file_name="log-ppa.txt",
    ):
        """"""
        # TODO Check initial state for type? If GSD file, take last frame, enforce frame only?
        self.head_particle_index = head_particle_index
        self.tail_particle_index = tail_particle_index
        self.frame, self.head_tail_indices = self._update_frame(initial_state)
        super(PrimitivePathAnalysis, self).__init(
            initial_state=self.frame,
            forcefield=None,
            device=device,
            seed=seed,
            gsd_write_freq=gsd_write_freq,
            gsd_file_name=gsd_file_name,
            log_write_freq=log_write_freq,
            log_file_name=log_file_name,
        )
        # Head and tail of each chain is held fixed in place
        self.integrate_group = hoomd.filter.SetDifference(
            hoomd.filter.All(), hoomd.filter.Tags(self.head_tail_indices)
        )

    def run_ppa(
        self,
        n_steps,
        dt,
        kT,
        tau_kT,
        bond_r0=0.01,
        bond_k=100,
        bond_delta=0.0,
        pair_epsilon=1,
        pair_sigma=1,
        pair_r_cut=2.5,
    ):
        """"""
        self.dt = dt
        if self.forces:
            for force in self.forces:
                self.remove_force(force)
        # Create PPA forcefield, add to integrator forces
        ppa_pair, ppa_bond = self._create_forces(
            bond_r0=bond_r0,
            bond_k=bond_k,
            bond_delta=bond_delta,
            pair_epsilon=pair_epsilon,
            pair_sigma=pair_sigma,
            pair_r_cut=pair_r_cut,
        )
        self.add_force(ppa_pair)
        self.add_force(ppa_bond)
        # TODO: Add action to monitor bonds
        # TODO: Add action that keeps track of state history?
        self.run_NVT(n_steps=n_steps, kT=kT, tau_kT=tau_kT)

    def _update_frame(self):
        """"""
        cluster, cl_props = get_molecule_cluster(snap=self.frame)
        n_chains = len(cluster.cluster_keys)
        types_list = [f"C{i}" for i in range(n_chains)]
        self.frame.particles.types = types_list

        type_ids = np.zeros_like(self.frame.particles.typeid)
        for idx, indices in enumerate(cluster.cluster_keys):
            type_ids[indices] = idx
        self.frame.particles.typeid = type_ids
        self.frame.particles.velocity = None  # Zero out the particle velocities
        # Add bonds
        bond_types = []
        bond_ids = []
        last_type = None
        id_count = -1

        for group in self.frame.bonds.group:
            type1 = self.frame.particles.types[
                self.rame.particles.typeid[group[0]]
            ]
            type2 = self.frame.particles.types[
                self.frame.particles.typeid[group[1]]
            ]
            bond_type = "-".join([type1, type2])
            if bond_type != last_type:
                last_type = bond_type
                bond_types.append(bond_type)
                id_count += 1
            bond_ids.append(id_count)

        self.frame.bonds.N = len(bond_ids)
        self.frame.bonds.types = bond_types
        self.frame.bonds.typeid = bond_ids
        # Index numbers for head and tail particles of each chain
        head_tail_indices = []
        for indices in cluster.cluster_keys:
            head_tail_indices.append(indices[self.head_index])
            head_tail_indices.append(indices[self.tail_index])
        return head_tail_indices

    def _create_forces(
        self,
        bond_r0,
        bond_k,
        bond_delta,
        pair_epsilon,
        pair_sigma,
        pair_r_cut,
    ):
        bond = hoomd.md.bond.FENEWCA()
        lj = hoomd.md.pair.LJ(
            default_r_cut=pair_r_cut, nlist=hoomd.md.nlist.Cell(buffer=0.40)
        )

        for btype in self.frame.bonds.types:
            bond.params[btype] = dict(
                k=bond_k, r0=bond_r0, delta=bond_delta, epsilon=0, sigma=0
            )

        all_combos = list(itertools.combinations(self.frame.particles.types, 2))
        all_same_pairs = [(p, p) for p in self.frame.particles.types]
        # LJ pair interactions of inter-chain particles
        for pair in all_combos:
            lj.params[pair] = dict(epsilon=pair_epsilon, sigma=pair_sigma)
            lj.r_cut[pair] = pair_r_cut
        # Same-chain particle pair interactions are turned off
        for pair in all_same_pairs:
            lj.params[pair] = dict(epsilon=0, sigma=0)
            lj.r_cut[pair] = 0
        return lj, bond
