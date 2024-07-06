import numpy as np

from flowermd.base.system import Lattice
from flowermd.library import OPLS_AA_PPS, PPS, Tensile
from flowermd.tests import BaseTest


class TestTensileSimulation(BaseTest):
    def test_tensile(self):
        pps = PPS(lengths=6, num_mols=32)
        system = Lattice(
            molecules=[pps],
            x=1.2,
            y=1.2,
            n=4,
        )
        system.apply_forcefield(
            r_cut=2.5, force_field=[OPLS_AA_PPS()], auto_scale=True
        )

        tensile_sim = Tensile(
            initial_state=system.hoomd_snapshot,
            forcefield=system.hoomd_forcefield,
            tensile_axis=(1, 0, 0),
            log_write_freq=1e6,
            gsd_write_freq=1e6,
        )
        tensile_sim.run_tensile(
            strain=0.05, kT=2.0, n_steps=1e3, period=10, tau_kt=0.001
        )
        assert np.allclose(tensile_sim.strain, 0.05, 1e-4)

    def test_log_data(self):
        pps = PPS(lengths=6, num_mols=32)
        system = Lattice(
            molecules=[pps],
            x=1.2,
            y=1.2,
            n=4,
        )
        system.apply_forcefield(
            r_cut=2.5, force_field=[OPLS_AA_PPS()], auto_scale=True
        )

        tensile_sim = Tensile(
            initial_state=system.hoomd_snapshot,
            forcefield=system.hoomd_forcefield,
            tensile_axis=(1, 0, 0),
            log_write_freq=10,
            gsd_write_freq=1e6,
        )
        tensile_sim.run_tensile(
            strain=0.05, kT=2.0, n_steps=1e3, period=100, tau_kt=0.001
        )
        assert len(tensile_sim._strain_logs) == 1
        assert len(tensile_sim._stress_logs) == 1
        assert len(tensile_sim._strain_logs[0][1:]) == 100
        strain_data = tensile_sim.strain_data
        unique_strains = np.unique(strain_data)
        dstrain = unique_strains[-1] - unique_strains[-2]
        assert np.allclose(strain_data[0], 0.0, atol=1e-5)
        assert np.allclose(strain_data[-1], 0.05 - dstrain, atol=1e-5)
