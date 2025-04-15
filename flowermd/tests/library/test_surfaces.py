import gsd.hoomd
import numpy as np

from flowermd.library import OPLS_AA, Graphene


class TestSurfaces:
    def test_create_graphene_surface(self):
        surface = Graphene(
            x_repeat=2,
            y_repeat=2,
            n_layers=3,
            periodicity=(False, False, False),
        )
        assert surface.system.n_particles == 4 * 2 * 2 * 3
        assert surface.system.n_bonds == 54
        graphene_sheet = surface.system
        for bond in graphene_sheet.bonds():
            bond_L = np.linalg.norm(bond[1].pos - bond[0].pos)
            assert np.round(bond_L, 3) == 0.142

    def test_graphene_with_periodicity(self):
        surface = Graphene(
            x_repeat=2, y_repeat=2, n_layers=3, periodicity=(True, True, False)
        )
        assert surface.system.n_particles == 4 * 2 * 2 * 3
        assert surface.system.periodicity == (True, True, False)
        assert surface.system.n_bonds == 72

    def test_graphene_apply_ff(self):
        surface = Graphene(
            x_repeat=2, y_repeat=2, n_layers=2, periodicity=(True, True, False)
        )
        surface.apply_forcefield(
            r_cut=2.5,
            force_field=OPLS_AA(),
            auto_scale=True,
            scale_charges=True,
            remove_charges=False,
            remove_hydrogens=False,
        )
        assert isinstance(surface.hoomd_snapshot, gsd.hoomd.Frame)
