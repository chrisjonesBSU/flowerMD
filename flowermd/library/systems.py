"""Library of Systems."""

import unyt as u

from flowermd.base import System


class PolymerRandomWalk(System):
    def __init__(self, molecules, density, base_units, neighbor_buffer):
        self.neighbor_buffer = neighbor_buffer
        if not isinstance(density, u.array.unyt_quantity):
            self.density = density * u.Unit("g") / u.Unit("cm**3")
            warnings.warn(
                "Units for density were not given, assuming "
                "units of g/cm**3."
            )
        else:
            self.density = density

    def _build_system(self):
        mass_density = u.Unit("kg") / u.Unit("m**3")
        number_density = u.Unit("m**-3")
        if self.density.units.dimensions == mass_density.dimensions:
            target_box = get_target_box_mass_density(
                density=self.density, mass=self.mass
            ).to("nm")
        elif self.density.units.dimensions == number_density.dimensions:
            target_box = get_target_box_number_density(
                density=self.density, n_beads=self.n_particles
            ).to("nm")
        else:
            raise ValueError(
                f"Density dimensions of {self.density.units.dimensions} "
                "were given, but only mass density "
                f"({mass_density.dimensions}) and "
                f"number density ({number_density.dimensions}) are supported."
            )
        chains = []
        monomers = []
        monomer_positions = None 
        
        chain_count = 0
        for mol in self.molecules:
            for N, L in zip(mol.n_mols, mol.lengths):
                # i is the ith molecule
                monomer_positions = np.zeros(N * L)
                chain_monomer_positions = np.zeros(L) 
                for i in range(N):
                    # j is the jth monomer of the ith molecule
                    start_pos = np.random.rand(3) * target_box[0]
                    chain_monomer_positions
                    for j in range(1, L):








