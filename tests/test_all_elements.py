from typing import TYPE_CHECKING

import pytest
from rydstate import RydbergStateSQDT
from rydstate.species import SpeciesObjectSQDT

if TYPE_CHECKING:
    from rydstate.angular.angular_ket import AngularKetLS
    from rydstate.angular.utils import AllKnown


@pytest.mark.parametrize("species_name", SpeciesObjectSQDT.get_available_species())
def test_sqdt_species(species_name: str) -> None:
    species = SpeciesObjectSQDT.from_name(species_name)
    i_c = species.i_c_number

    state: RydbergStateSQDT[AngularKetLS[AllKnown]]
    if species.number_valence_electrons == 1:
        state = RydbergStateSQDT(species, n=50, l_r=0, f_tot=i_c + 0.5)
        state.radial.create_wavefunction()
        with pytest.raises(ValueError, match="Invalid combination of angular quantum numbers provided"):
            RydbergStateSQDT(species, n=50, l_r=1)
    elif species.number_valence_electrons == 2 and species._quantum_defects is not None:  # noqa: SLF001
        for s_tot in [0, 1]:
            state = RydbergStateSQDT(species, n=50, l_r=1, s_tot=s_tot, j_tot=1 + s_tot, f_tot=s_tot + 1 + i_c)
            state.radial.create_wavefunction()
