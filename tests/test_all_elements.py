from typing import TYPE_CHECKING

import pytest
from rydstate import RydbergStateSQDTAlkali, RydbergStateSQDTAlkalineLS
from rydstate.species import ElementProperties, get_all_subclasses, get_element_properties, get_sqdt_class

if TYPE_CHECKING:
    from rydstate import RydbergStateSQDT
    from rydstate.angular.angular_ket import AngularKetLS
    from rydstate.angular.utils import AllKnown


ALL_AVAILABLE_SPECIES = [cls.species for cls in get_all_subclasses(ElementProperties)]


@pytest.mark.parametrize("species", ALL_AVAILABLE_SPECIES)
def test_sqdt_species(species: str) -> None:
    element_properties = get_element_properties(species)
    sqdt = get_sqdt_class(species)()
    i_c = element_properties.i_c

    state: RydbergStateSQDT[AngularKetLS[AllKnown]]
    if element_properties.number_valence_electrons == 1:
        state = RydbergStateSQDTAlkali(species, n=50, l=0, f=i_c + 0.5)
        state.radial.create_wavefunction()
        with pytest.raises(ValueError, match="Invalid combination of angular quantum numbers provided"):
            RydbergStateSQDTAlkali(species, n=50, l=1)
    elif element_properties.number_valence_electrons == 2 and sqdt.quantum_defects is not None:
        for s_tot in [0, 1]:
            state = RydbergStateSQDTAlkalineLS(species, n=50, l=1, s_tot=s_tot, j_tot=1 + s_tot, f_tot=s_tot + 1 + i_c)
            state.radial.create_wavefunction()
