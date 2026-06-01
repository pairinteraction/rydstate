from typing import ClassVar

from rydstate.species.element_properties import ElementProperties


class ElementPropertiesPotassium(ElementProperties):
    species = "K"

    Z = 19
    i_c = 0
    number_valence_electrons = 1
    ground_state_shell = (4, 0)
    additional_allowed_shells: ClassVar = [(3, 2)]
    core_electron_configuration = "3p6"

    corrected_rydberg_constant = (109735.774, "1/cm")
