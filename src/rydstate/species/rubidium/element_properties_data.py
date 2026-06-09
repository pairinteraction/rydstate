from typing import ClassVar

from rydstate.species.element_properties import ElementProperties


class ElementPropertiesRubidium(ElementProperties):
    species = "Rb"

    Z = 37
    i_c = 0
    number_valence_electrons = 1
    ground_state_shell = (5, 0)
    additional_allowed_shells: ClassVar = [(4, 2), (4, 3)]
    core_electron_configuration = "4p6"

    corrected_rydberg_constant = (109736.62301604665, "1/cm")
