from typing import ClassVar

from rydstate.species.element_properties import ElementProperties


class ElementPropertiesCesium(ElementProperties):
    species = "Cs"

    Z = 55
    i_c = 0
    number_valence_electrons = 1
    ground_state_shell = (6, 0)
    additional_allowed_shells: ClassVar = [(4, 3), (5, 2), (5, 3), (5, 4)]
    core_electron_configuration = "5p6"

    corrected_rydberg_constant = (109736.8627339, "1/cm")
