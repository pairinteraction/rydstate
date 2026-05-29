from typing import ClassVar

from rydstate.species.element_properties import ElementProperties


class ElementPropertiesRubidium(ElementProperties):
    species = "Rb"

    Z = 37
    i_c = 0
    number_valence_electrons = 1
    ground_state_shell = (5, 0)
    _additional_allowed_shells: ClassVar = [(4, 2), (4, 3)]
    _core_electron_configuration = "4p6"

    # https://journals.aps.org/pra/pdf/10.1103/PhysRevA.83.052515
    _ionization_energy = (1_010_029.164_6, "GHz")
