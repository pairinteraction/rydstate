from rydstate.species.element_properties import ElementProperties


class ElementPropertiesSodium(ElementProperties):
    species = "Na"

    Z = 11
    i_c = 0
    number_valence_electrons = 1
    ground_state_shell = (3, 0)
    core_electron_configuration = "2p6"

    corrected_rydberg_constant = (109734.69, "1/cm")
