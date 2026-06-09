from rydstate.species.element_properties import ElementProperties


class ElementPropertiesHydrogen(ElementProperties):
    species = "H"

    Z = 1
    i_c = 0
    number_valence_electrons = 1
    ground_state_shell = (1, 0)
    core_electron_configuration = "1s0"

    corrected_rydberg_constant = (109677.58340280356, "1/cm")


class ElementPropertiesHydrogenTextBook(ElementProperties):
    species = "H_textbook"

    Z = 1
    i_c = 0
    number_valence_electrons = 1
    ground_state_shell = (1, 0)
    core_electron_configuration = "1s0"

    corrected_rydberg_constant = (109737.31568160003, "1/cm")
