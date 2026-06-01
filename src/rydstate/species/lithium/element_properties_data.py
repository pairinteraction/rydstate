from rydstate.species.element_properties import ElementProperties


class ElementPropertiesLithium(ElementProperties):
    species = "Li"

    Z = 3
    i_c = 0
    number_valence_electrons = 1
    ground_state_shell = (2, 0)
    core_electron_configuration = "1s2"

    corrected_rydberg_constant = (109728.64, "1/cm")
