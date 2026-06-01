from rydstate.species.potential import Potential


class PotentialHydrogen(Potential):
    species = "H"
    potential_type = "coulomb"
    is_default = True


class PotentialHydrogenTextBook(Potential):
    species = "H_textbook"
    potential_type = "coulomb"
    is_default = True
