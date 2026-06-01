from typing import ClassVar

from rydstate.species.potential import PotentialFei2009, PotentialMarinescu1993


class _PotentialFei2009StrontiumAbstract(PotentialFei2009):
    is_default = True

    model_potential_parameter_fei_2009 = (0.9959, 16.9567, 0.2648, 0.1439)


class PotentialFei2009Strontium87(_PotentialFei2009StrontiumAbstract):
    species = "Sr87"


class PotentialFei2009Strontium88(_PotentialFei2009StrontiumAbstract):
    species = "Sr88"


class _PotentialMarinescu1993StrontiumAbstract(PotentialMarinescu1993):
    alpha_c_marinescu_1993 = 7.5
    r_c_dict_marinescu_1993: ClassVar = {0: 1.59, 1: 1.58, 2: 1.57, 3: 1.56}
    model_potential_parameter_marinescu_1993: ClassVar = {
        0: (3.36124, 1.3337, 0, 5.94337),
        1: (3.28205, 1.24035, 0, 3.78861),
        2: (2.155, 1.4545, 0, 4.5111),
        3: (2.1547, 1.14099, 0, 2.1987),
    }


class PotentialMarinescu1993Strontium87(_PotentialMarinescu1993StrontiumAbstract):
    species = "Sr87"


class PotentialMarinescu1993Strontium88(_PotentialMarinescu1993StrontiumAbstract):
    species = "Sr88"
