from abc import ABC
from typing import ClassVar

from rydstate.species.element_properties import ElementProperties
from rydstate.units import electron_mass, rydberg_constant


class _ElementPropertiesYtterbiumAbstract(ElementProperties, ABC):
    Z = 70
    number_valence_electrons = 2
    ground_state_shell = (6, 0)
    additional_allowed_shells: ClassVar = [(5, 2), (5, 3), (5, 4)]
    core_electron_configuration = "4f14.6s"


class ElementPropertiesYtterbium171(_ElementPropertiesYtterbiumAbstract):
    species = "Yb171"
    i_c = 1 / 2

    _isotope_mass_u = 170.936323
    corrected_rydberg_constant = (
        rydberg_constant.m / (1 + electron_mass.to("u").m / _isotope_mass_u),
        str(rydberg_constant.u),
    )


class ElementPropertiesYtterbium173(_ElementPropertiesYtterbiumAbstract):
    species = "Yb173"
    i_c = 5 / 2

    _isotope_mass_u = 172.938208
    corrected_rydberg_constant = (
        rydberg_constant.m / (1 + electron_mass.to("u").m / _isotope_mass_u),
        str(rydberg_constant.u),
    )


class ElementPropertiesYtterbium174(_ElementPropertiesYtterbiumAbstract):
    species = "Yb174"
    i_c = 0

    _isotope_mass_u = 173.938859
    corrected_rydberg_constant = (
        rydberg_constant.m / (1 + electron_mass.to("u").m / _isotope_mass_u),
        str(rydberg_constant.u),
    )
