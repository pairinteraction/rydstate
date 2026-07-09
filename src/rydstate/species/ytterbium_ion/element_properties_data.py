from abc import ABC
from typing import ClassVar

from rydstate.species.element_properties import ElementProperties
from rydstate.units import electron_mass, rydberg_constant


class _ElementPropertiesYtterbiumAbstractIon(ElementProperties, ABC):
    Z = 70
    net_charge = 2
    number_valence_electrons = 1
    ground_state_shell = (6, 0)
    additional_allowed_shells: ClassVar = [(5, 2), (5, 3), (5, 4)]
    core_electron_configuration = "4f14"


class ElementPropertiesYtterbium171Ion(_ElementPropertiesYtterbiumAbstractIon):
    species = "Yb171_ion"
    i_c = 1 / 2

    _isotope_mass_u = 170.9363258
    corrected_rydberg_constant = (
        rydberg_constant.m / (1 + electron_mass.to("u").m / _isotope_mass_u),
        str(rydberg_constant.u),
    )

    nuclear_dipole = 0.49367


class ElementPropertiesYtterbium173Ion(_ElementPropertiesYtterbiumAbstractIon):
    species = "Yb173_ion"
    i_c = 5 / 2

    _isotope_mass_u = 172.938216212
    corrected_rydberg_constant = (
        rydberg_constant.m / (1 + electron_mass.to("u").m / _isotope_mass_u),
        str(rydberg_constant.u),
    )

    nuclear_dipole = -0.68


class ElementPropertiesYtterbium174Ion(_ElementPropertiesYtterbiumAbstractIon):
    species = "Yb174_ion"
    i_c = 0

    _isotope_mass_u = 173.938859
    corrected_rydberg_constant = (
        rydberg_constant.m / (1 + electron_mass.to("u").m / _isotope_mass_u),
        str(rydberg_constant.u),
    )

    nuclear_dipole = 2.1
