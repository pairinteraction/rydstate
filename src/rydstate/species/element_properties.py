from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, ClassVar, overload

from rydstate.units import rydberg_constant, ureg

if TYPE_CHECKING:
    from rydstate.units import PintFloat


class ElementProperties:
    """Base class for all element properties classes.

    For the electronic ground state configurations and sorted shells,
    see e.g. https://www.webelements.com/atoms.html

    """

    species: ClassVar[str]
    """The short name of the atomic species."""
    Z: ClassVar[int]
    """Atomic number of the species."""
    i_c: ClassVar[float]
    """Nuclear spin."""
    number_valence_electrons: ClassVar[int]
    """Number of valence electrons (i.e. 1 for alkali atoms and 2 for alkaline earth atoms)."""

    corrected_rydberg_constant: tuple[float, str]
    r"""Corrected Rydberg constant stored as a tuple of the form (value, unit) for lazy unit conversion."""

    ground_state_shell: ClassVar[tuple[int, int]]
    """Shell (n, l) describing the electronic ground state configuration."""
    additional_allowed_shells: ClassVar[list[tuple[int, int]]] = []
    """Additional allowed shells (n, l), which (n, l) is smaller than the ground state shell."""
    core_electron_configuration: ClassVar[str]
    """Electron configuration of the core electrons, e.g. 4p6 for Rb or 5s for Sr."""

    nuclear_dipole: float
    """Nuclear dipole moment of the species."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __str__(self) -> str:
        return self.species

    @property
    def s_c(self) -> float:
        """Total spin of the core electrons (0 for alkali atoms, 0.5 for alkaline earth atoms)."""
        return 0.5 * (self.number_valence_electrons - 1)

    @overload
    def get_corrected_rydberg_constant(self, unit: None = None) -> PintFloat: ...

    @overload
    def get_corrected_rydberg_constant(self, unit: str) -> float: ...

    def get_corrected_rydberg_constant(self, unit: str | None = None) -> PintFloat | float:
        r"""Return the corrected Rydberg constant in the desired unit.

        The corrected Rydberg constant is defined as

        .. math::
            R_M = R_\infty \frac{m_{Core}}{m_{Core} + m_e}

        where :math:`R_\infty` is the Rydberg constant for infinite nuclear mass,
        :math:`m_{Core}` is the mass of the core,
        and :math:`m_e` is the mass of the electron.

        Args:
            unit: Desired unit for the corrected Rydberg constant. Default is atomic units "hartree".

        Returns:
            Corrected Rydberg constant in the desired unit.

        """
        corrected_rydberg_constant: PintFloat = ureg.Quantity(
            self.corrected_rydberg_constant[0], self.corrected_rydberg_constant[1]
        )
        corrected_rydberg_constant = corrected_rydberg_constant.to("hartree", "spectroscopy")
        if unit is None:
            return corrected_rydberg_constant
        if unit == "a.u.":
            return corrected_rydberg_constant.magnitude
        return corrected_rydberg_constant.to(unit, "spectroscopy").magnitude

    @cached_property  # don't remove this caching without benchmarking it!!!
    def reduced_mass_au(self) -> float:
        r"""The reduced mass mu in atomic units.

        The reduced mass in atomic units :math:`\mu / m_e` is given by

        .. math::
            \frac{\mu}{m_e} = \frac{m_{Core}}{m_{Core} + m_e}

        We calculate the reduced mass via the corrected Rydberg constant

        .. math::
            \frac{\mu}{m_e} = \frac{R_M}{R_\infty}

        """
        return self.get_corrected_rydberg_constant("hartree") / rydberg_constant.to("hartree").m
