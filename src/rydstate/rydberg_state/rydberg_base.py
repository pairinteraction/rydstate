from __future__ import annotations

import logging
from abc import ABC
from typing import TYPE_CHECKING, Any, overload

import numpy as np

from rydstate.units import BaseQuantities

if TYPE_CHECKING:
    from rydstate.angular.angular_ket import AngularKetBase
    from rydstate.angular.angular_state import AngularState
    from rydstate.rydberg_state.rydberg_ket import RydbergKet
    from rydstate.units import MatrixElementOperator, NDArray, PintFloat


logger = logging.getLogger(__name__)


class RydbergStateBase(ABC):
    species: str
    """The species of the Rydberg state."""

    coefficients: NDArray
    """The channel coefficients of the different Rydberg ket channels that form the Rydberg state."""
    rydberg_kets: list[RydbergKet]
    """The Rydberg kets that form the Rydberg state."""
    angular: AngularKetBase[Any] | AngularState[Any]
    """The angular part of the Rydberg state, i.e. the radial part is traced out."""

    nu: float
    """The effective principal quantum number nu.
    For SQDT states, this is also sometimes called n*.
    For MQDT nu is given in reference to the lowest ionization threshold.
    """
    _energy_au: float
    """The energy of the Rydberg state in atomic units (Hartree)."""

    @overload
    def get_energy(self, unit: None = None) -> PintFloat: ...

    @overload
    def get_energy(self, unit: str) -> float: ...

    def get_energy(self, unit: str | None = None) -> PintFloat | float:
        r"""Get the energy of the Rydberg state.

        The energy is defined as

        .. math::
            E = - \frac{1}{2} \frac{\mu}{\nu^2} + E_{ionization}

        where `\mu = R_M/R_\infty` is the reduced mass and `\nu` the effective principal quantum number,
        and `E_{ionization}` is the (reference) ionization energy of the species.
        """
        if unit == "a.u.":
            return self._energy_au
        energy: PintFloat = self._energy_au * BaseQuantities["energy"]
        if unit is None:
            return energy
        return energy.to(unit, "spectroscopy").magnitude

    def calc_reduced_overlap(self, other: RydbergStateBase) -> float:
        """Calculate the reduced overlap <self|other> (ignoring the magnetic quantum number m)."""
        ov = 0.0
        for coeff1, ket1 in zip(self.coefficients, self.rydberg_kets, strict=True):
            for coeff2, ket2 in zip(other.coefficients, other.rydberg_kets, strict=True):
                ov += np.conjugate(coeff1) * coeff2 * ket1.calc_reduced_overlap(ket2)
        return ov

    @overload
    def calc_reduced_matrix_element(
        self, other: RydbergStateBase, operator: MatrixElementOperator, unit: None = None
    ) -> PintFloat: ...

    @overload
    def calc_reduced_matrix_element(
        self, other: RydbergStateBase, operator: MatrixElementOperator, unit: str
    ) -> float: ...

    def calc_reduced_matrix_element(
        self, other: RydbergStateBase, operator: MatrixElementOperator, unit: str | None = None
    ) -> PintFloat | float:
        r"""Calculate the reduced matrix element.

        Calculate the reduced matrix element between self and other (ignoring m quantum numbers)

        .. math::
            \left\langle self || r^k_radial \hat{O}_{k_angular} || other \right\rangle

        where \hat{O}_{k_angular} is the operator of rank k_angular for which to calculate the matrix element.
        k_radial and k_angular are determined from the operator automatically.

        Args:
            other: The other Rydberg state for which to calculate the matrix element.
            operator: The operator for which to calculate the matrix element.
            unit: The unit to which to convert the radial matrix element.
                Can be "a.u." for atomic units (so no conversion is done), or a specific unit.
                Default None will return a pint quantity.

        Returns:
            The reduced matrix element for the given operator.

        """
        value = 0.0
        for coeff1, ket1 in zip(self.coefficients, self.rydberg_kets, strict=True):
            for coeff2, ket2 in zip(other.coefficients, other.rydberg_kets, strict=True):
                value += np.conjugate(coeff1) * coeff2 * ket1.calc_reduced_matrix_element(ket2, operator, unit=unit)
        return value

    @overload
    def calc_matrix_element(
        self, other: RydbergStateBase, operator: MatrixElementOperator, q: int, unit: None = None
    ) -> PintFloat: ...

    @overload
    def calc_matrix_element(
        self, other: RydbergStateBase, operator: MatrixElementOperator, q: int, unit: str
    ) -> float: ...

    def calc_matrix_element(
        self, other: RydbergStateBase, operator: MatrixElementOperator, q: int, unit: str | None = None
    ) -> PintFloat | float:
        r"""Calculate the matrix element.

        Calculate the full matrix element between self and other,
        also considering the magnetic quantum numbers m of self and other.

        .. math::
            \left\langle self || r^k_radial \hat{O}_{k_angular} || other \right\rangle

        where \hat{O}_{k_angular} is the operator of rank k_angular for which to calculate the matrix element.
        k_radial and k_angular are determined from the operator automatically.

        Args:
            other: The other Rydberg state for which to calculate the matrix element.
            operator: The operator for which to calculate the matrix element.
            q: The component of the operator.
            unit: The unit to which to convert the radial matrix element.
                Can be "a.u." for atomic units (so no conversion is done), or a specific unit.
                Default None will return a pint quantity.

        Returns:
            The matrix element for the given operator.

        """
        value = 0.0
        for coeff1, ket1 in zip(self.coefficients, self.rydberg_kets, strict=True):
            for coeff2, ket2 in zip(other.coefficients, other.rydberg_kets, strict=True):
                value += np.conjugate(coeff1) * coeff2 * ket1.calc_matrix_element(ket2, operator, q=q, unit=unit)
        return value
