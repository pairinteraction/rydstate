from __future__ import annotations

import logging
from abc import ABC
from typing import TYPE_CHECKING, Any, overload

import numpy as np

from rydstate.units import MatrixElementOperatorRanks

if TYPE_CHECKING:
    from rydstate.angular import AngularState
    from rydstate.angular.angular_ket import AngularKetBase
    from rydstate.rydberg_state.rydberg_ket import RydbergKet
    from rydstate.units import MatrixElementOperator, NDArray, PintFloat


logger = logging.getLogger(__name__)


class RydbergStateBase(ABC):
    angular: AngularState[Any] | AngularKetBase[Any]

    nu: float
    """The effective principal quantum number nu.

    For SQDT states, this is also sometimes called n*.
    For MQDT nu is given in reference to the lowest ionization threshold.
    """

    coefficients: NDArray
    """The coefficients of the Rydberg state in the basis of Rydberg kets."""

    rydberg_kets: list[RydbergKet]
    """The Rydberg kets that form the basis of the Rydberg state."""

    def calc_reduced_overlap(self, other: RydbergStateBase) -> float:
        """Calculate the reduced overlap <self|other> (ignoring the magnetic quantum number m)."""
        ov = 0.0
        for coeff1, sqdt1 in zip(self.coefficients, self.rydberg_kets, strict=True):
            for coeff2, sqdt2 in zip(other.coefficients, other.rydberg_kets, strict=True):
                ov += np.conjugate(coeff1) * coeff2 * sqdt1.calc_reduced_overlap(sqdt2)
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
        for coeff1, sqdt1 in zip(self.coefficients, self.rydberg_kets, strict=True):
            for coeff2, sqdt2 in zip(other.coefficients, other.rydberg_kets, strict=True):
                value += np.conjugate(coeff1) * coeff2 * sqdt1.calc_reduced_matrix_element(sqdt2, operator, unit=unit)
        return value

    @overload
    def calc_matrix_element(self, other: RydbergStateBase, operator: MatrixElementOperator, q: int) -> PintFloat: ...

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
        _k_radial, k_angular = MatrixElementOperatorRanks[operator]
        prefactor = self.rydberg_kets[0].angular._calc_wigner_eckart_prefactor(  # noqa: SLF001
            other.rydberg_kets[0].angular, k_angular, q
        )
        reduced_matrix_element = self.calc_reduced_matrix_element(other, operator, unit)
        return prefactor * reduced_matrix_element
