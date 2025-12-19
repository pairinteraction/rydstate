from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import numpy as np

from rydstate.angular import AngularState
from rydstate.rydberg.rydberg_sqdt import RydbergStateSQDTBase

if TYPE_CHECKING:
    from collections.abc import Iterator

    from typing_extensions import Self

    from rydstate.units import MatrixElementOperator


logger = logging.getLogger(__name__)


_RydbergState = TypeVar("_RydbergState", bound=RydbergStateSQDTBase)


class MQDTState(Generic[_RydbergState]):
    def __init__(
        self, coefficients: list[float], kets: list[_RydbergState], *, warn_if_not_normalized: bool = True
    ) -> None:
        self.coefficients = np.array(coefficients)
        self.kets = kets

        if len(coefficients) != len(kets):
            raise ValueError("Length of coefficients and kets must be the same.")
        if not all(type(ket) is type(kets[0]) for ket in kets):
            raise ValueError("All kets must be of the same type.")
        if len(set(kets)) != len(kets):
            raise ValueError("MQDTState initialized with duplicate kets.")
        if abs(self.norm - 1) > 1e-10 and warn_if_not_normalized:
            logger.warning(
                "MQDTState initialized with non-normalized coefficients (norm=%s, coefficients=%s, kets=%s)",
                self.norm,
                coefficients,
                kets,
            )
        if self.norm > 1:
            self.coefficients /= self.norm

    def __iter__(self) -> Iterator[tuple[float, _RydbergState]]:
        return zip(self.coefficients, self.kets).__iter__()

    def __repr__(self) -> str:
        terms = [f"{coeff}*{ket!r}" for coeff, ket in self]
        return f"{self.__class__.__name__}({', '.join(terms)})"

    def __str__(self) -> str:
        terms = [f"{coeff}*{ket!s}" for coeff, ket in self]
        return f"{', '.join(terms)}"

    @property
    def norm(self) -> float:
        """Return the norm of the state (should be 1)."""
        return np.linalg.norm(self.coefficients)  # type: ignore [return-value]

    @property
    def angular(self) -> AngularState[Any]:
        """Return the angular part of the MQDT state as an AngularState."""
        angular_kets = [ket.angular for ket in self.kets]
        return AngularState(self.coefficients, angular_kets)

    def calc_reduced_overlap(self, other: MQDTState[Any] | RydbergStateSQDTBase) -> float:
        """Calculate the reduced overlap <self|other> (ignoring the magnetic quantum number m)."""
        if isinstance(other, RydbergStateSQDTBase):
            other = MQDTState([1.0], [other])

        ov = 0
        for coeff1, ket1 in self:
            for coeff2, ket2 in other:
                ov += np.conjugate(coeff1) * coeff2 * ket1.calc_reduced_overlap(ket2)
        return ov

    def calc_matrix_element(
        self: Self, other: MQDTState[Any] | RydbergStateSQDTBase, operator: MatrixElementOperator, kappa: int
    ) -> float:
        r"""Calculate the reduced angular matrix element.

        This means, calculate the following matrix element:

        .. math::
            \left\langle self || \hat{O}^{(\kappa)} || other \right\rangle

        """
        if isinstance(other, RydbergStateSQDTBase):
            other = MQDTState([1.0], [other])

        value = 0
        for coeff1, ket1 in self:
            for coeff2, ket2 in other:
                value += np.conjugate(coeff1) * coeff2 * ket1.calc_matrix_element(ket2, operator, kappa)
        return value
