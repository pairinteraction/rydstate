from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import numpy as np

from rydstate.angular import AngularState
from rydstate.rydberg.rydberg_sqdt import RydbergStateSQDT

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from typing_extensions import Self

    from rydstate.units import MatrixElementOperator


logger = logging.getLogger(__name__)


_RydbergState = TypeVar("_RydbergState", bound=RydbergStateSQDT)


class RydbergStateMQDT(Generic[_RydbergState]):
    def __init__(
        self,
        coefficients: Sequence[float],
        sqdt_states: Sequence[_RydbergState],
        *,
        nu_energy: float | None = None,
        warn_if_not_normalized: bool = True,
    ) -> None:
        self.coefficients = np.array(coefficients)
        self.sqdt_states = sqdt_states
        self.nu_energy = nu_energy

        if len(coefficients) != len(sqdt_states):
            raise ValueError("Length of coefficients and sqdt_states must be the same.")
        if not all(type(sqdt_state) is type(sqdt_states[0]) for sqdt_state in sqdt_states):
            raise ValueError("All sqdt_states must be of the same type.")
        if len(set(sqdt_states)) != len(sqdt_states):
            raise ValueError("RydbergStateMQDT initialized with duplicate sqdt_states.")
        if abs(self.norm - 1) > 1e-10 and warn_if_not_normalized:
            logger.warning(
                "RydbergStateMQDT initialized with non-normalized coefficients (norm=%s, coefficients=%s, sqdt_states=%s)",
                self.norm,
                coefficients,
                sqdt_states,
            )
        if self.norm > 1:
            self.coefficients /= self.norm

    def __iter__(self) -> Iterator[tuple[float, _RydbergState]]:
        return zip(self.coefficients, self.sqdt_states).__iter__()

    def __repr__(self) -> str:
        terms = [f"{coeff}*{sqdt_state!r}" for coeff, sqdt_state in self]
        return f"{self.__class__.__name__}({', '.join(terms)})"

    def __str__(self) -> str:
        terms = [f"{coeff}*{sqdt_state!s}" for coeff, sqdt_state in self]
        return f"{', '.join(terms)}"

    @property
    def norm(self) -> float:
        """Return the norm of the state (should be 1)."""
        return np.linalg.norm(self.coefficients)  # type: ignore [return-value]

    @property
    def angular(self) -> AngularState[Any]:
        """Return the angular part of the MQDT state as an AngularState."""
        angular_kets = [ket.angular for ket in self.sqdt_states]
        return AngularState(self.coefficients.tolist(), angular_kets)

    def calc_reduced_overlap(self, other: RydbergStateMQDT[Any] | RydbergStateSQDT) -> float:
        """Calculate the reduced overlap <self|other> (ignoring the magnetic quantum number m)."""
        if isinstance(other, RydbergStateSQDT):
            other = RydbergStateMQDT([1.0], [other])

        ov = 0
        for coeff1, sqdt1 in self:
            for coeff2, sqdt2 in other:
                ov += np.conjugate(coeff1) * coeff2 * sqdt1.calc_reduced_overlap(sqdt2)
        return ov

    def calc_matrix_element(
        self: Self, other: RydbergStateMQDT[Any] | RydbergStateSQDT, operator: MatrixElementOperator, kappa: int
    ) -> float:
        r"""Calculate the reduced angular matrix element.

        This means, calculate the following matrix element:

        .. math::
            \left\langle self || \hat{O}^{(\kappa)} || other \right\rangle

        """
        if isinstance(other, RydbergStateSQDT):
            other = RydbergStateMQDT([1.0], [other])

        value = 0
        for coeff1, sqdt1 in self:
            for coeff2, sqdt2 in other:
                value += np.conjugate(coeff1) * coeff2 * sqdt1.calc_matrix_element(sqdt2, operator, kappa)
        return value
