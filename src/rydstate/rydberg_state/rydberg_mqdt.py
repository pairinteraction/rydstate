from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from rydstate.angular import AngularKetFJ, AngularState
from rydstate.rydberg_state.rydberg_base import RydbergStateBase

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rydstate.rydberg_state.rydberg_ket import RydbergKet
    from rydstate.units import NDArray


logger = logging.getLogger(__name__)


class RydbergStateMQDT(RydbergStateBase):
    angular: AngularState[AngularKetFJ[Any]]
    """Return the angular part of the MQDT state as an AngularState."""

    def __init__(
        self,
        coefficients: Sequence[float] | NDArray,
        rydberg_kets: Sequence[RydbergKet],
        nu: float,
        *,
        warn_if_not_normalized: bool = True,
        normalize: bool = True,
    ) -> None:
        self.coefficients = np.array(coefficients)
        self.rydberg_kets = list(rydberg_kets)
        self._nu = nu

        if len(rydberg_kets) == 0:
            raise ValueError("RydbergStateMQDT must be initialized with at least one state.")
        if len(coefficients) != len(rydberg_kets):
            raise ValueError("Length of coefficients and rydberg_kets must be the same.")
        if not all(isinstance(rydberg_ket.angular, AngularKetFJ) for rydberg_ket in rydberg_kets):
            raise ValueError("All rydberg_kets must have an angular part of type AngularKetFJ.")
        if not all((rydberg_ket.species is rydberg_kets[0].species) for rydberg_ket in rydberg_kets):
            raise ValueError("All rydberg_kets must be of the same species.")
        if len(set(rydberg_kets)) != len(rydberg_kets):
            raise ValueError("RydbergStateMQDT initialized with duplicate rydberg_kets.")

        if abs(self.norm - 1) > 1e-10 and warn_if_not_normalized:
            logger.warning(
                "RydbergStateMQDT initialized with non-normalized coefficients: %s, %s", coefficients, rydberg_kets
            )
        if normalize:
            self.coefficients /= self.norm

        self.species = rydberg_kets[0].species
        self.angular = AngularState(
            self.coefficients.tolist(),
            [ket.angular for ket in rydberg_kets],  # type: ignore [misc]
            normalize=False,
            warn_if_not_normalized=False,
        )

    def __repr__(self) -> str:
        terms = [
            f"{coeff}*{rydberg_ket!r}" for coeff, rydberg_ket in zip(self.coefficients, self.rydberg_kets, strict=True)
        ]
        return f"{self.__class__.__name__}({', '.join(terms)})"

    def __str__(self) -> str:
        terms = [
            f"{coeff}*{rydberg_ket!s}" for coeff, rydberg_ket in zip(self.coefficients, self.rydberg_kets, strict=True)
        ]
        return f"{', '.join(terms)}"

    @property
    def norm(self) -> float:
        """Return the norm of the state (should be 1)."""
        return float(np.linalg.norm(self.coefficients))
