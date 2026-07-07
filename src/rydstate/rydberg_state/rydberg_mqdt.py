from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

import numpy as np

from rydstate.angular import AngularKetFJ
from rydstate.rydberg_state.rydberg_base import RydbergState

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rydstate.angular import AngularState
    from rydstate.rydberg_state.rydberg_ket import RydbergKet
    from rydstate.species import MQDT, Potential
    from rydstate.units import NDArray


logger = logging.getLogger(__name__)


class RydbergStateMQDT(RydbergState):
    angular_state: AngularState[AngularKetFJ[Any]]
    """Return the angular part of the MQDT state as an AngularState."""

    def __init__(
        self,
        species: str,
        coefficients: Sequence[float] | NDArray,
        rydberg_kets: Sequence[RydbergKet],
        nu: float,
        energy_au: float,
        mqdt: MQDT,
        potential_class: type[Potential],
    ) -> None:
        self.mqdt = mqdt
        self.potential_class = potential_class

        if not all(isinstance(rydberg_ket.angular, AngularKetFJ) for rydberg_ket in rydberg_kets):
            raise ValueError("All rydberg_kets must have an angular part of type AngularKetFJ.")

        super().__init__(species, coefficients, rydberg_kets, nu, energy_au)

    @property
    def nui(self) -> NDArray:
        """Return the effective principal quantum numbers nui of the different channels."""
        return np.array([rydberg_ket.radial.nu for rydberg_ket in self.rydberg_kets])  # type: ignore [attr-defined]

    def calc_exp_qn(self, qn: str) -> float:
        if qn == "nui":
            coefficients2 = np.conjugate(self.coefficients) * self.coefficients / self.norm**2
            return float(np.sum(coefficients2 * self.nui))

        return super().calc_exp_qn(qn)

    def calc_std_qn(self, qn: str) -> float:
        if qn == "nui":
            coefficients2 = np.conjugate(self.coefficients) * self.coefficients / self.norm**2
            exp_q = np.sum(coefficients2 * self.nui)
            exp_q2 = np.sum(coefficients2 * self.nui * self.nui)
            if abs(exp_q2 - exp_q**2) < 1e-10:
                return 0
            return math.sqrt(exp_q2 - exp_q**2)

        return super().calc_std_qn(qn)
