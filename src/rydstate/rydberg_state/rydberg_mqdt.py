from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

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
