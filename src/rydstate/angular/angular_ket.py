from __future__ import annotations

import logging
from abc import ABC
from typing import TYPE_CHECKING

from rydstate.angular.angular_ket_base import AngularKetBase, AngularKetBaseFJ, AngularKetBaseJJ, AngularKetBaseLS
from rydstate.angular.utils import (
    is_unknown,
)

if TYPE_CHECKING:
    from rydstate.angular.utils import AngularMomentumQuantumNumbers

logger = logging.getLogger(__name__)


class AngularKet(AngularKetBase, ABC):
    """Base class for a angular ket where no quantum numbers are unknown (i.e. a simple canonical spin ketstate)."""

    i_c: float
    s_c: float
    l_c: int
    s_r: float
    l_r: int

    def sanity_check(self, msgs: list[str] | None = None) -> None:
        """Check that the quantum numbers are valid."""
        msgs = msgs if msgs is not None else []

        if any(is_unknown(qn) for qn in self.quantum_numbers):
            cls_name = type(self).__name__
            cls_base_name = cls_name.replace("AngularKet", "AngularKetBase")
            raise ValueError(f"Unknown quantum numbers are not allowed for {cls_name}, use {cls_base_name} instead.")

        super().sanity_check(msgs)

    def get_qn(self, qn: AngularMomentumQuantumNumbers) -> float:
        qn_value = super().get_qn(qn)
        if is_unknown(qn_value):
            raise ValueError(f"Quantum number {qn} is unknown for {self!r}.")
        return qn_value

    def calc_exp_qn(self, qn: AngularMomentumQuantumNumbers) -> float:
        exp = super().calc_exp_qn(qn)
        if is_unknown(exp):
            raise RuntimeError("This should never happen, since all quantum numbers are known for AngularKet.")
        return exp

    def calc_std_qn(self, qn: AngularMomentumQuantumNumbers) -> float:
        std = super().calc_std_qn(qn)
        if is_unknown(std):
            raise RuntimeError("This should never happen, since all quantum numbers are known for AngularKet.")
        return std


class AngularKetLS(AngularKet, AngularKetBaseLS):
    """Spin ket in LS coupling."""

    s_tot: float
    l_tot: int
    j_tot: float


class AngularKetJJ(AngularKet, AngularKetBaseJJ):
    """Spin ket in JJ coupling."""

    j_c: float
    j_r: float
    j_tot: float


class AngularKetFJ(AngularKet, AngularKetBaseFJ):
    """Spin ket in FJ coupling."""

    j_c: float
    f_c: float
    j_r: float
