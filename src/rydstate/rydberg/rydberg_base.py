from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from rydstate.angular import AngularState
from rydstate.angular.angular_ket import AngularKetBase

if TYPE_CHECKING:

    from rydstate.units import MatrixElementOperator


logger = logging.getLogger(__name__)


class RydbergStateBase(ABC):
    @property
    @abstractmethod
    def angular(self) -> AngularState[Any] | AngularKetBase: ...

    @abstractmethod
    def calc_reduced_overlap(self, other: RydbergStateBase) -> float: ...

    @abstractmethod
    def calc_reduced_matrix_element(
        self, other: RydbergStateBase, operator: MatrixElementOperator, unit: str | None = None
    ) -> float: ...
