from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar, overload

import numpy as np

from rydstate.angular import NotSet
from rydstate.angular.angular_ket import AngularKetBase, AngularKetFJ, AngularKetJJ, AngularKetLS
from rydstate.angular.utils import is_not_set
from rydstate.basis.basis_base import BasisBase
from rydstate.rydberg_state import RydbergStateSQDT
from rydstate.species import SQDT
from rydstate.species.utils import get_subclass

if TYPE_CHECKING:
    from rydstate.angular.utils import AllKnown, CouplingScheme

T_AngularKet = TypeVar("T_AngularKet", bound=AngularKetBase[Any])

logger = logging.getLogger(__name__)


class BasisSQDT(BasisBase[RydbergStateSQDT[T_AngularKet]], Generic[T_AngularKet]):
    states: list[RydbergStateSQDT[T_AngularKet]]

    @overload
    def __init__(
        self: BasisSQDT[AngularKetLS[AllKnown]],
        species: str,
        n: tuple[int, int],
        m: tuple[float, float] | None | NotSet = NotSet,
        *,
        coupling_scheme: Literal["LS"] = "LS",
    ) -> None: ...

    @overload
    def __init__(
        self: BasisSQDT[AngularKetJJ[AllKnown]],
        species: str,
        n: tuple[int, int],
        m: tuple[float, float] | None | NotSet = NotSet,
        *,
        coupling_scheme: Literal["JJ"],
    ) -> None: ...

    @overload
    def __init__(
        self: BasisSQDT[AngularKetFJ[AllKnown]],
        species: str,
        n: tuple[int, int],
        m: tuple[float, float] | None | NotSet = NotSet,
        *,
        coupling_scheme: Literal["FJ"],
    ) -> None: ...

    def __init__(
        self,
        species: str,
        n: tuple[int, int],
        m: tuple[float, float] | None | NotSet = NotSet,
        *,
        coupling_scheme: CouplingScheme = "LS",
    ) -> None:
        super().__init__(species)
        self.sqdt = get_subclass(SQDT, species)()

        self._init_states(n, m, coupling_scheme)

    def _init_states(
        self, n_range: tuple[int, int], m_range: tuple[float, float] | None | NotSet, coupling_scheme: CouplingScheme
    ) -> None:
        self.coupling_scheme = coupling_scheme
        self.states = []

        if coupling_scheme == "LS":
            add_states = self._add_states_ls
        elif coupling_scheme == "JJ":
            add_states = self._add_states_jj
        elif coupling_scheme == "FJ":
            add_states = self._add_states_fj
        else:
            raise ValueError(f"Invalid coupling scheme: {coupling_scheme}")

        for n in range(n_range[0], n_range[1] + 1):
            for l_r in range(n):
                add_states(n, l_r, m_range)

    def _add_states_ls(self, n: int, l_r: int, m_range: tuple[float, float] | None | NotSet = NotSet) -> None:
        i_c = self.element_properties.i_c
        s_r = 0.5
        s_c = self.element_properties.s_c
        s_tot_list = np.arange(s_r - s_c, s_r + s_c + 1)

        for s_tot in s_tot_list:
            if not self.sqdt.is_allowed_shell(n, l_r, s_tot):
                continue
            for j_tot in np.arange(abs(l_r - s_tot), l_r + s_tot + 1):
                for f_tot in np.arange(abs(j_tot - i_c), j_tot + i_c + 1):
                    for m in self._get_m_range(m_range, f_tot):
                        angular_ket = AngularKetLS(
                            l_r=l_r, s_tot=s_tot, j_tot=j_tot, f_tot=f_tot, m=m, species=self.species
                        )
                        state = RydbergStateSQDT(self.species, n=n, angular_ket=angular_ket)
                        self.states.append(state)  # type: ignore [arg-type]

    def _add_states_jj(self, n: int, l_r: int, m_range: tuple[float, float] | None | NotSet = NotSet) -> None:
        i_c = self.element_properties.i_c
        s_r = 0.5
        s_c = j_c = self.element_properties.s_c
        s_tot_list = np.arange(s_r - s_c, s_r + s_c + 1)

        allowed = [self.sqdt.is_allowed_shell(n, l_r, s) for s in s_tot_list]
        if not all(allowed):
            if any(allowed):
                logger.warning(
                    "For l=%d, n=%d one of the singlet/triplet states is not allowed. "
                    "In JJ coupling the state does not exist, thus skipping this shell",
                    *(l_r, n),
                )
            return

        for j_r in np.arange(abs(l_r - s_r), l_r + s_r + 1):
            for j_tot in range(int(abs(j_r - j_c)), int(j_r + j_c + 1)):
                for f_tot in np.arange(abs(j_tot - i_c), j_tot + i_c + 1):
                    for m in self._get_m_range(m_range, f_tot):
                        angular_ket = AngularKetJJ(
                            l_r=l_r, j_r=j_r, j_tot=j_tot, f_tot=f_tot, m=m, species=self.species
                        )
                        state = RydbergStateSQDT(self.species, n=n, angular_ket=angular_ket)
                        self.states.append(state)  # type: ignore [arg-type]

    def _add_states_fj(self, n: int, l_r: int, m_range: tuple[float, float] | None | NotSet = NotSet) -> None:
        i_c = self.element_properties.i_c
        s_r = 0.5
        s_c = j_c = self.element_properties.s_c
        s_tot_list = np.arange(s_r - s_c, s_r + s_c + 1)

        allowed = [self.sqdt.is_allowed_shell(n, l_r, s) for s in s_tot_list]
        if not all(allowed):
            if any(allowed):
                logger.warning(
                    "For l=%d, n=%d one of the singlet/triplet states is not allowed. "
                    "In FJ coupling the state does not exist, thus skipping this shell",
                    *(l_r, n),
                )
            return

        for j_r in np.arange(abs(l_r - s_r), l_r + s_r + 1):
            for f_c in np.arange(abs(j_c - i_c), j_c + i_c + 1):
                for f_tot in np.arange(abs(f_c - j_r), f_c + j_r + 1):
                    for m in self._get_m_range(m_range, f_tot):
                        angular_ket = AngularKetFJ(l_r=l_r, j_r=j_r, f_c=f_c, f_tot=f_tot, m=m, species=self.species)
                        state = RydbergStateSQDT(self.species, n=n, angular_ket=angular_ket)
                        self.states.append(state)  # type: ignore [arg-type]

    def _get_m_range(self, m_range: tuple[float, float] | None | NotSet, f_tot: float) -> list[NotSet | float]:
        if is_not_set(m_range):
            return [NotSet]
        if m_range is None:
            m_range = (-f_tot, f_tot)

        m_min = max(-f_tot, m_range[0])
        m_max = min(f_tot, m_range[1])
        return [float(_m) for _m in np.arange(m_min, m_max + 1)]
