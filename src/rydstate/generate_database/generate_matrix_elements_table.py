from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from rydstate.angular.angular_ket import AngularKetBase
from rydstate.angular.utils import is_unknown
from rydstate.units import MatrixElementOperatorRanks

if TYPE_CHECKING:
    import sqlite3

    from rydstate.basis import BasisMQDT, BasisSQDT
    from rydstate.rydberg_state.rydberg_base import RydbergStateBase
    from rydstate.units import MatrixElementOperator

logger = logging.getLogger(__name__)

COLUMNS = [
    "id_initial",
    "id_final",
    "val",
]

MATRIX_ELEMENTS_OF_INTEREST: dict[str, MatrixElementOperator] = {
    "matrix_elements_d": "electric_dipole",
    "matrix_elements_q": "electric_quadrupole",
    "matrix_elements_o": "electric_octupole",
    "matrix_elements_q0": "electric_quadrupole_zero",
    "matrix_elements_mu": "magnetic_dipole",
}


def get_l_r_difference(state1: RydbergStateBase, state2: RydbergStateBase) -> int:
    """Calculate the minimal difference in l_r between two states."""
    angular1, angular2 = state1.angular, state2.angular
    if isinstance(angular1, AngularKetBase) and isinstance(angular2, AngularKetBase):
        return abs(angular1.l_r - angular2.l_r)  # type: ignore [no-any-return]

    if isinstance(angular1, AngularKetBase):
        angular1 = angular1.to_state()
    if isinstance(angular2, AngularKetBase):
        angular2 = angular2.to_state()

    if any(ket1 == ket2 for ket1 in angular1.kets for ket2 in angular2.kets):
        # this also checks states with l_r = unknown
        return 0

    all_diffs = [
        ket1.l_r - ket2.l_r
        for ket1 in angular1.kets
        for ket2 in angular2.kets
        if not is_unknown(ket1.l_r) and not is_unknown(ket2.l_r)
    ]
    if len(all_diffs) == 0:
        raise RuntimeError("Could not calculate l_r difference, this should not happen.")
    return int(np.min(np.abs(all_diffs)))


def generate_matrix_elements_tables(  # noqa: C901
    basis: BasisMQDT | BasisSQDT[Any],
    conn: sqlite3.Connection | None = None,
    max_delta_nu: float = float("inf"),
    all_nu_up_to: float = float("inf"),
    *,
    free_memory: bool = False,
) -> dict[str, list[tuple[int, int, float]]]:
    """Populate matrix element tables for all relevant pairs of states."""
    k_angular_max = max(MatrixElementOperatorRanks[op][1] for op in MATRIX_ELEMENTS_OF_INTEREST.values())

    basis.sort_states("nu")  # sort by nu == sort by energy
    list_of_id_state = list(enumerate(basis.states))
    list_of_id_state = sorted(list_of_id_state, key=lambda x: (x[1].angular.calc_exp_qn("l_r"), x[1].nu, x[0]))

    matrix_elements: dict[str, list[tuple[int, int, float]]] = {tkey: [] for tkey in MATRIX_ELEMENTS_OF_INTEREST}
    for i, (id1, state1) in enumerate(list_of_id_state):
        for id2, state2 in list_of_id_state[i:]:
            if get_l_r_difference(state1, state2) > k_angular_max:
                # If the difference in l is larger than k_angular_max, no matrix elements have to be calculated
                continue
            if (
                all(nu > all_nu_up_to for nu in [state1.nu, state2.nu])
                and abs(state1.nu - state2.nu) > max_delta_nu + 0.5
            ):
                # If delta_nu is larger than max_delta_nu (+0.5 to not lose states compared to previous max_delta_n)
                # we dont calculate the matrix elements anymore,
                # since these are so small, that they are usually not relevant for further calculations
                # However, we keep all dipole interactions with small n (we choose all_nu_up_to as a cutoff)
                # since these are relevant for the spontaneous decay rates
                continue

            id_tuple = (id1, id2) if id1 <= id2 else (id2, id1)
            states = (state1, state2) if id1 <= id2 else (state2, state1)

            me_one_pair = calc_matrix_elements_one_pair(states[0], states[1], MATRIX_ELEMENTS_OF_INTEREST)
            for tkey, me in me_one_pair.items():
                matrix_elements[tkey].append((id_tuple[0], id_tuple[1], me))

            if id1 != id2:
                me_one_pair = calc_matrix_elements_one_pair(states[1], states[0], MATRIX_ELEMENTS_OF_INTEREST)
                for tkey, me in me_one_pair.items():
                    matrix_elements[tkey].append((id_tuple[1], id_tuple[0], me))

        if free_memory:
            state1.free_memory()

    for key, mes in matrix_elements.items():
        matrix_elements[key] = sorted(mes)
        assert len(mes) == 0 or len(COLUMNS) == len(mes[0])

    if conn is not None:
        for tkey, mes in matrix_elements.items():
            stmt = f"INSERT INTO {tkey} ({', '.join(COLUMNS)}) VALUES ({', '.join(['?'] * len(COLUMNS))})"  # noqa: S608
            conn.executemany(stmt, mes)
            num_rows = conn.execute(f"SELECT COUNT(*) FROM {tkey}").fetchone()[0]  # noqa: S608
            logger.info("Created the '%s' table (%s rows)", tkey, num_rows)

    return matrix_elements


def calc_matrix_elements_one_pair(
    state1: RydbergStateBase, state2: RydbergStateBase, matrix_elements_of_interest: dict[str, MatrixElementOperator]
) -> dict[str, float]:
    matrix_elements: dict[str, float] = {}
    for tkey, operator in matrix_elements_of_interest.items():
        me = state2.calc_reduced_matrix_element(state1, operator, unit="a.u.")
        if me != 0:
            matrix_elements[tkey] = me
    return matrix_elements
