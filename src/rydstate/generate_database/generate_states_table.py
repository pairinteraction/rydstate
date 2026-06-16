from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from rydstate.rydberg_state.rydberg_sqdt import RydbergStateSQDT

if TYPE_CHECKING:
    import sqlite3

    from rydstate.basis import BasisMQDT, BasisSQDT
    from rydstate.rydberg_state.rydberg_base import RydbergStateBase


logger = logging.getLogger(__name__)

COLUMNS = [
    "id",
    "energy",
    "parity",
    "n",
    "nu",
    "f",
    "exp_nui",
    "exp_l",
    "exp_j",
    "exp_s",
    "exp_l_ryd",
    "exp_j_ryd",
    "std_nui",
    "std_l",
    "std_j",
    "std_s",
    "std_l_ryd",
    "std_j_ryd",
    "is_j_total_momentum",
    "is_calculated_with_mqdt",
    "underspecified_channel_contribution",
]


def generate_states_table(
    basis: BasisMQDT | BasisSQDT[Any],
    conn: sqlite3.Connection | None = None,
) -> list[tuple[float | int | str | bool, ...]]:
    """Populate the states table for a given species and n-range using BasisSQDT."""
    basis.sort_states("nu")  # sort by nu == sort by energy

    states_data: list[tuple[float | int | str | bool, ...]] = []
    for ids, state in enumerate(basis.states):
        states_data.append(get_state_data(ids, state))

    assert len(states_data) == 0 or len(COLUMNS) == len(states_data[0])

    if conn is not None:
        stmt = f"INSERT INTO states ({', '.join(COLUMNS)}) VALUES ({', '.join(['?'] * len(COLUMNS))})"  # noqa: S608
        conn.executemany(stmt, states_data)
        num_rows = conn.execute("SELECT COUNT(*) FROM states").fetchone()[0]
        logger.info("Created the 'states' table (%s rows)", num_rows)

    return states_data


def get_state_data(ids: int, state: RydbergStateBase) -> tuple[float | int | str | bool, ...]:
    """Get the data for a given state as a tuple."""
    angular = state.angular
    underspecified_channel_contribution = sum(abs(coeff) ** 2 for coeff, ket in state if ket.angular.contains_unknown)

    n = state.n if isinstance(state, RydbergStateSQDT) else 0

    data = (
        ids,  # id
        state.get_energy("a.u."),  # energy
        angular.parity,  # parity = (-1)^l_tot
        n,  # n: quantum number
        state.nu,  # nu
        angular.f_tot,  # f_tot
        state.calc_exp_qn("nui"),  # exp_nui
        angular.calc_exp_qn("l_tot"),  # exp_l
        angular.calc_exp_qn("j_tot"),  # exp_j
        angular.calc_exp_qn("s_tot"),  # exp_s
        angular.calc_exp_qn("l_r"),  # exp_l_ryd
        angular.calc_exp_qn("j_r"),  # exp_j_ryd = j for sqdt only one valence electron
        state.calc_std_qn("nui"),  # std_nui = 0
        angular.calc_std_qn("l_tot"),  # std_l
        angular.calc_std_qn("j_tot"),  # std_j
        angular.calc_std_qn("s_tot"),  # std_s
        angular.calc_std_qn("l_r"),  # std_l_ryd
        angular.calc_std_qn("j_r"),  # std_j_ryd
        bool(angular.i_c == 0),  # is_j_total_momentum
        bool(len(state.rydberg_kets) > 1),  # is_calculated_with_mqdt
        underspecified_channel_contribution,  # underspecified_channel_contribution = 0 for sqdt
    )
    return tuple(x.item() if isinstance(x, np.generic) else x for x in data)
