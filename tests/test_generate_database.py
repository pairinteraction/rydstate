from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from rydstate import RydbergStateSQDTAlkali
from rydstate.basis import BasisMQDT, BasisSQDT
from rydstate.generate_database.generate_matrix_elements_table import generate_matrix_elements_tables
from rydstate.generate_database.generate_misc_table import generate_wigner_table
from rydstate.generate_database.generate_states_table import generate_states_table, get_state_data

TEST_SPECIES_SPECIFIER = [
    *["H", "Li", "Na", "K", "Rb", "Cs"],
    *["Sr88_sqdt", "Yb174_sqdt"],
    *["Sr87_mqdt", "Sr88_mqdt", "Yb171_mqdt", "Yb173_mqdt", "Yb174_mqdt"],
]


def test_generate_wigner_table_returns_rows() -> None:
    table = generate_wigner_table(f_max=0, kappa_max=0)

    assert table == {
        "f_initial": [0.0],
        "f_final": [0.0],
        "m_initial": [0.0],
        "m_final": [0.0],
        "kappa": [0],
        "q": [0],
        "val": [1.0],
    }


def test_get_state_data_for_sqdt_alkali_state() -> None:
    state = RydbergStateSQDTAlkali("H", n=1, l=0, j=0.5)
    row = get_state_data(7, state)
    assert row[0] == 7
    assert row[2:6] == (1, 1, 1.0, 0.5)
    assert row[6:12] == (1.0, 0, 0.5, 0.5, 0, 0.5)
    assert row[12:] == (0, 0, 0, 0, 0, 0, True, False, 0)


@pytest.mark.parametrize("species_specifier", TEST_SPECIES_SPECIFIER)
def test_generate_states_table(species_specifier: str) -> None:
    species = species_specifier.removesuffix("_mqdt").removesuffix("_sqdt")
    basis: BasisMQDT | BasisSQDT[Any]
    if species_specifier.endswith("_mqdt"):
        basis = BasisMQDT(species, nu=(50, 52), l_r=(0, 2))
    else:
        basis = BasisSQDT(species, n=(50, 52), l_r=(0, 2), coupling_scheme="LS")
    basis.sort_states("nu")

    table = generate_states_table(basis)

    assert len(basis.states) > 2
    assert all(len(values) == len(basis.states) for values in table.values())

    assert np.allclose(table["nu"], basis.calc_exp_qn("nu"))
    assert np.allclose(table["exp_l_ryd"], basis.calc_exp_qn("l_r"))
    assert np.allclose(table["exp_s"], basis.calc_exp_qn("s_tot"))


@pytest.mark.parametrize("species_specifier", TEST_SPECIES_SPECIFIER)
def test_generate_matrix_elements_table(species_specifier: str) -> None:
    species = species_specifier.removesuffix("_mqdt").removesuffix("_sqdt")
    basis: BasisMQDT | BasisSQDT[Any]
    if species_specifier.endswith("_mqdt"):
        basis = BasisMQDT(species, nu=(50, 52), l_r=(0, 2))
    else:
        basis = BasisSQDT(species, n=(50, 52), l_r=(0, 2), coupling_scheme="LS")
    basis.sort_states("nu")

    tables = generate_matrix_elements_tables(basis, free_memory=False)

    for table in tables.values():
        assert all(len(values) > 2 for values in table.values())

    states = basis.states
    table = tables["matrix_elements_d"]
    for id_initial, id_final, val in zip(table["id_initial"], table["id_final"], table["val"], strict=True):
        reference = states[int(id_final)].calc_reduced_matrix_element(
            states[int(id_initial)], "electric_dipole", unit="a.u."
        )
        assert np.isclose(val, reference)
