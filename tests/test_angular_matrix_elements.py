from __future__ import annotations

from typing import TYPE_CHECKING, get_args

import numpy as np
import pytest
from rydstate.angular import AngularKetFJ, AngularKetJJ, AngularKetLS
from rydstate.angular.utils import AngularMomentumQuantumNumbers, Unknown

if TYPE_CHECKING:
    from rydstate.angular.angular_ket import AngularKetBase
    from rydstate.angular.utils import AllKnown, AngularOperatorType, CouplingScheme

TEST_KET_PAIRS = [
    (
        AngularKetLS(s_tot=1, l_r=1, j_tot=1, f_tot=1.5, species="Yb173"),
        AngularKetLS(s_tot=1, l_r=1, j_tot=1, f_tot=1.5, species="Yb173"),
    ),
    (
        AngularKetLS(s_tot=1, l_r=1, j_tot=1, f_tot=1.5, species="Yb173"),
        AngularKetFJ(f_c=2, l_r=1, j_r=1.5, f_tot=2.5, species="Yb173"),
    ),
    (
        AngularKetFJ(f_c=2, l_r=1, j_r=1.5, f_tot=2.5, species="Yb173"),
        AngularKetFJ(f_c=2, l_r=1, j_r=1.5, f_tot=2.5, species="Yb173"),
    ),
    (
        AngularKetFJ(i_c=2.5, s_c=0.5, l_c=0, s_r=0.5, l_r=1, j_c=0.5, f_c=2.0, j_r=1.5, f_tot=2.5),
        AngularKetFJ(i_c=2.5, s_c=0.5, l_c=0, s_r=0.5, l_r=2, j_c=0.5, f_c=2.0, j_r=1.5, f_tot=2.5),
    ),
]

TEST_KETS = [
    AngularKetLS(s_tot=1, l_r=1, j_tot=1, f_tot=1.5, species="Yb173"),
    AngularKetLS(s_tot=1, l_r=1, j_tot=1, f_tot=1.5, species="Yb173"),
    AngularKetJJ(l_r=1, j_r=1.5, j_tot=2, f_tot=2.5, species="Yb173"),
    AngularKetFJ(f_c=2, l_r=1, j_r=1.5, f_tot=2.5, species="Yb173"),
    AngularKetFJ(f_c=2, l_r=1, j_r=1.5, f_tot=2.5, species="Yb173"),
]


@pytest.mark.parametrize("ket", TEST_KETS)
def test_exp_q_different_coupling_schemes(ket: AngularKetBase[AllKnown], coupling_scheme: CouplingScheme) -> None:
    """Expectation values and standard deviations of every quantum number are scheme-independent."""
    all_qs: tuple[AngularMomentumQuantumNumbers, ...] = get_args(AngularMomentumQuantumNumbers)
    for q in all_qs:
        # the LS scheme serves as the fixed reference the other schemes are compared against
        assert np.isclose(ket.to_state("LS").calc_exp_qn(q), ket.to_state(coupling_scheme).calc_exp_qn(q))
        assert np.isclose(ket.to_state("LS").calc_std_qn(q), ket.to_state(coupling_scheme).calc_std_qn(q))


@pytest.mark.parametrize(("ket1", "ket2"), TEST_KET_PAIRS)
def test_overlap_different_coupling_schemes(
    ket1: AngularKetBase[AllKnown], ket2: AngularKetBase[AllKnown], coupling_scheme: CouplingScheme
) -> None:
    ov = ket1.calc_reduced_overlap(ket2)

    assert np.isclose(ov, ket1.to_state().calc_reduced_overlap(ket2.to_state(coupling_scheme)))
    assert np.isclose(ov, ket1.to_state(coupling_scheme).calc_reduced_overlap(ket2))
    assert np.isclose(1, ket1.to_state(coupling_scheme).calc_reduced_overlap(ket1))
    assert np.isclose(1, ket2.to_state(coupling_scheme).calc_reduced_overlap(ket2))


def test_identical_unknown_kets_have_unit_overlap() -> None:
    ket1 = AngularKetLS(
        i_c=0,
        s_c=0,
        l_c=0,
        s_r=0.5,
        l_r=Unknown,
        s_tot=0.5,
        l_tot=Unknown,
        j_tot=Unknown,
        f_tot=0.5,
        parity=1,
        allow_unknown=True,
    )
    ket2 = AngularKetLS(
        i_c=0,
        s_c=0,
        l_c=0,
        s_r=0.5,
        l_r=Unknown,
        s_tot=0.5,
        l_tot=Unknown,
        j_tot=Unknown,
        f_tot=0.5,
        parity=1,
        allow_unknown=True,
    )
    assert ket1.calc_reduced_overlap(ket2) == 1.0
    assert ket2.calc_reduced_overlap(ket1) == 1.0


@pytest.mark.parametrize("ket", TEST_KETS)
def test_reduced_identity(ket: AngularKetBase[AllKnown], coupling_scheme: CouplingScheme) -> None:
    reduced_identity = np.sqrt(2 * ket.f_tot + 1)

    op: AngularMomentumQuantumNumbers
    state = ket.to_state(coupling_scheme)
    for op in state.kets[0].quantum_number_names:
        assert np.isclose(reduced_identity, state.calc_reduced_matrix_element(state, "identity_" + op, kappa=0))  # type: ignore [arg-type]


@pytest.mark.parametrize("ket", TEST_KETS)
def test_scalar_matrix_element_independent_of_m(ket: AngularKetBase[AllKnown]) -> None:
    """For a scalar operator (kappa=q=0) the matrix element is independent of m.

    This also checks that calc_matrix_element works when m is NotSet, in which case
    _calc_wigner_eckart_prefactor may pick an arbitrary m (see angular_ket.py).
    """
    operators: list[AngularOperatorType] = [
        "identity_" + op  # type: ignore [misc]
        for op in ket.quantum_number_names
    ]
    m_values = np.arange(-ket.f_tot, ket.f_tot + 1)

    for operator in operators:
        val_not_set = ket.calc_matrix_element(ket, operator, kappa=0, q=0)
        assert np.isclose(val_not_set, 1.0), f"{operator=}, {val_not_set=}"
        for m in m_values:
            ket_m = ket.replace_m(m)
            val_m = ket_m.calc_matrix_element(ket_m, operator, kappa=0, q=0)
            assert np.isclose(val_m, val_not_set), f"{operator=}, {m=}, {val_m=}, {val_not_set=}"


@pytest.mark.parametrize("ket", TEST_KETS)
def test_reduced_raw_value(ket: AngularKetBase[AllKnown]) -> None:
    # In its native coupling scheme every quantum number is definite, so raw_value_x = x^exponent * identity and its
    # reduced matrix element is x^exponent * sqrt(2 * f_tot + 1). This pins down the Wigner-Eckart normalization: the
    # full matrix element <ket| raw_value_x |ket> then evaluates to the raw value of x (and raw_value_x_2 to x^2).
    reduced_identity = np.sqrt(2 * ket.f_tot + 1)

    op: AngularMomentumQuantumNumbers
    for op in ket.quantum_number_names:
        raw = ket.get_qn(op)
        assert np.isclose(raw * reduced_identity, ket.calc_reduced_matrix_element(ket, "raw_value_" + op, kappa=0))  # type: ignore [arg-type]
        assert np.isclose(
            raw**2 * reduced_identity,
            ket.calc_reduced_matrix_element(ket, "raw_value_" + op + "_2", kappa=0),  # type: ignore [arg-type]
        )
        for m in np.arange(-ket.f_tot, ket.f_tot + 1):
            ket_m = ket.replace_m(m)
            assert np.isclose(raw, ket_m.calc_matrix_element(ket_m, "raw_value_" + op, kappa=0, q=0))  # type: ignore [arg-type]
            assert np.isclose(raw**2, ket_m.calc_matrix_element(ket_m, "raw_value_" + op + "_2", kappa=0, q=0))  # type: ignore [arg-type]


@pytest.mark.parametrize("ket", TEST_KETS)
def test_reduced_spin_squared(ket: AngularKetBase[AllKnown]) -> None:
    op: AngularMomentumQuantumNumbers
    coupling_schemes: list[CouplingScheme] = ["LS", "JJ", "FJ"]
    for scheme in coupling_schemes:
        state = ket.to_state(scheme)
        for op in state.kets[0].quantum_number_names:
            # the squared spin operator is diagonal in the scheme's own basis with eigenvalue qn * (qn + 1),
            # so for a superposition the reduced matrix element is the weighted average over the components
            exp_squared = sum(coeff**2 * s_ket.get_qn(op) * (s_ket.get_qn(op) + 1) for coeff, s_ket in state)
            reduced_squared = exp_squared * np.sqrt(2 * ket.f_tot + 1)
            assert np.isclose(reduced_squared, state.calc_reduced_matrix_element(state, "squared_" + op, kappa=0))  # type: ignore [arg-type]


def test_spin_squared_expectation_value() -> None:
    """The expectation value of s^2 must be s(s+1) for good quantum numbers and match the sum rule otherwise."""
    ket = AngularKetFJ(l_r=0, j_r=0.5, f_c=1, m=0.5, f_tot=0.5, species="Yb171")

    # s_r and s_c are good quantum numbers in every ket, so <s^2> = s(s+1) = 3/4
    assert np.isclose(ket.calc_matrix_element(ket, "squared_s_r", 0, q=0), 0.75)
    assert np.isclose(ket.calc_matrix_element(ket, "squared_s_c", 0, q=0), 0.75)

    # s_tot is not a good quantum number of an FJ ket, so <s_tot^2> is a weighted average
    # over the LS decomposition: sum_i |c_i|^2 * s_tot_i * (s_tot_i + 1)
    expected = sum(coeff**2 * ls_ket.s_tot * (ls_ket.s_tot + 1) for coeff, ls_ket in ket.to_state("LS"))
    assert np.isclose(ket.calc_matrix_element(ket, "squared_s_tot", 0, q=0), expected)

    # <s^2> must also equal the sum over all final states and components q of the squared
    # matrix elements of the (rank-1) spin operator: <s^2> = sum_{f,q} |<f|s_q|ket>|^2
    finals = [
        AngularKetFJ(l_r=0, j_r=0.5, f_c=f_c, m=m / 2, f_tot=f_tot, species="Yb171")
        for f_c in (0, 1)
        for f_tot in {abs(f_c - 0.5), f_c + 0.5}
        for m in range(-int(2 * f_tot), int(2 * f_tot) + 1, 2)
    ]
    for op in ("s_r", "s_c"):
        sum_rule = sum(f.calc_matrix_element(ket, op, 1, q=q) ** 2 for f in finals for q in (-1, 0, 1))
        assert np.isclose(sum_rule, ket.calc_matrix_element(ket, "squared_" + op, 0, q=0))  # type: ignore [arg-type]


@pytest.mark.parametrize(("ket1", "ket2"), TEST_KET_PAIRS)
def test_matrix_elements_in_different_coupling_schemes(
    ket1: AngularKetBase[AllKnown], ket2: AngularKetBase[AllKnown], coupling_scheme: CouplingScheme
) -> None:
    example_list: list[tuple[AngularOperatorType, int]] = [
        ("spherical", 0),
        ("spherical", 1),
        ("spherical", 2),
        ("spherical", 3),
        ("s_tot", 1),
        ("l_r", 1),
        ("i_c", 1),
        ("f_tot", 1),
        ("j_tot", 1),
    ]

    for operator, kappa in example_list:
        msg = f"{operator=}, {kappa=}, {ket1=}, {ket2=}, {coupling_scheme=}"
        val = ket1.calc_reduced_matrix_element(ket2, operator, kappa)

        assert np.isclose(
            val, ket1.to_state().calc_reduced_matrix_element(ket2.to_state(coupling_scheme), operator, kappa)
        ), msg
        assert np.isclose(val, ket1.to_state(coupling_scheme).calc_reduced_matrix_element(ket2, operator, kappa)), msg
