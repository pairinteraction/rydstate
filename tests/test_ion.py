import pytest
from rydstate.species import get_element_properties, get_potential_class, get_sqdt
from rydstate.species.utils import calc_energy_from_nu, calc_nu_from_energy


def test_ion_element_properties() -> None:
    element_properties = get_element_properties("Yb174_ion")
    assert element_properties.net_charge == 2  # electron orbits the doubly-charged Yb2+ core
    assert element_properties.number_valence_electrons == 1  # alkali-like above the closed-shell core
    assert element_properties.s_c == 0.0


def test_energy_nu_helpers_scale_with_charge_squared() -> None:
    # E = -1/2 Z^2 mu / nu^2, and nu is the inverse relation
    assert calc_energy_from_nu(1.0, 50, charge=2) == pytest.approx(-0.5 * 4 / 2500)
    assert calc_nu_from_energy(1.0, calc_energy_from_nu(1.0, 37.0, charge=2), charge=2) == pytest.approx(37.0)
    # charge=1 (default) reproduces the neutral-atom relation
    assert calc_energy_from_nu(1.0, 50) == pytest.approx(-0.5 / 2500)


def test_ion_coulomb_potential_uses_net_charge() -> None:
    potential = get_potential_class("Yb174_ion")(0)
    assert potential.calc_potential_coulomb(10.0) == pytest.approx(-2 / 10)


def test_ion_nist_data_is_loaded() -> None:
    sqdt = get_sqdt("Yb174_ion")
    levels = sqdt._nist_energy_levels  # noqa: SLF001
    # the parser must pick up the single-electron Yb+ levels over the 4f14 core
    assert levels[(6, 0, 0.5, 0.5)] == 0.0  # 4f14.6s 2S1/2 ground state
    assert (6, 1, 0.5, 0.5) in levels  # 4f14.6p 2P1/2
    assert (5, 2, 1.5, 0.5) in levels  # 4f14.5d 2D3/2
    # only doublets (s_tot = 1/2) since Yb+ is alkali-like (closed-shell 4f14 core)
    assert all(s_tot == 0.5 for (_n, _l, _j, s_tot) in levels)
