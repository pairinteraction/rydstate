import numpy as np
import pytest
from rydstate import RydbergStateSQDT, RydbergStateSQDTAlkali
from rydstate.angular import AngularKetLS
from rydstate.species import ElementProperties
from rydstate.species.utils import get_all_subclasses, get_subclass

ALL_AVAILABLE_SPECIES = [cls.species for cls in get_all_subclasses(ElementProperties)]


# Reference values from NIST Atomic Spectra Database (ASD), Einstein A coefficients:
#   H 2p -> 1s: A = 6.2648e8 s^-1
#   H 3p -> all lower: A_tot = 1.8971e8 s^-1  (A(3p->1s)=1.6725e8, A(3p->2s)=2.245e7)
#   H 3d -> 2p: A = 6.4651e7 s^-1
# Source: https://physics.nist.gov/PhysRefData/ASD/lines_form.html
@pytest.mark.parametrize(
    ("n", "l", "j", "expected_gamma"),
    [
        (2, 1, 1.5, 6.2648e8),
        (3, 1, 1.5, 1.8971e8),
        (3, 2, 2.5, 6.4651e7),
    ],
)
def test_hydrogen_textbook_lifetimes(n: int, l: int, j: float, expected_gamma: float) -> None:
    """Test that calculated H lifetimes match NIST textbook values within 5%."""
    state = RydbergStateSQDTAlkali("H_textbook", n=n, l=l, j=j, m=0.5)
    tau = state.get_lifetime(unit="s")
    np.testing.assert_allclose(tau, 1 / expected_gamma, rtol=0.05)


def test_bbr_shortens_lifetime() -> None:
    """Test that black body radiation at 300 K shortens the lifetime relative to T=0."""
    state = RydbergStateSQDTAlkali("Rb", n=30, l=0, j=0.5, m=0.5)
    tau_0 = state.get_lifetime(unit="mus")
    tau_300 = state.get_lifetime(300, temperature_unit="K", unit="mus")
    assert tau_300 < tau_0


@pytest.mark.parametrize("species", ALL_AVAILABLE_SPECIES)
def test_lifetime_n_scaling(species: str) -> None:
    """Test that Rydberg state lifetimes scale as nu^3 (effective quantum number)."""
    if species != "Na":
        pytest.skip("Skip this test for most species for now, since this test is rather slow.")

    if species in ["Sr87", "Yb171", "Yb173"]:
        pytest.skip("No quantum defect data available")
    if species in ["Yb174", "Yb174_sqdt"]:
        pytest.skip("Quantum defects not correct for low n states")

    n1, n2 = 30, 60
    element_properties = get_subclass(ElementProperties, species)()
    s_tot = (element_properties.number_valence_electrons / 2) % 1
    f = abs(s_tot - element_properties.i_c)
    angular = AngularKetLS(l_r=0, s_tot=s_tot, f_tot=f, m=f, species=species)

    state1 = RydbergStateSQDT.from_angular_ket(species, angular, n=n1)
    state2 = RydbergStateSQDT.from_angular_ket(species, angular, n=n2)
    tau1 = state1.get_lifetime(unit="mus")
    tau2 = state2.get_lifetime(unit="mus")
    nu1 = state1.nu
    nu2 = state2.nu
    expected_ratio = (nu2 / nu1) ** 3
    np.testing.assert_allclose(tau2 / tau1, expected_ratio, rtol=0.05)
