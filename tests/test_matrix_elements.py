import numpy as np
import pytest
from rydstate import RydbergStateSQDT
from rydstate.units import BaseUnits, ureg


@pytest.mark.parametrize("l_r", [0, 1, 20])
def test_magnetic(l_r: int) -> None:
    """Test magnetic units."""
    g_s = 2.002319304363
    g_l = 1

    state = RydbergStateSQDT("Rb", n=max(l_r + 1, 10), l_r=l_r, j_r=l_r + 0.5, m=l_r + 0.5)

    # Check that for m = j_tot = l + s_tot the magnetic matrix element is - mu_B * (g_l * l_tot + g_s * s_tot)
    mu = state.calc_matrix_element(state, "magnetic_dipole", q=0)
    mu = mu.to("bohr_magneton")
    assert np.isclose(mu.magnitude, -(g_l * l_r + g_s * 0.5)), f"{mu.magnitude} != {-(g_l * l_r + g_s * 0.5)}"

    # Check dimensionality
    magnetic_field = ureg.Quantity(1, "T")
    zeeman_energy = -mu * magnetic_field
    assert zeeman_energy.dimensionality == BaseUnits["energy"].dimensionality, (
        f"{zeeman_energy.dimensionality} != {BaseUnits['energy'].dimensionality}"
    )
