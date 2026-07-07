from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, overload

from rydstate.radial import RadialDummy
from rydstate.units import MatrixElementOperatorRanks, ureg

if TYPE_CHECKING:
    from rydstate.angular.angular_ket import AngularKetBase
    from rydstate.radial.radial_base import RadialBase
    from rydstate.units import MatrixElementOperator, PintFloat


logger = logging.getLogger(__name__)


ELECTRIC_MULTIPOLE_PREFACTORS: dict[int, float] = {
    k_angular: math.sqrt(4 * math.pi / (2 * k_angular + 1)) for _, k_angular in MatrixElementOperatorRanks.values()
}


class RydbergKet:
    """Create a Rydberg ket, i.e. a tensor product of a radial ket and an angular ket."""

    def __init__(
        self,
        species: str,
        angular: AngularKetBase[Any],
        radial: RadialBase,
    ) -> None:
        r"""Initialize the Rydberg state."""
        self.species = species
        self.angular = angular
        self.radial = radial

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.species}, {self.radial!r}, {self.angular!r})"

    def __str__(self) -> str:
        return f"({self.species}, {self.radial}, {self.angular})"

    def calc_reduced_overlap(self, other: RydbergKet) -> float:
        """Calculate the reduced overlap <self|other> (ignoring the magnetic quantum number m)."""
        if isinstance(self.radial, RadialDummy) and isinstance(other.radial, RadialDummy):
            return 1 if abs(self.radial.nu - other.radial.nu) < 1e-10 else 0
        if isinstance(self.radial, RadialDummy) or isinstance(other.radial, RadialDummy):
            return 0

        angular_overlap = self.angular.calc_reduced_overlap(other.angular)
        if angular_overlap == 0:
            return 0
        radial_overlap = self.radial.calc_overlap(other.radial)
        return radial_overlap * angular_overlap

    @overload
    def calc_reduced_matrix_element(
        self, other: RydbergKet, operator: MatrixElementOperator, unit: None = None
    ) -> PintFloat: ...

    @overload
    def calc_reduced_matrix_element(self, other: RydbergKet, operator: MatrixElementOperator, unit: str) -> float: ...

    def calc_reduced_matrix_element(
        self, other: RydbergKet, operator: MatrixElementOperator, unit: str | None = None
    ) -> PintFloat | float:
        r"""Calculate the reduced matrix element.

        Calculate the reduced matrix element between self and other (ignoring m quantum numbers)

        .. math::
            \left\langle self || r^k_radial \hat{O}^{(k_{angular})} || other \right\rangle

        where \hat{O}^{(k_{angular})} is the operator of rank k_angular for which to calculate the matrix element.
        k_radial and k_angular are determined from the operator automatically.

        Args:
            other: The other Rydberg state for which to calculate the matrix element.
            operator: The operator for which to calculate the matrix element.
            unit: The unit to which to convert the radial matrix element.
                Can be "a.u." for atomic units (so no conversion is done), or a specific unit.
                Default None will return a pint quantity.

        Returns:
            The reduced matrix element for the given operator.

        """
        try:
            k_radial, k_angular = MatrixElementOperatorRanks[operator]
        except KeyError as err:
            raise ValueError(
                f"Operator {operator} not supported, must be one of {list(MatrixElementOperatorRanks.keys())}."
            ) from err

        if self.radial._is_dummy or other.radial._is_dummy:  # noqa: SLF001
            # no matrix element can be calculated for dummy radial wavefunctions, return 0
            return 0

        if operator == "magnetic_dipole":
            # Magnetic dipole operator: mu = - mu_B (g_l <l_tot> + g_s <s_tot>)
            g_s = 2.0023192
            value_s_tot = self.angular.calc_reduced_matrix_element(other.angular, "s_tot", k_angular)
            g_l = 1
            value_l_tot = self.angular.calc_reduced_matrix_element(other.angular, "l_tot", k_angular)
            angular_matrix_element = g_s * value_s_tot + g_l * value_l_tot
            prefactor = -0.5
            # Note: we use the convention, that the magnetic dipole moments are given
            # as the same dimensionality as the Bohr magneton (mu = - mu_B (g_l l + g_s s_tot))
            # such that - mu * B (where the magnetic field B is given in dimension Tesla) is an energy

        elif operator in ["electric_dipole", "electric_quadrupole", "electric_octupole", "electric_quadrupole_zero"]:
            # Electric multipole operator: p_{k,q} = e r^k_radial * sqrt(4pi / (2k+1)) * Y_{k_angular,q}(\theta, phi)
            angular_matrix_element = self.angular.calc_reduced_matrix_element(other.angular, "spherical", k_angular)
            # Prefactor sqrt(4 pi / (2 k_angular + 1)) for the electric multipole operators, precomputed for performance
            prefactor = ELECTRIC_MULTIPOLE_PREFACTORS[k_angular]

        else:
            raise NotImplementedError(f"Operator {operator} not implemented.")

        if angular_matrix_element == 0:
            matrix_element = 0.0
        else:
            radial_matrix_element = self.radial.calc_matrix_element(other.radial, k_radial, unit="a.u.")
            matrix_element = prefactor * radial_matrix_element * angular_matrix_element

        if unit == "a.u.":
            return matrix_element

        radial_unit: PintFloat = ureg.Quantity(1, "bohr_radius") ** k_radial
        matrix_element_unit: PintFloat
        if operator == "magnetic_dipole":
            matrix_element_unit = radial_unit * ureg.Quantity(2, "bohr_magneton")
        elif operator in ["electric_dipole", "electric_quadrupole", "electric_octupole", "electric_quadrupole_zero"]:
            matrix_element_unit = radial_unit * ureg.Quantity(1, "e")
        else:
            raise NotImplementedError(f"Operator {operator} not implemented.")

        if unit is None:
            return matrix_element * matrix_element_unit.to_base_units()  # type: ignore [no-any-return]
        return matrix_element * matrix_element_unit.to(unit).magnitude

    @overload
    def calc_matrix_element(
        self, other: RydbergKet, operator: MatrixElementOperator, q: int, unit: None = None
    ) -> PintFloat: ...

    @overload
    def calc_matrix_element(self, other: RydbergKet, operator: MatrixElementOperator, q: int, unit: str) -> float: ...

    def calc_matrix_element(
        self, other: RydbergKet, operator: MatrixElementOperator, q: int, unit: str | None = None
    ) -> PintFloat | float:
        r"""Calculate the matrix element.

        Calculate the full matrix element between self and other,
        also considering the magnetic quantum numbers m of self and other.

        .. math::
            \left\langle self | r^k_radial \hat{O}^{(k_{angular})}_q | other \right\rangle

        where \hat{O}^{(k_{angular})}_q is the operator of rank k_angular for which to calculate the matrix element.
        k_radial and k_angular are determined from the operator automatically.

        Args:
            other: The other Rydberg state for which to calculate the matrix element.
            operator: The operator for which to calculate the matrix element.
            q: The component of the operator.
            unit: The unit to which to convert the radial matrix element.
                Can be "a.u." for atomic units (so no conversion is done), or a specific unit.
                Default None will return a pint quantity.

        Returns:
            The matrix element for the given operator.

        """
        _k_radial, k_angular = MatrixElementOperatorRanks[operator]
        prefactor = self.angular._calc_wigner_eckart_prefactor(other.angular, k_angular, q)  # noqa: SLF001
        reduced_matrix_element = self.calc_reduced_matrix_element(other, operator, unit)
        return prefactor * reduced_matrix_element
