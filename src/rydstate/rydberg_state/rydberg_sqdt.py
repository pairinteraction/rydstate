from __future__ import annotations

import logging
from functools import cached_property
from typing import TYPE_CHECKING, Generic, TypeVar, overload

import numpy as np
from scipy.special import exprel

from rydstate.angular import NotSet
from rydstate.angular.angular_ket import AngularKetBase, AngularKetLS
from rydstate.angular.utils import AllKnown, is_not_set, quantum_numbers_to_angular_ket
from rydstate.radial import RadialKet
from rydstate.rydberg_state.rydberg_base import RydbergStateBase
from rydstate.rydberg_state.rydberg_ket import RydbergKet
from rydstate.species import get_element_properties, get_sqdt
from rydstate.species.potential import Potential, get_potential_class
from rydstate.species.sqdt import SQDT
from rydstate.species.utils import calc_energy_from_nu
from rydstate.units import BaseQuantities, ureg

if TYPE_CHECKING:
    from typing_extensions import Self

    from rydstate.angular.angular_ket import AngularKetFJ, AngularKetJJ
    from rydstate.units import NDArray, PintArray, PintFloat

GenericT_AngularKet = TypeVar("GenericT_AngularKet", bound=AngularKetBase[AllKnown])
T_AngularKet = TypeVar("T_AngularKet", bound=AngularKetBase[AllKnown])

logger = logging.getLogger(__name__)


class RydbergStateSQDT(RydbergStateBase, Generic[GenericT_AngularKet]):
    """Create a Rydberg SQDT state, including the radial and angular states."""

    species: str
    """The atomic species of the Rydberg state."""

    angular: GenericT_AngularKet
    """The angular/spin part of the Rydberg electron."""

    @overload
    def __init__(
        self: RydbergStateSQDT[T_AngularKet],
        species: str,
        n: int,
        *,
        angular_ket: T_AngularKet,
        sqdt: str | SQDT | None = None,
        potential: str | Potential | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self: RydbergStateSQDT[AngularKetLS[AllKnown]],
        species: str,
        n: int,
        *,
        s_c: float | None = None,
        l_c: int | None = None,
        s_r: float | None = None,
        l_r: int,
        s_tot: float | None = None,
        l_tot: int | None = None,
        j_tot: float | None = None,
        f_tot: float | None = None,
        m: float | NotSet = NotSet,
        sqdt: str | SQDT | None = None,
        potential: str | Potential | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self: RydbergStateSQDT[AngularKetJJ[AllKnown]],
        species: str,
        n: int,
        *,
        s_c: float | None = None,
        l_c: int | None = None,
        j_c: float | None = None,
        s_r: float | None = None,
        l_r: int,
        j_r: float | None = None,
        j_tot: float,
        f_tot: float | None = None,
        m: float | NotSet = NotSet,
        sqdt: str | SQDT | None = None,
        potential: str | Potential | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self: RydbergStateSQDT[AngularKetFJ[AllKnown]],
        species: str,
        n: int,
        *,
        s_c: float | None = None,
        l_c: int | None = None,
        j_c: float | None = None,
        f_c: float,
        s_r: float | None = None,
        l_r: int,
        j_r: float | None = None,
        f_tot: float | None = None,
        m: float | NotSet = NotSet,
        sqdt: str | SQDT | None = None,
        potential: str | Potential | None = None,
    ) -> None: ...

    def __init__(
        self: RydbergStateSQDT[AngularKetBase[AllKnown]],
        species: str,
        n: int,
        *,
        s_c: float | None = None,
        l_c: int | None = None,
        j_c: float | None = None,
        f_c: float | None = None,
        s_r: float | None = None,
        l_r: int | None = None,
        j_r: float | None = None,
        s_tot: float | None = None,
        l_tot: int | None = None,
        j_tot: float | None = None,
        f_tot: float | None = None,
        m: float | NotSet = NotSet,
        angular_ket: AngularKetBase[AllKnown] | None = None,
        # potential and sqdt parameters
        sqdt: str | SQDT | None = None,
        potential: str | Potential | None = None,
    ) -> None:
        r"""Initialize the Rydberg state.

        Args:
            species: Atomic species.
            n: Principal quantum number of the rydberg electron.
            s_c: Spin quantum number of the core electron (0 for Alkali, 0.5 for divalent atoms).
            l_c: Orbital angular momentum quantum number of the core electron.
            j_c: Total angular momentum quantum number of the core electron.
            f_c: Total angular momentum quantum number of the core (core electron + nucleus).
            s_r: Spin quantum number of the rydberg electron always 0.5)
            l_r: Orbital angular momentum quantum number of the rydberg electron.
            j_r: Total angular momentum quantum number of the rydberg electron.
            s_tot: Total spin quantum number of all electrons.
            l_tot: Total orbital angular momentum quantum number of all electrons.
            j_tot: Total angular momentum quantum number of all electrons.
            f_tot: Total angular momentum quantum number of the atom (rydberg electron + core)
            m: Total magnetic quantum number.
                Optional, only needed for concrete angular matrix elements.
            angular_ket: The angular ket to use for the state.
                Either angular_ket or the quantum numbers for the angular ket must be given.
            sqdt: The SQDT to use for the state.
                Either a string representing the tag of the SQDT class to use,
                or an instance of an SQDT class.
            potential: The potential to use for the radial ket.
                Either a string representing the tag of the potential to use,
                or an instance of a potential class.

        """
        self.species = species
        self.element_properties = get_element_properties(species)

        if angular_ket is not None:
            if any(
                q is not None for q in [s_c, l_c, j_c, f_c, s_r, l_r, j_r, s_tot, l_tot, j_tot, f_tot]
            ) or not is_not_set(m):
                raise ValueError("Specify either angular_ket or the quantum numbers for the angular ket, not both.")
            self.angular = angular_ket
        else:
            l_c = 0 if l_c is None else l_c

            self.angular = quantum_numbers_to_angular_ket(
                species=self.species,
                s_c=s_c,
                l_c=l_c,
                j_c=j_c,
                f_c=f_c,
                s_r=s_r,
                l_r=l_r,
                j_r=j_r,
                s_tot=s_tot,
                l_tot=l_tot,
                j_tot=j_tot,
                f_tot=f_tot,
                m=m,
            )

        self.n = n
        self.sqdt = sqdt if isinstance(sqdt, SQDT) else get_sqdt(species, tag=sqdt)
        _s_tot = self.angular.get_qn("s_tot", allow_unknown=True)
        if not self.sqdt.is_allowed_shell(self.n, self.angular.l_r, _s_tot):
            raise ValueError(f"The Rydberg state {self} is not allowed due to forbidden shell configurations.")

        if isinstance(potential, Potential):
            if potential.l_r != self.angular.l_r:
                raise ValueError("The potential must have the same l_r as the angular ket.")
            self.potential = potential
        else:
            self.potential = get_potential_class(species, tag=potential)(self.angular.l_r)

        super().__init__()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.species}, n={self.n}, {self.angular!r})"

    def __str__(self) -> str:
        return f"|{self.species}:n={self.n}, {self.angular!s}⟩"

    @cached_property
    def nu(self) -> float:  # type: ignore [override]
        return self.sqdt.calc_nu(self.n, self.angular)

    @cached_property
    def _energy_au(self) -> float:  # type: ignore [override]
        return calc_energy_from_nu(self.element_properties.reduced_mass_au, self.nu) + self.sqdt.ionization_energy_au

    @cached_property
    def radial(self) -> RadialKet:
        """The radial part of the Rydberg electron."""
        return RadialKet(self.nu, self.potential, n_expected=self.n)

    @cached_property
    def coefficients(self) -> NDArray:  # type: ignore [override]
        return np.array([1.0])

    @cached_property
    def rydberg_kets(self) -> list[RydbergKet]:  # type: ignore [override]
        return [RydbergKet(self.angular, self.radial)]

    def free_memory(self) -> None:
        super().free_memory()
        # For SQDT the radial ket is held by these cached properties of the state itself,
        # so they have to be dropped as well to actually release the radial wavefunction.
        self.__dict__.pop("rydberg_kets", None)
        self.__dict__.pop("radial", None)

    @overload
    def get_radial_energy(self, unit: None = None) -> PintFloat: ...

    @overload
    def get_radial_energy(self, unit: str) -> float: ...

    def get_radial_energy(self, unit: str | None = None) -> PintFloat | float:
        r"""Get the energy of the radial part of the Rydberg state.

        The radial part of the energy is given by

        .. math::
            E = - \frac{1}{2} \frac{\mu}{\nu^2}

        where `\mu = R_M/R_\infty` is the reduced mass and `\nu` the effective principal quantum number.
        """
        _energy_au = calc_energy_from_nu(self.element_properties.reduced_mass_au, self.nu)
        if unit == "a.u.":
            return _energy_au
        energy: PintFloat = _energy_au * BaseQuantities["energy"]
        if unit is None:
            return energy
        return energy.to(unit, "spectroscopy").magnitude

    @overload
    def get_spontaneous_transition_rates(self: Self, unit: None = None) -> tuple[list[Self], PintArray]: ...

    @overload
    def get_spontaneous_transition_rates(self: Self, unit: str) -> tuple[list[Self], NDArray]: ...

    def get_spontaneous_transition_rates(self: Self, unit: str | None = None) -> tuple[list[Self], NDArray | PintArray]:
        """Calculate the spontaneous transition rates for the Rydberg state.

        The spontaneous transition rates are given by the Einstein A coefficients.

        Args:
            unit: The unit to which to convert the result.
                Default None will return a `pint.Quantity`.

        Returns:
            The relevant states and the transition rates.

        """
        relevant_states_masked, transition_rates_au = self._get_transition_rates_au(only_spontaneous=True)

        if unit == "a.u.":
            return relevant_states_masked, transition_rates_au
        transition_rates = ureg.Quantity(transition_rates_au, "1/atomic_unit_of_time")
        if unit is None:
            return relevant_states_masked, transition_rates
        return relevant_states_masked, transition_rates.to(unit).magnitude

    @overload
    def get_black_body_transition_rates(
        self: Self, temperature: float | PintFloat, temperature_unit: str | None = None, unit: None = None
    ) -> tuple[list[Self], PintArray]: ...

    @overload
    def get_black_body_transition_rates(
        self: Self, temperature: PintFloat, *, unit: str
    ) -> tuple[list[Self], NDArray]: ...

    @overload
    def get_black_body_transition_rates(
        self: Self, temperature: float, temperature_unit: str, unit: str
    ) -> tuple[list[Self], NDArray]: ...

    def get_black_body_transition_rates(
        self: Self, temperature: float | PintFloat, temperature_unit: str | None = None, unit: str | None = None
    ) -> tuple[list[Self], NDArray | PintArray]:
        """Calculate the black body transition rates of the Rydberg state.

        The black body transition rates are given by the Einstein B coefficients,
        with a weight factor given by Planck's law.

        Args:
            temperature: The temperature, for which to calculate the black body transition rates.
            temperature_unit: The unit of the temperature.
                Default None will assume the temperature is given as `pint.Quantity`.
            unit: The unit to which to convert the result.
                Default None will return a `pint.Quantity`.

        Returns:
            The relevant states and the transition rates.

        """
        temperature_au = ureg.Quantity(temperature, temperature_unit).to_base_units().magnitude
        relevant_states_masked, transition_rates_au = self._get_transition_rates_au(
            temperature_au, only_spontaneous=False
        )

        if unit == "a.u.":
            return relevant_states_masked, transition_rates_au
        transition_rates = ureg.Quantity(transition_rates_au, "1/atomic_unit_of_time")
        if unit is None:
            return relevant_states_masked, transition_rates
        return relevant_states_masked, transition_rates.to(unit).magnitude

    def _get_transition_rates_au(
        self: Self,
        temperature_au: float | None = None,
        *,
        only_spontaneous: bool = False,
    ) -> tuple[list[Self], NDArray]:
        r"""Calculate the transition rates in atomic units.

        The transition rates are given by the Einstein A coefficients for spontaneous transitions,
        and by the Einstein B coefficients with a weight factor given by Planck's law for black body transitions.

        Concretely the transition rates are calculated as

        .. math::
            \Gamma^{spontaneous}_{self \to other} = \frac{4}{3} \frac{\alpha}{c^2} \omega^3
                |\langle self || r^k_{radial} \hat{d}_q || other \rangle|^2

        where :math:`\alpha = 1/c` in atomic units.

        and

        .. math::
            \Gamma^{blackbody}_{self \to other} = \Gamma^{spontaneous}_{self \to other} \frac{1}{\exp(\omega / T) - 1}

        """
        if self.angular.coupling_scheme != "LS":
            raise NotImplementedError("Transition rates are currently only implemented for LS coupled states.")
        from rydstate.basis import BasisSQDT  # noqa: PLC0415

        m = self.angular.m
        if is_not_set(m):
            raise RuntimeError("m quantum number must be defined to calculate transition rates.")

        basis = BasisSQDT(
            self.species, n=(1, int(self.nu + 35)), m=(m - 1, m + 1), coupling_scheme=self.angular.coupling_scheme
        )
        basis.filter_states("l_r", (self.angular.l_r - 1, self.angular.l_r + 1))

        if only_spontaneous:
            basis.filter_states("nu", (0, self.nu))

        relevant_states = basis.states
        energy_differences_au = self.get_energy("a.u.") - np.array([s.get_energy("a.u.") for s in relevant_states])
        electric_dipole_moments_au = np.zeros(len(relevant_states))
        for q in [-1, 0, 1]:
            el_di_m = np.array(
                [s.calc_matrix_element(self, "electric_dipole", q, unit="a.u.") for s in relevant_states]
            )
            electric_dipole_moments_au += np.abs(el_di_m) ** 2

        transition_rates_au = (
            (4 / 3) * electric_dipole_moments_au / ureg.Quantity(1, "speed_of_light").to_base_units().magnitude ** 3
        )

        if only_spontaneous:
            transition_rates_au *= energy_differences_au**3
        else:
            assert temperature_au is not None, "Temperature must be given for black body transitions."
            if temperature_au == 0:
                transition_rates_au *= 0
            else:  # for numerical stability we use 1 / exprel(x) = x / (exp(x) - 1)
                transition_rates_au *= (
                    energy_differences_au**2 * temperature_au / exprel(energy_differences_au / temperature_au)
                )

        mask = transition_rates_au != 0
        relevant_states_masked = [ket for ket, is_relevant in zip(relevant_states, mask, strict=True) if is_relevant]
        transition_rates_au = transition_rates_au[mask]

        if np.any(transition_rates_au < 0):
            raise RuntimeError("Got negative transition rates, which should not happen.")

        return relevant_states_masked, transition_rates_au  # type: ignore [return-value]

    @overload
    def get_lifetime(
        self,
        temperature: float | PintFloat | None = None,
        temperature_unit: str | None = None,
        unit: None = None,
    ) -> PintFloat: ...

    @overload
    def get_lifetime(self, *, unit: str) -> float: ...

    @overload
    def get_lifetime(self, temperature: PintFloat, *, unit: str) -> float: ...

    @overload
    def get_lifetime(self, temperature: float, temperature_unit: str, unit: str) -> float: ...

    def get_lifetime(
        self,
        temperature: float | PintFloat | None = None,
        temperature_unit: str | None = None,
        unit: str | None = None,
    ) -> float | PintFloat:
        """Calculate the lifetime of the Rydberg state.

        The lifetime is the inverse of the sum of all transition rates.

        Args:
            temperature: The temperature, for which to calculate the black body transition rates.
                Default None will not include black body transitions.
            temperature_unit: The unit of the temperature.
                Default None will assume the temperature is given as `pint.Quantity`.
            unit: The unit to which to convert the result.
                Default None will return a `pint.Quantity`.

        Returns:
            The lifetime of the state.

        """
        _, transition_rates = self.get_spontaneous_transition_rates()
        transition_rates_au = transition_rates.to_base_units().magnitude
        if temperature is not None:
            _, black_body_transition_rates = self.get_black_body_transition_rates(temperature, temperature_unit)
            transition_rates_au = np.append(transition_rates_au, black_body_transition_rates.to_base_units().magnitude)

        lifetime_au: float = 1 / np.sum(transition_rates_au)

        if unit == "a.u.":
            return lifetime_au
        lifetime = ureg.Quantity(lifetime_au, "atomic_unit_of_time")
        if unit is None:
            return lifetime
        return lifetime.to(unit).magnitude


class RydbergStateSQDTAlkali(RydbergStateSQDT[AngularKetLS[AllKnown]]):
    """Create an Alkali Rydberg state, including the radial and angular states."""

    def __init__(
        self,
        species: str,
        n: int,
        *,
        l: int | None = None,
        j: float | None = None,
        m: float | NotSet = NotSet,
        angular_ket: AngularKetLS[AllKnown] | None = None,
        # potential and sqdt parameters
        sqdt: SQDT | str | None = None,
        potential: Potential | str | None = None,
    ) -> None:
        r"""Initialize the Rydberg state.

        Args:
            species: Atomic species.
            n: Principal quantum number of the rydberg electron.
            l: Orbital angular momentum quantum number of the rydberg electron.
            j: Angular momentum quantum number of the rydberg electron.
            m: Total magnetic quantum number.
              Optional, only needed for concrete angular matrix elements.
            angular_ket: The angular ket to use for the state.
              Either angular_ket or the quantum numbers for the angular ket must be given.
            sqdt: The SQDT to use for the state.
              Either a string representing the tag of the SQDT class to use,
              or an instance of an SQDT class.
            potential: The potential to use for the radial ket.
              Either a string representing the tag of the potential to use,
              or an instance of a potential class.

        """
        super().__init__(  # type: ignore [call-overload,misc]
            species=species, n=n, l_r=l, j_tot=j, m=m, angular_ket=angular_ket, sqdt=sqdt, potential=potential
        )
