from __future__ import annotations

import logging
from functools import cached_property
from typing import TYPE_CHECKING, Any, ClassVar, overload

import numpy as np

from rydstate.angular.core_ket import CoreKet
from rydstate.angular.utils import is_unknown
from rydstate.species.mqdt.fmodel import FModel, FModelSQDT
from rydstate.species.species_object import SpeciesObject
from rydstate.species.sqdt.species_object_sqdt import SpeciesObjectSQDT
from rydstate.species.utils import calc_energy_from_nu, calc_nu_from_energy
from rydstate.units import ureg

if TYPE_CHECKING:
    from rydstate.angular.angular_ket import AngularKetFJ
    from rydstate.angular.core_ket import CoreKet
    from rydstate.units import PintFloat

logger = logging.getLogger(__name__)


class SpeciesObjectMQDT(SpeciesObject):
    i_c: ClassVar[float]

    _ionization_threshold_dict: dict[CoreKet, tuple[float, float | None, str]]
    """Dictionary containing the ionization thresholds for the different core states.
    The thresholds are given in the form of a tuple (ionization_threshold, uncertainty, unit).
    """

    core_ground_state: CoreKet
    """The ground state configuration of the atomic core."""

    nuclear_dipole: float
    """Nuclear dipole moment of the species."""

    def is_allowed_shell(self, n: int, l: int, s_tot: float | None = None) -> bool:
        if is_unknown(l):
            return True
        sqdt_species = SpeciesObjectSQDT.from_name(self.name.replace("_mqdt", ""))
        return sqdt_species.is_allowed_shell(n, l, s_tot)

    @cached_property
    def models(self) -> list[FModel]:
        """List of MQDT models available for the species."""
        models = [model for model in FModel.__subclasses__() if getattr(model, "species_name", None) == self.name]
        return [model() for model in models]

    @overload
    def get_ionization_threshold(self, core_ket: CoreKet, unit: None = None) -> PintFloat: ...

    @overload
    def get_ionization_threshold(self, core_ket: CoreKet, unit: str) -> float: ...

    def get_ionization_threshold(self, core_ket: CoreKet, unit: str | None = "hartree") -> PintFloat | float:
        """Return the ionization energy of the channel given by the core_ket in the desired unit.

        Args:
            core_ket: The core ket for which to return the ionization energy.
            unit: Desired unit for the ionization energy. Default is atomic units "hartree".

        Returns:
            Ionization energy in the desired unit.

        """
        try:
            matching_core_ket = core_ket.find_matching_core_ket(self._ionization_threshold_dict.keys())
        except ValueError as e:
            raise ValueError(f"Ionization energy for core ket {core_ket} is not defined.") from e

        ionization_energy_tuple = self._ionization_threshold_dict[matching_core_ket]
        ionization_energy: PintFloat = ureg.Quantity(ionization_energy_tuple[0], ionization_energy_tuple[2])
        ionization_energy = ionization_energy.to("hartree", "spectroscopy")
        if unit is None:
            return ionization_energy
        if unit == "a.u.":
            return ionization_energy.magnitude
        return ionization_energy.to(unit, "spectroscopy").magnitude

    @cached_property
    def reference_ionization_energy_au(self) -> float:
        """Ionization energy in atomic units (Hartree)."""
        return self.get_ionization_threshold(self.core_ground_state, unit="hartree")

    def get_mqdt_models(self, outer_channel: AngularKetFJ[Any]) -> list[FModel]:
        """Return a list of MQDT models for the outer_channel."""
        models = [model for model in self.models if any(ket == outer_channel for ket in model.outer_channels)]
        if len(models) == 0:
            models = [FModelSQDT(self.name, outer_channel)]
        return models

    def calc_nu(self, n: int, angular_ket: AngularKetFJ[Any], prev_nus: list[float] | None = None) -> float:
        fitting_models = [model for model in self.models if any(ket == angular_ket for ket in model.outer_channels)]
        if len(fitting_models) == 0:
            return n

        fitting_models = [model for model in fitting_models if model.nu_range[0] <= n <= model.nu_range[1]]
        if len(fitting_models) == 0:
            logger.warning(
                "calc_nu MQDT models found for %s %s, but not for n=%d, returning n.", self.name, angular_ket, n
            )
            return n
        if len(fitting_models) > 1:
            logger.warning(
                "Multiple MQDT models found for %s %s and n=%d, using the first one. Models: %s",
                *(self.name, angular_ket, n, fitting_models),
            )

        fmodel = fitting_models[0]
        ind = fmodel.outer_channels.index(angular_ket)

        prev_nus = [n] if prev_nus is None else prev_nus
        nu_ref = _calc_nu_ref(self, angular_ket, prev_nus[-1])

        trafo = fmodel.calc_frame_transformation(nu_ref)
        mu_closecoupling = np.diag(fmodel.calc_eigen_quantum_defects(nu_ref))
        mu_fj = trafo @ mu_closecoupling @ trafo.T

        mu1 = mu_fj[ind, ind]

        k = trafo @ np.tan(np.pi * mu_closecoupling) @ trafo.T
        tan_mu = k[ind, ind]
        mu = np.arctan(tan_mu) / np.pi
        # TODO this is a bit hacky
        mu += round(mu1)

        nu = float(n - mu)
        prev_nus.append(nu)
        if abs(prev_nus[-1] - prev_nus[-2]) < 1e-3:
            return nu

        if len(prev_nus) > 10:
            logger.warning(
                "calc_nu did not converge for %s at %d after %s iterations, returning last value %f.",
                *(angular_ket, n, prev_nus, nu),
            )
            return nu

        return self.calc_nu(n, angular_ket, prev_nus=prev_nus)


def _calc_nu_ref(species: SpeciesObjectMQDT, angular_ket: AngularKetFJ[Any], prev_nu: float) -> float:
    core_energy = species.get_ionization_threshold(angular_ket.get_core_ket(), unit="hartree")
    reference_core_energy = species.reference_ionization_energy_au

    eps_i = calc_energy_from_nu(species.reduced_mass_au, prev_nu)
    # E_tot = I_i + eps_i = I_ref + eps_ref with eps_i = -1 / (2 * (nu_i^2))
    # => eps_ref = eps_i + I_i - I_ref
    eps_ref = eps_i + core_energy - reference_core_energy
    if eps_ref >= 0:
        # channel state is above reference ionization threshold
        return 120.0  # large nu_ref value for continuum states
        # TODO larger value breaks for Yb171

    return calc_nu_from_energy(species.reduced_mass_au, eps_ref)
