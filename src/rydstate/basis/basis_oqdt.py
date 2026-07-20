from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from rydstate.angular import NotSet
from rydstate.angular.angular_ket import AngularKetFJ, AngularKetJJ, AngularKetLS
from rydstate.angular.utils import get_possible_quantum_number_values, is_unknown
from rydstate.basis.basis_base import BasisBase
from rydstate.basis.utils import get_m_range, is_allowed_qn
from rydstate.linalg import find_roots
from rydstate.radial.radial_ket import RadialDummy, RadialKet
from rydstate.rydberg_state.rydberg_ket import RydbergKet
from rydstate.rydberg_state.rydberg_mqdt import RydbergStateMQDT
from rydstate.species.mqdt import MQDT, get_mqdt
from rydstate.species.potential import Potential, get_potential_class

if TYPE_CHECKING:
    from rydstate.angular.angular_ket import AngularKetBase
    from rydstate.angular.core_ket import CoreKet
    from rydstate.species.fmodel import FModel


logger = logging.getLogger(__name__)


class BasisOQDT(BasisBase[RydbergStateMQDT]):
    """Basis for Rydberg states in the FJ coupling scheme, using Outer Channel QDT."""

    states: list[RydbergStateMQDT]

    def __init__(
        self,
        species: str,
        nu: tuple[float, float],
        *,
        f_tot: tuple[float, float] | None = None,
        l_r: tuple[int, int] | None = None,
        m: tuple[float, float] | None | NotSet = NotSet,
        # potential and qdt parameters
        mqdt: MQDT | str | None = None,
        potential_class: type[Potential] | str | None = None,
    ) -> None:
        super().__init__(species)
        self.mqdt = mqdt if isinstance(mqdt, MQDT) else get_mqdt(species, tag=mqdt)

        if isinstance(potential_class, type) and issubclass(potential_class, Potential):
            self.potential_class = potential_class
        else:
            self.potential_class = get_potential_class(species, tag=potential_class)

        self._init_states(nu, f_tot, l_r, m)

    def _init_states(
        self,
        nu_range: tuple[float, float],
        f_tot_range: tuple[float, float] | None,
        l_r_range: tuple[int, int] | None,
        m_range: tuple[float, float] | None | NotSet,
    ) -> None:
        self.states = []

        for core_ket in self.mqdt.get_core_kets():
            logger.info("Generating states for core ket: %s", core_ket)
            if is_unknown(core_ket.l_c):
                self._add_states_unknown(core_ket, nu_range, f_tot_range, m_range)
                continue

            for l_r in range(int(nu_range[1]) + 10):
                if not is_allowed_qn(l_r_range, l_r):
                    continue

                if not core_ket.contains_unknown:
                    self._add_states_fj(core_ket, nu_range, l_r, f_tot_range, m_range)
                elif not is_unknown(core_ket.j_c):
                    self._add_states_jj(core_ket, nu_range, l_r, f_tot_range, m_range)
                elif not is_unknown(core_ket.l_c):
                    self._add_states_ls(core_ket, nu_range, l_r, f_tot_range, m_range)
                else:
                    raise RuntimeError("This should never happen.")
        self.sort_states("nu")

    def _add_states_fj(
        self,
        core_ket: CoreKet,
        nu_range: tuple[float, float],
        l_r: int,
        f_tot_range: tuple[float, float] | None,
        m_range: tuple[float, float] | None | NotSet = NotSet,
    ) -> None:
        s_r = 0.5

        for j_r in get_possible_quantum_number_values(l_r, s_r, None):
            for f_tot in get_possible_quantum_number_values(core_ket.f_c, j_r, None):
                if is_unknown(f_tot):
                    raise ValueError("Cannot determine f_tot for BasisOQDT (FJ).")
                if not is_allowed_qn(f_tot_range, f_tot):
                    continue

                angular_ket = AngularKetFJ(
                    l_c=core_ket.l_c,
                    j_c=core_ket.j_c,
                    l_r=l_r,
                    j_r=j_r,
                    f_c=core_ket.f_c,
                    f_tot=f_tot,
                    species=self.species,
                )

                self._create_states(nu_range, angular_ket, m_range)

    def _add_states_jj(
        self,
        core_ket: CoreKet,
        nu_range: tuple[float, float],
        l_r: int,
        f_tot_range: tuple[float, float] | None,
        m_range: tuple[float, float] | None | NotSet = NotSet,
    ) -> None:
        assert is_unknown(core_ket.f_c)
        assert not is_unknown(core_ket.j_c)
        s_r = 0.5

        for j_r in get_possible_quantum_number_values(l_r, s_r, None):
            for j_tot in get_possible_quantum_number_values(core_ket.j_c, j_r, None):
                for f_tot in get_possible_quantum_number_values(j_tot, core_ket.i_c, None):
                    if is_unknown(f_tot):
                        raise ValueError("Cannot determine f_tot for BasisOQDT (JJ).")
                    if not is_allowed_qn(f_tot_range, f_tot):
                        continue

                    angular_ket = AngularKetJJ(
                        l_c=core_ket.l_c,
                        j_c=core_ket.j_c,
                        l_r=l_r,
                        j_r=j_r,
                        j_tot=j_tot,
                        f_tot=f_tot,
                        species=self.species,
                    )

                    self._create_states(nu_range, angular_ket, m_range)

    def _add_states_ls(
        self,
        core_ket: CoreKet,
        nu_range: tuple[float, float],
        l_r: int,
        f_tot_range: tuple[float, float] | None,
        m_range: tuple[float, float] | None | NotSet = NotSet,
    ) -> None:
        assert is_unknown(core_ket.j_c)
        s_r = 0.5

        for s_tot in get_possible_quantum_number_values(core_ket.s_c, s_r, None):
            for l_tot in get_possible_quantum_number_values(core_ket.l_c, l_r, None):
                for j_tot in get_possible_quantum_number_values(s_tot, l_tot, None):
                    for f_tot in get_possible_quantum_number_values(j_tot, core_ket.i_c, None):
                        if is_unknown(f_tot):
                            raise ValueError("Cannot determine f_tot for BasisOQDT (LS).")
                        if not is_allowed_qn(f_tot_range, f_tot):
                            continue

                        angular_ket = AngularKetLS(
                            l_c=core_ket.l_c,
                            l_r=l_r,
                            s_tot=s_tot,
                            l_tot=l_tot,  # type: ignore [arg-type]
                            j_tot=j_tot,
                            f_tot=f_tot,
                            species=self.species,
                        )

                        self._create_states(nu_range, angular_ket, m_range)

    def _add_states_unknown(
        self,
        core_ket: CoreKet,
        nu_range: tuple[float, float],
        f_tot_range: tuple[float, float] | None,
        m_range: tuple[float, float] | None | NotSet = NotSet,
    ) -> None:
        assert is_unknown(core_ket.l_c)

        fmodels = [model for model in self.mqdt.models if core_ket in model.get_core_kets()]
        angular_kets = {
            channel for model in fmodels for channel in model.outer_channels if channel.get_core_ket() == core_ket
        }

        for angular_ket in angular_kets:
            if not is_allowed_qn(f_tot_range, angular_ket.f_tot):
                continue
            self._create_states(nu_range, angular_ket, m_range)

    def _create_states(
        self,
        nu_range: tuple[float, float],
        angular_ket: AngularKetBase[Any],
        m_range: tuple[float, float] | None | NotSet,
    ) -> None:
        for model in self.mqdt.get_mqdt_models(angular_ket):
            if angular_ket not in model.outer_channels:
                logger.warning(
                    "Angular ket %s has overlap but was not found in model %s outer channels.", angular_ket, model.name
                )
                continue
            states = get_oqdt_states_from_fmodel(angular_ket, model, nu_range, m_range, self.potential_class)
            self.states.extend(states)


def get_oqdt_states_from_fmodel(
    angular_ket: AngularKetBase[Any],
    model: FModel,
    nu_range: tuple[float, float],
    m_range: tuple[float, float] | None | NotSet,
    potential_class: type[Potential],
) -> list[RydbergStateMQDT]:
    """Calculate MQDT states from an FModel by finding zeros of the OQDT condition (see model._calc_oqdt_condition).

    Args:
        angular_ket: The angular ket to compute states for.
        model: The MQDT model to compute states for.
        nu_range: Tuple of (nu_min, nu_max) for the search range.
        m_range: Tuple of (m_min, m_max) for the magnetic quantum number range.
            NotSet will only include states with m=NotSet.
        potential_class: The potential class to use for the radial ket.

    Returns:
        List of :class:`RydbergStateMQDT` objects, one per root of the OQDT condition.

    """
    nu_min = max(nu_range[0], model.nu_min)
    nu_max = min(nu_range[1], model.nu_max)
    if np.isinf(nu_max):
        raise ValueError("nu_max must be finite to calculate MQDT states.")
    if nu_min > nu_max:
        return []

    ind = next((i for i, ket in enumerate(model.outer_channels) if ket == angular_ket), None)
    if ind is None:
        raise ValueError(f"Angular ket {angular_ket} not found in model {model.name} outer channels.")
    nu_list = find_roots(lambda nu: model._calc_oqdt_condition(nu, ind), nu_min, nu_max)  # noqa: SLF001

    if len(nu_list) == 0:
        logger.warning(
            "No MQDT states found in the range nu_min=%s, nu_max=%s for model %s", nu_min, nu_max, model.name
        )
        return []

    states: list[RydbergStateMQDT] = []
    for nu in nu_list:
        nuis = model.calc_channel_nuis(nu)
        nui = nuis[ind]

        radial: RadialKet | RadialDummy
        if not is_unknown(angular_ket.l_r):
            potential = potential_class(angular_ket.l_r)
            radial = RadialKet(float(nui), potential, sign_convention="positive_at_outer_bound")
        else:
            radial = RadialDummy(1.0, nui)

        energy_au = model.calc_energy_au(nu)
        for m in get_m_range(model.f_tot, m_range):
            rydberg_ket = RydbergKet(model.species, angular_ket.replace_m(m), radial)
            states.append(
                RydbergStateMQDT(
                    model.species,
                    [1],
                    [rydberg_ket],
                    nu=nu,
                    energy_au=energy_au,
                    mqdt=model.mqdt,
                    potential_class=potential_class,
                )
            )

    return states
