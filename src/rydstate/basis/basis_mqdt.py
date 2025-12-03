from __future__ import annotations

import logging
from pathlib import Path

from rydstate.angular.angular_ket import julia_qn_to_dict
from rydstate.basis.basis_base import BasisBase
from rydstate.rydberg.rydberg_mqdt import RydbergStateMQDT
from rydstate.rydberg.rydberg_sqdt import RydbergStateSQDT

try:
    USE_JULIACALL = True
    from juliacall import (
        Main as jl,  # noqa: N813
        convert,
    )
except ImportError:
    USE_JULIACALL = False


logger = logging.getLogger(__name__)

if USE_JULIACALL:
    # TODO also try except and print some meaningful warning if it fails
    jl.seval("using MQDT")
    jl.seval("using CGcoefficient")
    jl.include(str(Path(__file__).parent / "tables.jl"))


class BasisMQDT(BasisBase):
    def __init__(self, species: str, n_min: int = 0, n_max: int | None = None, *, skip_high_l: bool = True):
        super().__init__(species)

        if n_max is None:
            raise ValueError("n_max must be given")

        # initialize Wigner symbol calculation
        if skip_high_l:
            jl.CGcoefficient.wigner_init_float(5, "Jmax", 9)
        else:
            jl.CGcoefficient.wigner_init_float(n_max - 1, "Jmax", 9)

        parameters = jl.PARA_TABLE[species]

        logger.debug("Calculating low l MQDT states...")
        models = jl.MODELS_TABLE[species]
        states = [jl.eigenstates(n_min, n_max, M, parameters) for M in models]

        if skip_high_l:
            logger.debug("Skipping high l states.")
        else:
            logger.debug("Calculating high l SQDT states...")
            l_max = n_max - 1
            l_start = jl.FMODEL_MAX_L[species] + 1
            high_l_models = jl.single_channel_models(species, range(l_start, l_max + 1), parameters)
            high_l_states = [jl.eigenstates(n_min, n_max, M, parameters) for M in high_l_models]
            states = jl.vcat(states, high_l_states)
            models = jl.vcat(models, high_l_models)

        jl_states = convert(jl.Vector, states)

        jl_basis = jl.basisarray(jl_states, models)
        logger.debug("Generated state table with %d states", len(jl_basis.states))

        mqdt_states: list[RydbergStateMQDT] = []

        for jl_state in jl_basis.states:
            coeffs = jl_state.coeff
            nus = jl_state.nu
            nu_energy = jl_state.energy
            qns = jl_state.channels.i
            qns = [julia_qn_to_dict(qn) for qn in qns]

            sqdt_states = [RydbergStateSQDT(species, nu=nu, **qn) for nu, qn in zip(nus, qns)]
            # check angular and radial are created correctly
            [(s.angular, s.radial) for s in sqdt_states]

            mqdt_state = RydbergStateMQDT(coeffs, sqdt_states, nu_energy=nu_energy, warn_if_not_normalized=False)
            mqdt_states.append(mqdt_state)

        self.states = mqdt_states
