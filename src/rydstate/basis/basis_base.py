from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, get_args, overload

import numpy as np
from typing_extensions import Self

from rydstate.angular.angular_matrix_element import AngularMomentumQuantumNumbers
from rydstate.species.species_object import SpeciesObject

if TYPE_CHECKING:
    from rydstate.rydberg import RydbergStateMQDT, RydbergStateSQDT
    from rydstate.rydberg.rydberg_base import RydbergStateBase
    from rydstate.units import MatrixElementOperator, NDArray, PintArray


class BasisBase(ABC):
    states: list[RydbergStateSQDT | RydbergStateMQDT]

    def __init__(self, species: str | SpeciesObject) -> None:
        if isinstance(species, str):
            species = SpeciesObject.from_name(species)
        self.species = species

    def __len__(self) -> int:
        return len(self.states)

    def filter_states(self, qn: str, qn_mi: float, qn_max: float) -> Self:
        if qn in get_args(AngularMomentumQuantumNumbers):
            self.states = [state for state in self.states if qn_mi <= state.angular.calc_exp_qn(qn) <= qn_max]
        elif qn in ["n", "nu", "nu_energy"]:
            self.states = [state for state in self.states if qn_mi <= getattr(state, qn) <= qn_max]
        else:
            raise ValueError(f"Unknown quantum number {qn}")

        return self

    def calc_exp_qn(self, qn: str) -> list[float]:
        if qn in get_args(AngularMomentumQuantumNumbers):
            return [state.angular.calc_exp_qn(qn) for state in self.states]
        if qn in ["n", "nu", "nu_energy"]:
            return [getattr(state, qn) for state in self.states]
        raise ValueError(f"Unknown quantum number {qn}")

    def calc_std_qn(self, qn: str) -> list[float]:
        if qn in get_args(AngularMomentumQuantumNumbers):
            return [state.angular.calc_std_qn(qn) for state in self.states]
        if qn in ["n", "nu", "nu_energy"]:
            return [0 for state in self.states]
        raise ValueError(f"Unknown quantum number {qn}")

    def calc_reduced_overlap(self, other: RydbergStateBase) -> NDArray:
        """Calculate the reduced overlap <self|other> (ignoring the magnetic quantum number m)."""
        return np.array([bra.calc_reduced_overlap(other) for bra in self.states])

    def calc_reduced_overlaps(self, other: BasisBase) -> NDArray:
        """Calculate the reduced overlap <bra|ket> for all states in the bases self and other.

        Returns a numpy array overlaps, where overlaps[i,j] corresponds to the overlap of the
        i-th state of self and the j-th state of other.
        """
        return np.array([[bra.calc_reduced_overlap(ket) for ket in other.states] for bra in self.states])

    @overload
    def calc_reduced_matrix_element(
        self, other: RydbergStateBase, operator: MatrixElementOperator, unit: None = None
    ) -> PintArray: ...

    @overload
    def calc_reduced_matrix_element(
        self, other: RydbergStateBase, operator: MatrixElementOperator, unit: str
    ) -> NDArray: ...

    def calc_reduced_matrix_element(
        self, other: RydbergStateBase, operator: MatrixElementOperator, unit: str | None = None
    ) -> PintArray | NDArray:
        r"""Calculate the reduced matrix element.

        See also ...

        """
        return np.array([bra.calc_reduced_matrix_element(other, operator, unit=unit) for bra in self.states])

    @overload
    def calc_reduced_matrix_elements(
        self, other: RydbergStateBase, operator: MatrixElementOperator, unit: None = None
    ) -> PintArray: ...

    @overload
    def calc_reduced_matrix_elements(
        self, other: RydbergStateBase, operator: MatrixElementOperator, unit: str
    ) -> NDArray: ...

    def calc_reduced_matrix_elements(
        self, other: RydbergStateBase, operator: MatrixElementOperator, unit: str | None = None
    ) -> PintArray | NDArray:
        r"""Calculate the reduced matrix element.

        See also ...

        """
        return np.array(
            [[bra.calc_reduced_matrix_element(ket, operator, unit=unit) for ket in other.states] for bra in self.states]
        )
