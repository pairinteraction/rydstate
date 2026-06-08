from __future__ import annotations

from functools import cache, cached_property
from typing import TYPE_CHECKING, Any, ClassVar, overload

from rydstate.species.fmodel import FModelSQDT
from rydstate.species.utils import get_all_subclasses
from rydstate.units import ureg

if TYPE_CHECKING:
    from rydstate.angular.angular_ket import AngularKetFJ
    from rydstate.angular.core_ket import CoreKet
    from rydstate.species.fmodel import FModel
    from rydstate.species.utils import cache  # type: ignore [assignment]  # noqa: TC004
    from rydstate.units import PintFloat


class MQDT:
    """Base class for all MQDT classes."""

    species: ClassVar[str]
    """The short name of the atomic species."""
    tag: ClassVar[str]
    """The tag for these MQDT parameters."""

    ionization_threshold_dict: ClassVar[dict[CoreKet, tuple[float, str]]]
    """Dictionary containing the ionization thresholds for the different core states.
    The thresholds are given in the form of a tuple (ionization_threshold, unit).
    """

    model_classes: list[type[FModel]]
    """List of the MQDT :class:`~rydstate.species.fmodel.FModel` models available for this species.

    :meta hide-value:

    """

    def __init__(self) -> None:
        self.models: list[FModel] = [model_class(self) for model_class in self.model_classes]

    def __repr__(self) -> str:
        return f"MQDT({self.species}, {self.tag})"

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
            matching_core_ket = core_ket.find_matching_core_ket(self.ionization_threshold_dict.keys())
        except ValueError as e:
            raise ValueError(f"Ionization energy for core ket {core_ket} is not defined.") from e

        ionization_energy_tuple = self.ionization_threshold_dict[matching_core_ket]
        ionization_energy: PintFloat = ureg.Quantity(ionization_energy_tuple[0], ionization_energy_tuple[1])
        ionization_energy = ionization_energy.to("hartree", "spectroscopy")
        if unit is None:
            return ionization_energy
        if unit == "a.u.":
            return ionization_energy.magnitude
        return ionization_energy.to(unit, "spectroscopy").magnitude

    @cached_property
    def reference_ionization_energy_au(self) -> float:
        """Reference ionization energy in atomic units (Hartree).

        We define the reference ionization energy as the smallest ionization energy in the ionization_threshold_dict.
        """
        return min(self.get_ionization_threshold(core_ket, unit="a.u.") for core_ket in self.ionization_threshold_dict)

    def get_mqdt_models(self, outer_channel: AngularKetFJ[Any]) -> list[FModel]:
        """Return a list of MQDT models for the outer_channel."""
        models = [model for model in self.models if any(ket == outer_channel for ket in model.outer_channels)]
        if len(models) == 0:
            models = [FModelSQDT(self.species, outer_channel, mqdt=self)]
        return models


@cache
def get_mqdt(species: str, tag: str | None = None) -> MQDT:
    """Get an instance of the subclass of MQDT for the given species and tag."""
    subclasses = get_all_subclasses(MQDT, species, tag)

    if tag is None:
        subclasses = [cls for cls in subclasses if getattr(cls, "is_default", False)]

    if len(subclasses) == 0:
        raise ValueError(f"No subclass of MQDT found for {species=} and {tag=}.")
    if len(subclasses) == 1:
        return subclasses[0]()
    raise ValueError(f"Multiple subclasses of MQDT found for {species=} and {tag=}: {subclasses}.")
