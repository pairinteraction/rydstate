from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any, ClassVar, overload

from rydstate.species.fmodel import FModel, FModelSQDT
from rydstate.species.registry_singleton_meta import RegistrySingletonMeta
from rydstate.units import ureg

if TYPE_CHECKING:
    from rydstate.angular.angular_ket import AngularKetFJ
    from rydstate.angular.core_ket import CoreKet
    from rydstate.units import PintFloat


class MQDT(metaclass=RegistrySingletonMeta):
    """Base class for all MQDT classes."""

    species: ClassVar[str]
    """The short name of the atomic species."""
    tag: ClassVar[str]
    """The tag for these MQDT parameters."""

    ionization_threshold_dict: dict[CoreKet, tuple[float, float | None, str]]
    """Dictionary containing the ionization thresholds for the different core states.
    The thresholds are given in the form of a tuple (ionization_threshold, uncertainty, unit).
    """

    core_ground_state: CoreKet
    """The ground state configuration of the atomic core."""

    models: list[FModel]
    """List of MQDT FModel's available for this species."""

    models_file: str | None = None
    """A file containing all the MQDT models for this species.
    Specify either this attribute or the models attribute directly, but not both."""

    def __init__(self, species: str | None = None, tag: str | None = None) -> None:
        if getattr(self, "_initialized", False):
            return

        self._initialized = True

        if hasattr(self, "models"):
            if self.models_file is not None:
                raise ValueError("Either define the models attribute or the models_file attribute, not both.")
            return

        if self.models_file is None:
            raise ValueError("Either the models attribute or the models_file attribute must be specified.")

        # If models are not defined directly, try to load them from the specified file.
        # load the file and extract all FModel subclasses defined in it that match the species
        raise NotImplementedError("Loading MQDT models from a file is not implemented yet.")


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
            models = [FModelSQDT(self.species, outer_channel)]
        return models
