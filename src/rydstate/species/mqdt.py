from __future__ import annotations

import importlib.util
import inspect
import sys
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, overload

from rydstate.species.fmodel import FModel, FModelSQDT
from rydstate.units import ureg

if TYPE_CHECKING:
    from rydstate.angular.angular_ket import AngularKetFJ
    from rydstate.angular.core_ket import CoreKet
    from rydstate.units import PintFloat


class MQDT:
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

    model_classes: list[type[FModel]] | None = None
    """List of MQDT FModel classes available for this species."""

    model_classes_file: Path | str | None = None
    """A file containing all the MQDT model classes for this species.
    Specify either this attribute or the model_classes attribute directly, but not both."""

    def __init__(self) -> None:
        if self.model_classes is not None and self.model_classes_file is not None:
            raise ValueError("Either define the model_classes attribute or the model_classes_file attribute, not both.")
        if self.model_classes is None:
            if self.model_classes_file is None:
                raise ValueError("Either the model_classes or the model_classes_file attribute must be specified.")
            # If model classes are not defined directly, try to load them from the specified file.
            # load the file and extract all FModel subclasses defined in it that match the species
            self.model_classes = self._load_model_classes_from_file(Path(self.model_classes_file))

        self.models: list[FModel] = [model_class(self) for model_class in self.model_classes]

    def __repr__(self) -> str:
        return f"MQDT({self.species}, {self.tag})"

    @classmethod
    def _load_model_classes_from_file(cls, file: Path) -> list[type[FModel]]:
        if not file.is_absolute():
            defining_module = inspect.getmodule(cls)
            defining_module_file = getattr(defining_module, "__file__", None)
            if defining_module_file is None:
                raise ValueError(f"Cannot resolve relative MQDT models_file path {file!r}.")
            file = Path(defining_module_file).resolve().parent / file
        file = file.resolve()

        module_name = f"_rydstate_mqdt_models_{cls.__module__}_{cls.__name__}_{file.stem}"
        module_name = module_name.replace(".", "_")
        spec = importlib.util.spec_from_file_location(module_name, file)
        if spec is None or spec.loader is None:
            raise ValueError(f"Cannot load MQDT models from {file}.")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(module_name, None)

        model_classes = [
            model_cls
            for model_cls in module.__dict__.values()
            if (
                inspect.isclass(model_cls)
                and model_cls.__module__ == module.__name__
                and issubclass(model_cls, FModel)
                and model_cls is not FModel
                and getattr(model_cls, "species", None) == cls.species
            )
        ]
        if len(model_classes) == 0:
            raise ValueError(f"No MQDT model_classes for species {cls.species!r} found in {file}.")

        return model_classes

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
