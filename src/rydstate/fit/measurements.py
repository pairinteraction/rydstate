from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rydstate import units
from rydstate.angular.angular_ket import AngularKetBase, AngularKetLS
from rydstate.angular.short_label import format_short_label, parse_short_label

if TYPE_CHECKING:
    from collections.abc import Mapping
    from os import PathLike

MEASUREMENTS_COLUMNS = ("level", "lower_level", "value", "uncertainty", "unit", "source_label")
"""Required columns of measurements.csv, see :func:`load_measurements`."""


@dataclass(frozen=True)
class SQDTLevel:
    """A Rydberg level, identified by the short label of its channel (e.g. "np_3/2") and its principal quantum number.

    The channel label must be the name of the SQDT model describing the level, see :class:`~rydstate.qdt.ModelSQDT`.
    """

    channel: str
    n: int

    def __str__(self) -> str:
        return self.channel.replace("n", str(self.n), 1)

    @classmethod
    def from_label(cls, label: str, species: str) -> SQDTLevel:
        """Create a level from its short label including n, e.g. "45p_3/2" or "5s 40d 1D2".

        The label is the short label of the channel (see
        :meth:`~rydstate.angular.angular_ket.AngularKetBase.get_short_label`) with the principal quantum number n
        in place of the placeholder "n". The channel is stored with its canonical short label.

        Args:
            label: The short label of the level.
            species: The species, which determines the format of the label (single or two valence electrons).

        """
        for template in (AngularKetLS.short_label_template_single_valence, AngularKetLS.short_label_template):
            try:
                values = parse_short_label(template, label)
            except ValueError:
                continue
            if "n" in values:
                if extra := sorted(values.keys() & {"m", "label"}):
                    raise ValueError(
                        f"The label {label!r} of a level must not contain {extra}, "
                        "a level is given by its channel and n."
                    )
                channel = AngularKetBase.from_short_label(format_short_label(template, {**values, "n": None}), species)
                return cls(channel.get_short_label(), int(values["n"]))
        raise ValueError(f"The label {label!r} is no short label of a level with principal quantum number n.")


@dataclass(frozen=True)
class Measurement:
    """A measured energy of a Rydberg level, or a measured interval between two Rydberg levels.

    The measured value is E(level) if lower_level is None, i.e. the energy of the level measured from the ground state
    (the zero of the energies of rydstate), otherwise the interval E(level) - E(lower_level).
    The attributes are named like the columns of measurements.csv, see :func:`load_measurements`.
    """

    level: SQDTLevel
    value_au: float
    """Measured energy or interval in atomic units (Hartree)."""
    uncertainty_au: float
    """Uncertainty (standard deviation) of the measured value in atomic units (Hartree)."""
    lower_level: SQDTLevel | None = None
    dataset: str = ""
    """Name of the dataset the measurement belongs to, see :func:`~rydstate.fit.fit_sqdt_models` for free offsets."""
    source_label: str = ""
    """The label of the value in the source, unique within the dataset."""
    reference: str = ""
    """The publication of the measured value, e.g. "Shen 2024, Phys. Rev. Lett. 133, 233005, https://doi.org/..."."""

    def __post_init__(self) -> None:
        if not self.uncertainty_au > 0:
            raise ValueError(f"The uncertainty of the measurement {self} must be positive.")

    @property
    def levels(self) -> tuple[SQDTLevel, ...]:
        """The measured level, or the level and the lower level of an interval."""
        return (self.level,) if self.lower_level is None else (self.level, self.lower_level)

    @classmethod
    def from_row(cls, row: Mapping[str, str], species: str, dataset: str = "", reference: str = "") -> Measurement:
        """Create a measurement from a row of measurements.csv, see :func:`load_measurements`.

        Args:
            row: The row, with (at least) the columns of measurements.csv.
            species: The species of the levels, see :meth:`SQDTLevel.from_label`.
            dataset: The name of the dataset.
            reference: The publication of the measured value.

        """
        if not _is_proportional_to_energy(row["unit"]):
            raise ValueError(
                f"the unit {row['unit']!r} of {row['source_label']!r} is not "
                "an energy, frequency or wave number unit, see rydstate.fit.load_measurements."
            )
        return cls(
            level=SQDTLevel.from_label(row["level"], species),
            value_au=units.user_to_au(float(row["value"]), row["unit"], "energy"),
            uncertainty_au=units.user_to_au(float(row["uncertainty"]), row["unit"], "energy"),
            lower_level=SQDTLevel.from_label(row["lower_level"], species) if row["lower_level"] else None,
            dataset=dataset,
            source_label=row["source_label"],
            reference=reference,
        )


def load_measurements(directory: str | PathLike[str]) -> list[Measurement]:
    """Load the measured energies and intervals of a dataset.

    A dataset is a directory with the files ``metadata.json`` and ``measurements.csv``.
    The name of the directory is the name of the dataset (see :attr:`~rydstate.fit.Measurement.dataset`).
    ``metadata.json`` describes the source, at least by the keys "species" and "reference"
    (e.g. "Shen 2024, Phys. Rev. Lett. 133, 233005, https://doi.org/10.1103/PhysRevLett.133.233005"),
    see :func:`load_metadata`; the reference is used as :attr:`~rydstate.fit.Measurement.reference`.
    Every row of ``measurements.csv`` is one measured value (see :meth:`~rydstate.fit.Measurement.from_row`)
    with the columns

    - level: the short label of the level including n, e.g. "45p_3/2" or "5s 40d 1D2",
      see :meth:`~rydstate.fit.SQDTLevel.from_label`,
    - lower_level: empty for an energy of the level measured from the ground state
      (the zero of the energies of rydstate, i.e. the hyperfine centre of gravity of the ground state),
      otherwise the short label of the lower level of the measured interval E(level) - E(lower_level),
    - value, uncertainty, unit: the measured value and its uncertainty (standard deviation) in the unit,
      which can be any energy, frequency or wave number unit of :mod:`rydstate.units`, e.g. "1/cm", "GHz", "MHz"
      or "kHz"; wavelengths are not allowed (the uncertainty would not convert like the value, and published
      wavelengths are often air wavelengths), they must be converted to one of these units in the file;
      they are stored in atomic units as :attr:`~rydstate.fit.Measurement.value_au` and
      :attr:`~rydstate.fit.Measurement.uncertainty_au`,
    - source_label: the label of the value in the source, unique within the dataset,

    and optionally further columns (e.g. source_location and note), which are ignored.
    All corrections of the values (e.g. to the energy zero of rydstate) are already applied in the file.

    Args:
        directory: The directory of the dataset.

    Returns:
        The measurements of the dataset, in the order of the rows.

    """
    directory = Path(directory)
    path = directory / "measurements.csv"
    metadata = load_metadata(directory)
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        if missing := set(MEASUREMENTS_COLUMNS) - set(reader.fieldnames or []):
            raise ValueError(f"{path}: missing columns {sorted(missing)}.")
        rows = list(reader)
    source_labels = [row["source_label"] for row in rows]
    if len(set(source_labels)) != len(source_labels):
        raise ValueError(f"{path}: the source labels are not unique.")
    try:
        return [Measurement.from_row(row, metadata["species"], directory.name, metadata["reference"]) for row in rows]
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error


def _is_proportional_to_energy(unit: str) -> bool:
    """Return whether values in the unit are proportional to the energy, so their uncertainties convert likewise.

    Angular units (e.g. rad/s) are not allowed: pint treats radians as dimensionless, so an angular frequency
    would be converted as E = h f instead of E = hbar omega.
    """
    quantity = units.ureg.Quantity(1, unit)
    if "radian" in dict(quantity.to_root_units().unit_items()):
        return False
    return any(quantity.check(dimension) for dimension in ("[energy]", "[frequency]", "1/[length]"))


def load_metadata(directory: str | PathLike[str]) -> dict[str, Any]:
    """Load the metadata.json of a dataset, which must contain at least the keys "species" and "reference".

    Args:
        directory: The directory of the dataset, see :func:`load_measurements`.

    """
    path = Path(directory) / "metadata.json"
    metadata: dict[str, Any] = json.loads(path.read_text())
    missing = {"species", "reference"} - set(metadata)
    if missing:
        raise ValueError(f"{path}: missing keys {sorted(missing)}.")
    return metadata
