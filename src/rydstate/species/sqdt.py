from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from rydstate.species.registry_singleton_meta import RegistrySingletonMeta

if TYPE_CHECKING:
    from pathlib import Path

    from rydstate.species.utils import RydbergRitzParameters


class SQDT(metaclass=RegistrySingletonMeta):
    """Base class for all SQDT classes."""

    species: ClassVar[str]
    """The short name of the atomic species."""
    tag: ClassVar[str]
    """The tag for these SQDT parameters."""

    _quantum_defects: ClassVar[dict[tuple[int, float, float], RydbergRitzParameters] | None] = None
    """Dictionary containing the quantum defects for each (l, j_tot, s_tot) combination, i.e.
    _quantum_defects[(l,j_tot,s_tot)] = (d0, d2, d4, d6, d8)
    """

    _nist_energy_levels_file: Path | None = None
    """Path to the NIST energy levels file for this species.
    The file should be directly downloaded from https://physics.nist.gov/PhysRefData/ASD/levels_form.html
    in the 'Tab-delimited' format and in units of Hartree.
    """

    def __init__(self, species: str | None = None, tag: str | None = None) -> None:
        pass

    def __repr__(self) -> str:
        return f"SQDT({self.species}, {self.tag})"
