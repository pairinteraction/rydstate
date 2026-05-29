from __future__ import annotations

from typing import ClassVar

from rydstate.species.registry_singleton_meta import RegistrySingletonMeta


class Potential(metaclass=RegistrySingletonMeta):
    """Base class for all potential classes."""

    species: ClassVar[str]
    """The short name of the atomic species."""
    tag: ClassVar[str]
    """The tag for this potential."""

    def __init__(self, species: str | None = None, tag: str | None = None) -> None:
        pass

    def __repr__(self) -> str:
        return f"Potential({self.species}, {self.tag})"


class PotentialMarinescu1993(Potential):
    """Model potential for alkali atoms from Marinescu et al. (1994).

    See also: Phys. Rev. A 49, 982 (1994)
    """

    tag = "marinescu_1993"


class PotentialFei2009(Potential):
    """Model potential for alkaline earth atoms from Fei et al. (2009).

    See also: Phys. Rev. A 79, 052507 (2009)
    """

    tag = "fei_2009"
