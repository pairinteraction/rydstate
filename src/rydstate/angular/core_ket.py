from __future__ import annotations

from typing import TYPE_CHECKING

from rydstate.angular.utils import is_unknown, try_trivial_spin_addition

if TYPE_CHECKING:
    from rydstate.angular.utils import Unknown


class CoreKet:
    __slots__ = ("i_c", "s_c", "l_c", "j_c", "f_c", "name")

    def __init__(
        self,
        i_c: float | None = None,
        s_c: float | None = None,
        l_c: Unknown | int | None = None,
        j_c: Unknown | float | None = None,
        f_c: Unknown | float | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the core angular ket."""
        if i_c is None:
            raise ValueError("Nuclear spin i_c must be set.")
        self.i_c = float(i_c)

        if s_c is None:
            raise ValueError("Core spin s_c must be set.")
        self.s_c = float(s_c)

        if l_c is None:
            raise ValueError("Core orbital angular momentum l_c must be set.")
        self.l_c = int(l_c) if not is_unknown(l_c) else l_c

        self.j_c = try_trivial_spin_addition(self.l_c, self.s_c, j_c)
        self.f_c = try_trivial_spin_addition(self.j_c, self.i_c, f_c)

        self.name = name

        if any(is_unknown(x) for x in (self.l_c, self.j_c, self.f_c)) and name is None:
            raise ValueError("Name must be set if any of l_c, j_c, or f_c is unknown.")

    def __repr__(self) -> str:
        name_str = f", name={self.name!r}" if self.name is not None else ""
        return f"CoreKet(i_c={self.i_c}, s_c={self.s_c}, l_c={self.l_c}, j_c={self.j_c}, f_c={self.f_c}{name_str})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CoreKet):
            return NotImplemented
        return (
            self.i_c == other.i_c
            and self.s_c == other.s_c
            and self.l_c == other.l_c
            and self.j_c == other.j_c
            and self.f_c == other.f_c
            and self.name == other.name
        )

    def __hash__(self) -> int:
        return hash((str(type(self)), self.i_c, self.s_c, self.l_c, self.j_c, self.f_c, self.name))
