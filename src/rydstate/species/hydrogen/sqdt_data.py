from typing import ClassVar

from rydstate.species.sqdt import SQDT
from rydstate.units import rydberg_constant


class SQDTHydrogen(SQDT):
    species = "H"
    is_default = True

    # https://webbook.nist.gov/cgi/inchi?ID=C1333740&Mask=20
    ionization_energy = (15.425_93, "eV")

    quantum_defects: ClassVar = {}

    def _setup_nist_energy_levels(self) -> None:
        self._nist_energy_levels = {}


class SQDTHydrogenTextBook(SQDT):
    species = "H_textbook"
    is_default = True

    ionization_energy = (rydberg_constant.m, str(rydberg_constant.u))

    quantum_defects: ClassVar = {}

    def _setup_nist_energy_levels(self) -> None:
        self._nist_energy_levels = {}
