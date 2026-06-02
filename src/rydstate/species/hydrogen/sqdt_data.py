from typing import ClassVar

from rydstate.species.sqdt import SQDT
from rydstate.units import rydberg_constant


class SQDTHydrogen(SQDT):
    species = "H"
    is_default = True

    quantum_defects: ClassVar = {}

    # https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=H&units=1&at_num_out=on&el_name_out=on&seq_out=on&shells_out=on&level_out=on&e_out=0&unc_out=on&biblio=on
    ionization_energy = (13.598_434_599_702, "eV")

    def _setup_nist_energy_levels(self) -> None:
        self._nist_energy_levels = {}


class SQDTHydrogenTextBook(SQDT):
    species = "H_textbook"
    is_default = True

    quantum_defects: ClassVar = {}

    ionization_energy = (rydberg_constant.m, str(rydberg_constant.u))

    def _setup_nist_energy_levels(self) -> None:
        self._nist_energy_levels = {}
