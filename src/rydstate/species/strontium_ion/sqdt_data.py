from abc import ABC
from typing import ClassVar

from rydstate.species.sqdt import SQDT


class _SQDTStrontiumIon(SQDT, ABC):
    is_default = True
    # Sr II (Sr+) energy levels, referenced to the Sr+ ground state (4p6.5s 2S1/2)
    nist_data_file = "nist_data.txt"

    # https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Sr&units=1&at_num_out=on&el_name_out=on&seq_out=on&shells_out=on&level_out=on&e_out=0&unc_out=on&biblio=on
    ionization_energy = (11.0302765, "eV")

    quantum_defects: ClassVar = None


class SQDTStrontium87Ion(_SQDTStrontiumIon):
    species = "Sr87_ion"


class SQDTStrontium88Ion(_SQDTStrontiumIon):
    species = "Sr88_ion"
