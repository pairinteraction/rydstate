from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from rydstate.angular.core_ket import CoreKet
from rydstate.species.mqdt import MQDT


class MQDTStrontium87(MQDT):
    species = "Sr87_mqdt"
    is_default = True
    i_c = 4.5

    ionization_threshold_dict: ClassVar = {
        CoreKet(i_c, 0.5, 0, 0.5, 4): (45932.287373577, None, "1/cm"),
        CoreKet(i_c, 0.5, 0, 0.5, 5): (45932.120512528, None, "1/cm"),
    }
    core_ground_state = CoreKet(i_c, 0.5, 0, 0.5, 4)
    nuclear_dipole = -1.0936030
    model_classes_file = Path(__file__).with_name("sr87_mqdt_fmodel_data.py")


class MQDTStrontium88(MQDT):
    species = "Sr88_mqdt"
    is_default = True
    i_c = 0

    ionization_threshold_dict: ClassVar = {
        CoreKet(i_c, 0.5, 0, 0.5): (45932.1956, None, "1/cm"),
    }
    core_ground_state = CoreKet(i_c, 0.5, 0, 0.5)
    nuclear_dipole = 2.3
    model_classes_file = Path(__file__).with_name("sr88_mqdt_fmodel_data.py")
