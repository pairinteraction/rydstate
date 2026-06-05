from __future__ import annotations

from typing import ClassVar

from rydstate.angular.core_ket import CoreKet
from rydstate.species.fmodel import get_fmodels
from rydstate.species.mqdt import MQDT
from rydstate.species.strontium import sr87_mqdt_fmodel_data, sr88_mqdt_fmodel_data


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
    model_classes = get_fmodels(sr87_mqdt_fmodel_data, species)


class MQDTStrontium88(MQDT):
    species = "Sr88_mqdt"
    is_default = True
    i_c = 0

    ionization_threshold_dict: ClassVar = {
        CoreKet(i_c, 0.5, 0, 0.5): (45932.1956, None, "1/cm"),
    }
    core_ground_state = CoreKet(i_c, 0.5, 0, 0.5)
    nuclear_dipole = 2.3
    model_classes = get_fmodels(sr88_mqdt_fmodel_data, species)
