from __future__ import annotations

from typing import ClassVar

from rydstate.angular.core_ket import CoreKet
from rydstate.angular.utils import Unknown
from rydstate.species.fmodel import get_fmodels
from rydstate.species.mqdt import MQDT
from rydstate.species.ytterbium import (
    yb171_mqdt_fmodel_data,
    yb173_mqdt_fmodel_data,
    yb174_mqdt_fmodel_data,
)


class MQDTYtterbium171(MQDT):
    species = "Yb171"
    is_default = True

    ionization_threshold_dict: ClassVar = {
        CoreKet(0.5, 0.5, 0, 0.5, 0): (50442.795744, None, "1/cm"),
        CoreKet(0.5, 0.5, 0, 0.5, 1): (50443.217463, None, "1/cm"),
        CoreKet(0.5, 0.5, 1, 0.5, Unknown, label=Unknown): (77504.98, None, "1/cm"),
        CoreKet(0.5, 0.5, 1, Unknown, Unknown, label=Unknown): (79725.35, None, "1/cm"),
        CoreKet(0.5, 0.5, 1, 1.5, Unknown, label=Unknown): (80835.39, None, "1/cm"),
        CoreKet(0.5, 0.5, Unknown, Unknown, Unknown, label="4f13 5d 6s"): (83967.7, None, "1/cm"),
    }
    core_ground_state = CoreKet(0.5, 0.5, 0, 0.5, 1)
    nuclear_dipole = 0.49367
    model_classes = get_fmodels(yb171_mqdt_fmodel_data, species)


class MQDTYtterbium173(MQDT):
    species = "Yb173"
    is_default = True

    ionization_threshold_dict: ClassVar = {
        CoreKet(2.5, 0.5, 0, 0.5, 2): (50443.291203, None, "1/cm"),
        CoreKet(2.5, 0.5, 0, 0.5, 3): (50442.941262, None, "1/cm"),
        CoreKet(2.5, 0.5, 1, 0.5, Unknown, label=Unknown): (77504.98, None, "1/cm"),
        CoreKet(2.5, 0.5, 1, 1.5, Unknown, label=Unknown): (80835.39, None, "1/cm"),
        CoreKet(2.5, 0.5, Unknown, Unknown, Unknown, label="4f13 5d 6s"): (83967.7, None, "1/cm"),
    }
    core_ground_state = CoreKet(2.5, 0.5, 0, 0.5, 2)
    nuclear_dipole = -0.68
    model_classes = get_fmodels(yb173_mqdt_fmodel_data, species)


class MQDTYtterbium174(MQDT):
    species = "Yb174"
    is_default = True

    ionization_threshold_dict: ClassVar = {
        CoreKet(0, 0.5, 0, 0.5): (50443.070393, None, "1/cm"),
        CoreKet(0, 0.5, 1, 0.5): (77504.98, None, "1/cm"),
        CoreKet(0, 0.5, 1, 1.5): (80835.39, None, "1/cm"),
        CoreKet(0, 0.5, 1, Unknown, label=Unknown): (79725.35, None, "1/cm"),
        CoreKet(0, 0.5, Unknown, Unknown, label="4f13 5d 6s"): (83967.7, None, "1/cm"),
    }
    core_ground_state = CoreKet(0, 0.5, 0, 0.5)
    nuclear_dipole = 2.1
    model_classes = get_fmodels(yb174_mqdt_fmodel_data, species)
