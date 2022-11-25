import sys
if sys.version_info.major >= 3 and sys.version_info.minor >= 7:  # most code needs at least python 3.7
    from .trainer.trainer_node import TrainerNode
    from .converter.converter_node import ConverterNode

from .context import Context
from .detector.detector_node import DetectorNode
from .detector.detector import Detector
from .model_information import ModelInformation
from .data_classes.category import CategoryType
from .globals import GLOBALS
from .loop import loop

from . import log_conf

log_conf.init()
