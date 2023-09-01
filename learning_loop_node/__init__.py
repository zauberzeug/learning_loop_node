import sys

if sys.version_info.major >= 3 and sys.version_info.minor >= 8:  # most code needs at least python 3.8
    from .trainer.trainer_node import TrainerNode
    from .converter.converter_node import ConverterNode

from . import log_conf
from .data_classes import CategoryType, Context, ModelInformation
from .detector.detector import Detector
from .detector.detector_node import DetectorNode
from .globals import GLOBALS
from .loop_communication import LoopCommunication
