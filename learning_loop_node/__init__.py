import logging
import os
import sys

# from . import log_conf
from .detector.detector_logic import DetectorLogic
from .detector.detector_node import DetectorNode
from .globals import GLOBALS

if sys.version_info.major >= 3 and sys.version_info.minor >= 8:  # most code needs at least python 3.8
    from .converter.converter_node import ConverterNode
    from .trainer.trainer_node import TrainerNode


logging.info('>>>>>>>>>>>>>>>>>> LOOP INITIALIZED <<<<<<<<<<<<<<<<<<<<<<<')
