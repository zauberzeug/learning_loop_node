import logging
import os
import sys

from .converter.converter_node import ConverterNode
# from . import log_conf
from .detector.detector_logic import DetectorLogic
from .detector.detector_node import DetectorNode
from .globals import GLOBALS
from .trainer.trainer_node import TrainerNode

logging.info('>>>>>>>>>>>>>>>>>> LOOP INITIALIZED <<<<<<<<<<<<<<<<<<<<<<<')
