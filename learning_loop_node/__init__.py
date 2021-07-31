import sys
if sys.version_info.major >= 3 and sys.version_info.minor >= 7: # most code needs at least python 3.7
    from .trainer.capability import Capability
    from .trainer.trainer_node import TrainerNode
    from .converter.converter_node import ConverterNode

from .detector.detector_node import DetectorNode
from .loop import loop
