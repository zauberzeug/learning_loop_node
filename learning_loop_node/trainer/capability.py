from enum import Enum


class Capability(str, Enum):
    Box = "box"
    Segmentations = "segmentations"
