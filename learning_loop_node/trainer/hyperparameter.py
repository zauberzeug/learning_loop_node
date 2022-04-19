
from dataclasses import dataclass
from typing import Optional


@dataclass
class Hyperparameter():
    resolution: int
    flip_rl: bool
    flip_ud: bool

    @staticmethod
    def from_dict(value: dict) -> 'Hyperparameter':
        return Hyperparameter(
            resolution=value['resolution'],
            flip_rl=value['flip_rl'],
            flip_ud=value['flip_ud']
        )
