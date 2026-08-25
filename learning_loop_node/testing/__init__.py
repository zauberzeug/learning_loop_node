"""Helpers for testing a node, shipped with the library.

Unlike `learning_loop_node.tests` — the library's own suite, which is excluded from the wheel —
everything here is importable by a node repository.

The pytest fixtures live in `learning_loop_node.testing.fixtures` and are not re-exported here,
so that this package stays importable without pytest. Opt into them from a `conftest.py`:

    pytest_plugins = ['learning_loop_node.testing.fixtures']
"""

from .detections import get_dummy_detections, get_dummy_metadata
from .detector import TestingDetectorFactory, TestingDetectorLogic
from .helpers import (
    assert_not_production_loop,
    condition,
    get_files_in_folder,
    get_latest_model_id,
    unzip,
    update_attributes,
)
from .trainer import TestingTrainerLogic, assert_training_state, create_active_training_file

__all__ = [
    'TestingDetectorFactory',
    'TestingDetectorLogic',
    'TestingTrainerLogic',
    'assert_not_production_loop',
    'assert_training_state',
    'condition',
    'create_active_training_file',
    'get_dummy_detections',
    'get_dummy_metadata',
    'get_files_in_folder',
    'get_latest_model_id',
    'unzip',
    'update_attributes',
]
