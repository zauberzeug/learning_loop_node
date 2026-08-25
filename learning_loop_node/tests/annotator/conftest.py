import asyncio
import logging
import sys

import pytest

from ...loop_communication import LoopCommunicator
from ...testing import assert_not_production_loop
from ...testing.fixtures import clear_loggers, data_folder  # noqa: F401  pylint: disable=unused-import


@pytest.fixture()
async def setup_test_project():  # pylint: disable=redefined-outer-name
    assert_not_production_loop()
    loop_communicator = LoopCommunicator()
    try:
        await loop_communicator.delete("/zauberzeug/projects/pytest_nodelib_annotator?keep_images=true", timeout=10)
    except Exception:
        logging.exception("Failed to delete project pytest_nodelib_annotator")
        sys.exit(1)
    await asyncio.sleep(1)
    project_conf = {
        'project_name': 'pytest_nodelib_annotator', 'inbox': 0, 'annotate': 0, 'review': 0, 'complete': 3, 'image_style': 'beautiful',
        'box_categories': 2, 'point_categories': 2, 'segmentation_categories': 2, 'thumbs': False, 'tags': 0,
        'trainings': 1, 'box_detections': 3, 'box_annotations': 0}
    assert (await loop_communicator.post("/zauberzeug/projects/generator", json=project_conf)).status_code == 200
    yield
    await loop_communicator.delete("/zauberzeug/projects/pytest_nodelib_annotator?keep_images=true", timeout=10)
    await loop_communicator.shutdown()
