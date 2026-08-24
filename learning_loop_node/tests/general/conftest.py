import asyncio
import logging
import sys

import pytest

from ...data_classes import Context
from ...data_exchanger import DataExchanger
from ...loop_communication import LoopCommunicator
from ...testing import assert_not_production_loop
from ...testing.fixtures import clear_loggers, data_folder  # noqa: F401  pylint: disable=unused-import


@pytest.fixture(autouse=True, scope='function')
async def create_project_for_module():
    assert_not_production_loop()
    loop_communicator = LoopCommunicator()
    try:
        await loop_communicator.delete("/zauberzeug/projects/pytest_nodelib_general", timeout=10)
    except Exception:
        logging.warning("Failed to delete project pytest_nodelib_general")
        sys.exit(1)

    await asyncio.sleep(1)
    project_configuration = {
        'project_name': 'pytest_nodelib_general', 'inbox': 0, 'annotate': 0, 'review': 0, 'complete': 3, 'image_style': 'beautiful',
        'box_categories': 2, 'point_categories': 2, 'segmentation_categories': 2, 'thumbs': False, 'tags': 0,
        'trainings': 1, 'box_detections': 3, 'box_annotations': 0}
    assert (await loop_communicator.post("/zauberzeug/projects/generator", json=project_configuration)).status_code == 200
    yield
    await loop_communicator.delete("/zauberzeug/projects/pytest_nodelib_general", timeout=10)
    await loop_communicator.shutdown()


@pytest.fixture()
async def data_exchanger():
    loop_communicator = LoopCommunicator()
    context = Context(organization='zauberzeug', project='pytest_nodelib_general')
    dx = DataExchanger(context, loop_communicator)
    yield dx
    await loop_communicator.shutdown()
