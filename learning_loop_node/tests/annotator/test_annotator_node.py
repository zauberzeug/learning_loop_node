import os
from dataclasses import asdict
from typing import Dict

import pytest
from fastapi.encoders import jsonable_encoder

from ...annotation.annotator_logic import AnnotatorLogic
from ...annotation.annotator_node import AnnotatorNode
from ...data_classes import (
    AnnotationData,
    Category,
    Context,
    Point,
    ToolOutput,
    UserInput,
)
from ...enums import AnnotationEventType, CategoryType


class MockedAnnotatatorLogic(AnnotatorLogic):
    async def handle_user_input(self, user_input: UserInput, history: Dict) -> ToolOutput:
        return ToolOutput(svg="")

    def create_empty_history(self) -> dict:
        return {}

    def logout_user(self, sid: str) -> bool:
        return True


def default_user_input() -> UserInput:
    annotation_data = AnnotationData(
        coordinate=Point(x=0, y=0),
        event_type=AnnotationEventType.LeftMouseDown,
        context=Context(organization='zauberzeug', project='pytest_nodelib_annotator'),
        image_uuid='10f7d7d2-4076-7006-af37-fa28a6a848ae',
        category=Category(id='some_id', name='category_1', description='',
                          hotkey='', color='', type=CategoryType.Segmentation)
    )
    return UserInput(frontend_id='some_id', data=annotation_data)

# ------------------------------------------------------------ TESTS ------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.usefixtures('setup_test_project')
async def test_image_download():
    # pylint: disable=protected-access

    image_folder = '/tmp/learning_loop_lib_data/zauberzeug/pytest_nodelib_annotator/images'

    assert os.path.exists(image_folder) is False or len(os.listdir(image_folder)) == 0

    node = AnnotatorNode(name="", uuid="", annotator_logic=MockedAnnotatatorLogic())
    user_input = default_user_input()
    await node._ensure_sio_connection()  # This is required as the node is not "started"
    _ = await node._handle_user_input(jsonable_encoder(asdict(user_input)))

    assert os.path.exists(image_folder) is True and len(os.listdir(image_folder)) == 1
