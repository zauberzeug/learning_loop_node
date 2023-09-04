import os

from fastapi.encoders import jsonable_encoder

from learning_loop_node.annotation.annotator_logic import (AnnotatorLogic,
                                                           UserInput)
from learning_loop_node.annotation.annotator_node import AnnotatorNode
from learning_loop_node.data_classes import (AnnotationData, Category,
                                             CategoryType, Context, EventType,
                                             Point, ToolOutput)


class MockedAnnotatatorLogic(AnnotatorLogic):
    async def handle_user_input(self, user_input: UserInput, history: dict) -> ToolOutput:
        return ToolOutput(svg="")

    def create_empty_history(self) -> dict:
        return {}

    def logout_user(self, sid: str) -> bool:
        return True


def default_user_input() -> UserInput:
    annotation_data = AnnotationData(
        coordinate=Point(x=0, y=0),
        event_type=EventType.LeftMouseDown,
        context=Context(organization='zauberzeug', project='pytest'),
        image_uuid='285a92db-bc64-240d-50c2-3212d3973566',
        category=Category(identifier='some_id', name='category_1', description='',
                          hotkey='', color='', ctype=CategoryType.Segmentation)
    )
    return UserInput(frontend_id='some_id', data=annotation_data)

# ------------------------------------------------------------ TESTS ------------------------------------------------------------


async def test_image_download(setup_test_project):  # pylint: disable=unused-argument
    image_path = '/tmp/learning_loop_lib_data/zauberzeug/pytest/images/285a92db-bc64-240d-50c2-3212d3973566.jpg'

    assert os.path.exists(image_path) is False

    node = AnnotatorNode(name="", uuid="", annotator_logic=MockedAnnotatatorLogic())
    user_input = default_user_input()
    _ = await node._handle_user_input(jsonable_encoder(user_input))

    assert os.path.exists(image_path) is True
