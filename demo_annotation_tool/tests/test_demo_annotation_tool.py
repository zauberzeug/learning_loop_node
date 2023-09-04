from fastapi.encoders import jsonable_encoder

from demo_annotation_tool.annotation_tool import SegmentationTool
from learning_loop_node.annotation.annotator_logic import UserInput
from learning_loop_node.annotation.annotator_node import AnnotatorNode
from learning_loop_node.data_classes import AnnotationData, EventType, Point
from learning_loop_node.data_classes.general import (Category, CategoryType,
                                                     Context)


class MockAsyncClient():
    def __init__(self):
        self.history = []

    async def call(self, *args, **kwargs):
        self.history.append((args, kwargs))
        return True


def default_user_input() -> UserInput:
    context = Context(organization='zauberzeug', project='pytest')
    category = Category(identifier='some_id', name='category_1', description='',
                        hotkey='', color='', ctype=CategoryType.Segmentation, point_size=None)
    annotation_data = AnnotationData(
        coordinate=Point(x=0, y=0),
        event_type=EventType.LeftMouseDown,
        context=context,
        image_uuid='285a92db-bc64-240d-50c2-3212d3973566',
        category=category,
        is_shift_key_pressed=None
    )
    return UserInput(frontend_id='some_id', data=annotation_data)


# pylint: disable=unused-argument
async def test_start_creating(setup_test_project):

    mock_async_client = MockAsyncClient()
    node = AnnotatorNode(name='', uuid='', annotator_logic=SegmentationTool())
    node._sio_client = mock_async_client  # TODO why  is sio client set to this and not the async client?

    user_input = default_user_input()
    result = await node._handle_user_input(jsonable_encoder(user_input))
    assert result == '<rect x="0" y="0" width="0" height="0" stroke="blue" fill="transparent">'
