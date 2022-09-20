
from learning_loop_node.context import Context
import pytest
from fastapi.encoders import jsonable_encoder
from icecream import ic
from demo_segmentation_tool import DemoSegmentationTool
from learning_loop_node.annotation_node.data_classes import AnnotationData, EventType, Point, UserInput, EventType
from learning_loop_node.annotation_node.annotation_node import AnnotationNode
from learning_loop_node.data_classes.category import Category, CategoryType
from learning_loop_node.tests.mock_async_client import MockAsyncClient
from learning_loop_node.rest.downloader import DataDownloader


def default_user_input() -> UserInput:
    annotation_data = AnnotationData(
        coordinate=Point(x=0, y=0),
        event_type=EventType.MouseDown,
        context=Context(organization='zauberzeug', project='pytest'),
        image_uuid='285a92db-bc64-240d-50c2-3212d3973566',
        category=Category(id='some_id', name='category_1', description='',
                          hotkey='', color='', type=CategoryType.Segmentation)
    )
    return UserInput(frontend_id='some_id', data=annotation_data)


async def test_start_creating(create_project):

    mock_async_client = MockAsyncClient()
    node = AnnotationNode(name='', uuid='', tool=DemoSegmentationTool())
    node.sio_client = mock_async_client

    input = default_user_input()
    result = await node.handle_user_input(jsonable_encoder(input))
    assert result == '<rect x="0" y="0" width="0" height="0" stroke="blue" fill="transparent">'
