
from learning_loop_node.context import Context
from learning_loop_node.trainer.downloader_factory import DownloaderFactory
import pytest
from fastapi.encoders import jsonable_encoder
from icecream import ic
from demo_segmentation_tool import DemoSegmentationTool
import os
from learning_loop_node.annotation_node.data_classes import AnnotationData, EventType, Point, UserInput, EventType
from learning_loop_node.annotation_node.annotation_node import AnnotationNode
from learning_loop_node.model_information import Category, CategoryType
from learning_loop_node.tests.mock_async_client import MockAsyncClient


def default_user_input() -> UserInput:
    annotation_data = AnnotationData(
        coordinate=Point(x=0, y=0),
        event_type=EventType.MouseDown,
        context=Context(organization='zauberzeug', project='pytest'),
        image_uuid='285a92db-bc64-240d-50c2-3212d3973566',
        category=Category(id='some_id', name='category_1', description='',
                          hotkey='', color='', type=CategoryType.Segmentation)
    )
    return UserInput(data=annotation_data)


async def download_basic_data():
    downloader = DownloaderFactory.create(Context(organization='zauberzeug', project='pytest'))
    basic_data = await downloader.download_basic_data()
    image_id = basic_data.image_ids[0]
    ic(image_id)


@pytest.mark.asyncio
async def test_start_creating(create_project):

    mock_async_client = MockAsyncClient()
    node = AnnotationNode(name='', uuid='', tool=DemoSegmentationTool())
    node.sio_client = mock_async_client

    input = default_user_input()
    result = await node.handle_user_input('zauberzeug', 'pytest', jsonable_encoder(input))
    assert 'update_segmentation_annotation' in str(mock_async_client.history)
    assert result == ''
