from learning_loop_node.context import Context
from learning_loop_node.trainer.downloader_factory import DownloaderFactory
from learning_loop_node.annotation_node.data_classes import AnnotationData, EventType, Point, UserInput, EventType
from learning_loop_node.annotation_node.annotation_node import AnnotationNode
from learning_loop_node.annotation_node.annotation_tool import EmptyAnnotationTool
import pytest
from fastapi.encoders import jsonable_encoder
from icecream import ic
from learning_loop_node.model_information import Category, CategoryType
import os


def default_user_input() -> UserInput:
    annotation_data = AnnotationData(
        coordinate=Point(x=0, y=0),
        event_type=EventType.MouseDown,
        context=Context(organization='zauberzeug', project='pytest'),
        image_uuid='285a92db-bc64-240d-50c2-3212d3973566',
        category = Category(id='some_id', name = 'category_1', description='', hotkey='', color = '', type=CategoryType.Segmentation)
    )
    return UserInput(data=annotation_data)


async def download_basic_data():
    downloader = DownloaderFactory.create(Context(organization='zauberzeug', project='pytest'))
    basic_data = await downloader.download_basic_data()
    image_id = basic_data.image_ids[0]
    ic(image_id)


@pytest.mark.asyncio
async def test_image_download(create_project):
    image_path = '/tmp/learning_loop_lib_data/zauberzeug/pytest/images/285a92db-bc64-240d-50c2-3212d3973566.jpg'

    assert os.path.exists(image_path) == False

    node = AnnotationNode(name="", uuid="", tool=EmptyAnnotationTool())
    input = default_user_input()
    _ = await node.handle_user_input('zauberzeug', 'pytest', jsonable_encoder(input))

    assert os.path.exists(image_path) == True
