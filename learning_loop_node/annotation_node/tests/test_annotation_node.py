import pytest
import os
from fastapi.encoders import jsonable_encoder
from learning_loop_node.context import Context
from learning_loop_node.annotation_node.data_classes import AnnotationData, EventType, Point, UserInput, EventType
from learning_loop_node.annotation_node.annotation_node import AnnotationNode
from learning_loop_node.annotation_node.annotation_tool import EmptyAnnotationTool
from learning_loop_node.data_classes.category import Category, CategoryType
from icecream import ic


def default_user_input() -> UserInput:
    annotation_data = AnnotationData(
        coordinate=Point(x=0, y=0),
        event_type=EventType.LeftMouseDown,
        context=Context(organization='zauberzeug', project='pytest'),
        image_uuid='285a92db-bc64-240d-50c2-3212d3973566',
        category=Category(id='some_id', name='category_1', description='',
                          hotkey='', color='', type=CategoryType.Segmentation)
    )
    return UserInput(frontend_id='some_id', data=annotation_data)


async def test_image_download(create_project):
    image_path = '/tmp/learning_loop_lib_data/zauberzeug/pytest/images/285a92db-bc64-240d-50c2-3212d3973566.jpg'

    assert os.path.exists(image_path) == False

    node = AnnotationNode(name="", uuid="", tool=EmptyAnnotationTool())
    input = default_user_input()
    _ = await node.handle_user_input(jsonable_encoder(input))

    assert os.path.exists(image_path) == True
