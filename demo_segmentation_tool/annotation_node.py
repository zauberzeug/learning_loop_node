from abc import abstractmethod
from threading import Event
from typing import List, Optional
from learning_loop_node.node import Node
import logging
import os
from fastapi.encoders import jsonable_encoder
from learning_loop_node.status import State, AnnotationNodeStatus
from icecream import ic
from learning_loop_node.trainer.trainer import Trainer
from pydantic import BaseModel
import logging
from enum import Enum
from learning_loop_node.context import Context
from learning_loop_node.model_information import Category
from learning_loop_node.trainer.downloader_factory import DownloaderFactory


class Point(BaseModel):
    x: int
    y: int


class Shape(BaseModel):
    points: List[Point]


class SegmentationAnnotation(BaseModel):
    id: str
    shape: Shape
    image_id: str
    category_id: str


class EventType(str, Enum):
    MouseDown = 'mouse_down',
    MouseMove = 'mouse_move',
    MouseUp = 'mouse_up',

    # TODO Introduce Keyboard Event?
    Enter_Pressed = 'enter_pressed'
    ESC_Pressed = 'esc_pressed'


class AnnotationData(BaseModel):
    coordinate: Point
    event_type: EventType
    context: Context
    image_uuid: str
    category: Category
    # keyboard_modifiers: Optional[List[str]]
    # new_annotation_uuid: Optional[str]
    # edit_annotation_uuid: Optional[str]


class UserInput(BaseModel):

    data: AnnotationData


class ToolOutput(BaseModel):
    svg: str
    annotation: Optional[SegmentationAnnotation]


class AnnotationTool(BaseModel):

    @abstractmethod
    async def handle_user_input(self, user_input: UserInput) -> ToolOutput:
        pass


class EmptyAnnotationTool():
    async def handle_user_input(self, user_input: UserInput) -> ToolOutput:
        return ToolOutput(svg="", shape={})


class AnnotationNode(Node):
    tool: AnnotationTool

    def __init__(self, name: str, uuid: str, tool: AnnotationTool):
        super().__init__(name, uuid)
        self.tool = tool

        @self.sio_client.on('handle_user_input')
        async def on_handle_user_input(organization, project, user_input):
            return await self.handle_user_input(organization, project, user_input)

    async def handle_user_input(self, organization, project, user_input) -> str:
        ic(user_input)
        if user_input['data']['event_type'] != EventType.MouseDown:
            return""
        input = UserInput.parse_obj(user_input)
        await self.download_image(input.data.context, input.data.image_uuid)
        tool_result = await self.tool.handle_user_input(input)
        ic(tool_result)
        result = await self.sio_client.call('update_segmentation_annotation', (organization,
                                                                               project, jsonable_encoder(tool_result.annotation)), timeout=2)
        ic(result)
        # if result != True:
        #     raise Exception(result)
        return tool_result.svg

    async def send_status(self):
        status = AnnotationNodeStatus(
            id=self.uuid,
            name=self.name,
            state=State.Online,
            capabilities=['segmentation']
        )

        logging.info(f'sending status {status}')
        result = await self.sio_client.call('update_annotation_node', jsonable_encoder(status), timeout=2)
        if result != True:
            raise Exception(result)

    async def download_image(self, context: Context, uuid: str):
        project_folder = Node.create_project_folder(context)
        images_folder = Trainer.create_image_folder(project_folder)

        downloader = DownloaderFactory.create(context=context)
        await downloader.download_images([uuid], images_folder)
