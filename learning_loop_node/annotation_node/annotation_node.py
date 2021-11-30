
from learning_loop_node.node import Node
import logging
from fastapi.encoders import jsonable_encoder
from learning_loop_node.status import State, AnnotationNodeStatus
from icecream import ic
from learning_loop_node.trainer.trainer import Trainer
from learning_loop_node.context import Context
from learning_loop_node.trainer.downloader_factory import DownloaderFactory
from learning_loop_node.annotation_node.annotation_tool import AnnotationTool
from learning_loop_node.annotation_node.data_classes import EventType, UserInput


class AnnotationNode(Node):
    tool: AnnotationTool
    histories: dict = {}

    def __init__(self, name: str, uuid: str, tool: AnnotationTool):
        super().__init__(name, uuid)
        self.tool = tool

        @self.sio_client.on('handle_user_input')
        async def on_handle_user_input(organization, project, user_input):
            return await self.handle_user_input(organization, project, user_input)

    async def handle_user_input(self, organization, project, user_input) -> str:
        raise Exception('Test Drone build')
        ic(user_input)

        input = UserInput.parse_obj(user_input)
        await self.download_image(input.data.context, input.data.image_uuid)
        history = self.get_history(input.frontend_id)
        tool_result = await self.tool.handle_user_input(input, history)
        ic(tool_result)
        if tool_result.annotation:
            result = await self.sio_client.call('update_segmentation_annotation', (organization,
                                                                                   project, jsonable_encoder(tool_result.annotation)), timeout=2)

        return tool_result.svg

    def get_history(self, frontend_id: str) -> dict:
        if not frontend_id in self.histories.keys():
            self.histories[frontend_id] = self.tool.create_empty_history()

        return self.histories[frontend_id]

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
