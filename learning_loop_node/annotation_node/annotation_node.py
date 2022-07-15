
from fastapi.encoders import jsonable_encoder
from learning_loop_node.node import Node
from learning_loop_node.status import State, AnnotationNodeStatus
from learning_loop_node.context import Context
from learning_loop_node.annotation_node.annotation_tool import AnnotationTool
from learning_loop_node.annotation_node.data_classes import EventType, UserInput
from learning_loop_node.rest.downloader import DataDownloader
from learning_loop_node.rest.downloader import node_helper
import logging
from icecream import ic
from ..socket_response import SocketResponse


class AnnotationNode(Node):
    tool: AnnotationTool
    histories: dict = {}

    def __init__(self, name: str, uuid: str, tool: AnnotationTool):
        super().__init__(name, uuid)
        self.tool = tool

        @self.sio_client.on('handle_user_input')
        async def on_handle_user_input(user_input):
            return await self.handle_user_input(user_input)

        @self.sio_client.on('user_logout')
        async def on_logout_user(sid):
            self.reset_history(sid)
            return self.tool.logout_user(sid)

    async def handle_user_input(self, user_input) -> str:
        input = UserInput.parse_obj(user_input)

        if input.data.key_up == 'Escape':
            self.reset_history(input.frontend_id)
            return ''

        await self.download_image(input.data.context, input.data.image_uuid)
        history = self.get_history(input.frontend_id)
        try:
            tool_result = await self.tool.handle_user_input(input, history)
        except:
            self.reset_history(input.frontend_id)
            raise

        if tool_result.annotation:
            result = await self.sio_client.call('update_segmentation_annotation', (input.data.context.organization,
                                                                                   input.data.context.project, jsonable_encoder(tool_result.annotation)), timeout=2)

        return tool_result.svg

    def reset_history(self, frontend_id: str) -> None:
        try:
            del self.histories[frontend_id]
        except:
            pass

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
        response = SocketResponse.from_dict(result)

        if not response.success:
            logging.error(f'Error for updating: Response from loop was : {response.__dict__}')

    async def download_image(self, context: Context, uuid: str):
        project_folder = Node.create_project_folder(context)
        images_folder = node_helper.create_image_folder(project_folder)

        downloader = DataDownloader(context=context)
        await downloader.download_images([uuid], images_folder)
