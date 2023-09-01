
import logging

from fastapi.encoders import jsonable_encoder

from learning_loop_node.annotation.annotator_model import (AnnotatatorModel,
                                                           UserInput)
from learning_loop_node.data_classes.general import Context
from learning_loop_node.node import Node
from learning_loop_node.rest.downloader import DataDownloader, node_helper
from learning_loop_node.socket_response import SocketResponse
from learning_loop_node.status import AnnotationNodeStatus, State


class AnnotatorNode(Node):
    tool: AnnotatatorModel

    def __init__(self, name: str, uuid: str, tool: AnnotatatorModel):
        super().__init__(name, uuid)
        self.tool = tool
        self.histories: dict = {}

    async def create_sio_client(self):
        await super().create_sio_client()

        assert self.sio_client is not None
        assert self.sio_client.on is not None

        dec_input = self.sio_client.on('handle_user_input')
        assert dec_input is not None

        @dec_input
        async def on_handle_user_input(user_input):
            return await self.handle_user_input(user_input)

        dec_logout = self.sio_client.on('user_logout')
        assert dec_logout is not None

        @dec_logout
        async def on_logout_user(sid):
            self.reset_history(sid)
            return self.tool.logout_user(sid)

    async def handle_user_input(self, user_input) -> str:
        user_input = UserInput.parse_obj(user_input)

        if user_input.data.key_up == 'Escape':
            self.reset_history(user_input.frontend_id)
            return ''

        await self.download_image(user_input.data.context, user_input.data.image_uuid)
        history = self.get_history(user_input.frontend_id)
        try:
            tool_result = await self.tool.handle_user_input(user_input, history)
        except:
            self.reset_history(user_input.frontend_id)
            raise

        if tool_result.annotation:
            if self.sio_client is None:
                raise Exception('No socket client')
            await self.sio_client.call('update_segmentation_annotation', (user_input.data.context.organization,
                                                                          user_input.data.context.project, jsonable_encoder(tool_result.annotation)), timeout=2)

        return tool_result.svg

    def reset_history(self, frontend_id: str) -> None:
        try:
            del self.histories[frontend_id]
        except Exception:
            pass

    def get_history(self, frontend_id: str) -> dict:
        if not frontend_id in self.histories:
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
        if self.sio_client is None:
            raise Exception('No socket client')
        result = await self.sio_client.call('update_annotation_node', jsonable_encoder(status), timeout=2)
        assert isinstance(result, dict)
        response = SocketResponse.from_dict(result)

        if not response.success:
            logging.error(f'Error for updating: Response from loop was : {response.__dict__}')

    async def download_image(self, context: Context, uuid: str):
        project_folder = Node.create_project_folder(context)
        images_folder = node_helper.create_image_folder(project_folder)

        downloader = DataDownloader(context=context)
        await downloader.download_images([uuid], images_folder)

    async def get_state(self):
        return State.Online

    def get_node_type(self):
        return 'annotation_node'
