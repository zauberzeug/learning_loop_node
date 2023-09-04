from typing import Dict

from fastapi.encoders import jsonable_encoder
from socketio import AsyncClient

from learning_loop_node.annotation.annotator_logic import (AnnotatorLogic,
                                                           UserInput)
from learning_loop_node.data_classes import AnnotationNodeStatus, NodeState
from learning_loop_node.data_classes.general import Context
from learning_loop_node.node import Node
from learning_loop_node.rest_helpers.downloader import (DataDownloader,
                                                        node_helper)
from learning_loop_node.socket_response import SocketResponse

# TODO: The use case 'segmentation' is hardcoded here. This should be more flexible.


class AnnotatorNode(Node):

    def __init__(self, name: str, uuid: str, annotator_logic: AnnotatorLogic):
        super().__init__(name, uuid)
        self.tool = annotator_logic
        self.histories: Dict = {}

    def register_sio_events(self, sio_client: AsyncClient):

        @sio_client.event
        async def handle_user_input(user_input):
            return await self._handle_user_input(user_input)

        @sio_client.event
        async def user_logout(sid):
            self.reset_history(sid)
            return self.tool.logout_user(sid)

    async def _handle_user_input(self, user_input) -> str:
        user_input = UserInput.parse_obj(user_input)

        if user_input.data.key_up == 'Escape':
            self.reset_history(user_input.frontend_id)
            return ''

        await self.download_image(user_input.data.context, user_input.data.image_uuid)

        try:
            tool_result = await self.tool.handle_user_input(user_input, self.get_history(user_input.frontend_id))
        except Exception:
            self.reset_history(user_input.frontend_id)
            raise

        if tool_result.annotation:
            if not self.sio_is_initialized():
                raise Exception('Socket client waas not initialized')
            await self.sio_client.call('update_segmentation_annotation', (user_input.data.context.organization,
                                                                          user_input.data.context.project,
                                                                          jsonable_encoder(tool_result.annotation)), timeout=2)
        return tool_result.svg

    def reset_history(self, frontend_id: str) -> None:
        """Reset the history for a given frontend_id."""
        if frontend_id in self.histories:
            del self.histories[frontend_id]

    def get_history(self, frontend_id: str) -> Dict:
        """Get the history for a given frontend_id. If no history exists, create a new one."""
        return self.histories.setdefault(frontend_id, self.tool.create_empty_history())

    async def send_status(self):
        status = AnnotationNodeStatus(
            id=self.uuid,
            name=self.name,
            state=NodeState.Online,
            capabilities=['segmentation']
        )

        self.log.info(f'Sending status {status}')
        if self._sio_client is None:
            raise Exception('No socket client')
        result = await self._sio_client.call('update_annotation_node', jsonable_encoder(status), timeout=2)
        assert isinstance(result, Dict)
        response = SocketResponse.from_dict(result)

        if not response.success:
            self.log.error(f'Error for updating: Response from loop was : {response.__dict__}')

    async def download_image(self, context: Context, uuid: str):
        project_folder = Node.create_project_folder(context)
        images_folder = node_helper.create_image_folder(project_folder)

        downloader = DataDownloader(context=context)
        await downloader.download_images([uuid], images_folder)

    async def get_state(self):
        return NodeState.Online

    def get_node_type(self):
        return 'annotation_node'

    async def on_startup(self):
        pass

    async def on_shutdown(self):
        pass

    async def on_repeat(self):
        pass
