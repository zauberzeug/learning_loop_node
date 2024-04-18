from dataclasses import asdict
from typing import Dict, Optional

from dacite import from_dict
from fastapi.encoders import jsonable_encoder
from socketio import AsyncClient

from ..data_classes import AnnotationNodeStatus, Context, NodeState, UserInput
from ..data_classes.socket_response import SocketResponse
from ..data_exchanger import DataExchanger
from ..helpers.misc import create_image_folder, create_project_folder
from ..node import Node
from .annotator_logic import AnnotatorLogic

# TODO: The use case 'segmentation' is hardcoded here. This should be more flexible.


class AnnotatorNode(Node):

    def __init__(self, name: str, annotator_logic: AnnotatorLogic, uuid: Optional[str] = None):
        super().__init__(name, uuid, 'annotation_node')
        self.tool = annotator_logic
        self.histories: Dict = {}
        annotator_logic.init(self)
        self.status_sent = False

    def register_sio_events(self, sio_client: AsyncClient):

        @sio_client.event
        async def handle_user_input(user_input_dict):
            return await self._handle_user_input(user_input_dict)

        @sio_client.event
        async def user_logout(sid):
            self.reset_history(sid)
            return self.tool.logout_user(sid)

    async def _handle_user_input(self, user_input_dict: Dict) -> str:
        user_input = from_dict(data_class=UserInput, data=user_input_dict)

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
            await self.sio_client.call('update_segmentation_annotation', (user_input.data.context.organization,
                                                                          user_input.data.context.project,
                                                                          jsonable_encoder(asdict(tool_result.annotation))), timeout=30)
        return tool_result.svg

    def reset_history(self, frontend_id: str) -> None:
        """Reset the history for a given frontend_id."""
        if frontend_id in self.histories:
            del self.histories[frontend_id]

    def get_history(self, frontend_id: str) -> Dict:
        """Get the history for a given frontend_id. If no history exists, create a new one."""
        return self.histories.setdefault(frontend_id, self.tool.create_empty_history())

    async def send_status(self):
        if self.status_sent:
            return

        status = AnnotationNodeStatus(
            id=self.uuid,
            name=self.name,
            state=NodeState.Online,
            capabilities=['segmentation']
        )

        self.log.info(f'Sending status {status}')
        try:
            result = await self.sio_client.call('update_annotation_node', jsonable_encoder(asdict(status)), timeout=10)
        except Exception as e:
            self.log.error(f'Error for updating: {str(e)}')
            return

        assert isinstance(result, Dict)
        response = from_dict(data_class=SocketResponse, data=result)

        if not response.success:
            self.log.error(f'Error for updating: Response from loop was : {asdict(response)}')
        else:
            self.status_sent = True

    async def download_image(self, context: Context, uuid: str):
        project_folder = create_project_folder(context)
        images_folder = create_image_folder(project_folder)

        downloader = DataExchanger(context=context, loop_communicator=self.loop_communicator)
        await downloader.download_images([uuid], images_folder)

    async def on_startup(self):
        pass

    async def on_shutdown(self):
        pass

    async def on_repeat(self):
        await self.send_status()
