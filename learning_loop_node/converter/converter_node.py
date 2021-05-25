from learning_loop_node.trainer.capability import Capability
from learning_loop_node.converter.converter import Converter
from learning_loop_node.trainer.downloader_factory import DownloaderFactory
from learning_loop_node.status import TrainingStatus
from learning_loop_node.context import Context
from learning_loop_node.node import Node
import asyncio
from status import State
from fastapi.encoders import jsonable_encoder
from fastapi_utils.tasks import repeat_every
from icecream import ic


class ConverterNode(Node):
    converter: Converter
    skip_check_state: bool = False

    def __init__(self, name: str, uuid: str, converter: Converter):
        super().__init__(name, uuid)
        self.converter = converter

        @self.sio.on('begin_training')
        async def on_convert_model(organization, project, source_model):
            loop = asyncio.get_event_loop()
            loop.create_task(self.convert_model(organization, project, source_model))
            return True

        @self.on_event("startup")
        @repeat_every(seconds=5, raise_exceptions=True, wait_first=False)
        async def check_state():
            if not self.skip_check_state:
                await self.check_state()

    async def convert_model(self, organization: str, project: str, source_model: dict):
        self.status.latest_error = None
        await self.update_state(State.Running)

        context = Context(organization=organization, project=project)

        downloader = DownloaderFactory.create(self.url, self.headers, context,
                                              Capability.Box)  # TODO make Capability optional?
        await self.converter.convert(context, source_model, downloader)
        await self.converter.save_model(self.url, self.headers, organization, project, source_model['id'])
        await self.update_state(State.Idle)

    async def check_state(self):
        ic(f'checking state: {self.status.state}')
        try:
            if self.status.state == State.Running and not self.converter.is_conversion_alive():
                raise Exception()
        except:
            await self.update_error_msg(f'Conversion crashed.')

    async def send_status(self):
        status = TrainingStatus(
            id=self.uuid,
            name=self.name,
            state=self.status.state,
            uptime=self.status.uptime,
            latest_error=self.status.latest_error
        )

        result = await self.sio.call('update_trainer', jsonable_encoder(status), timeout=1)
        if not result == True:
            raise Exception(result)
        print('status send', flush=True)

    async def update_error_msg(self, msg: str) -> None:
        self.status.latest_error = msg
        await self.send_status()
