from learning_loop_node.trainer.downloader_factory import DownloaderFactory
from learning_loop_node.status import TrainingStatus
from learning_loop_node.trainer.trainer import Trainer
from learning_loop_node.context import Context
from learning_loop_node.node import Node
import asyncio
from status import State
from fastapi.encoders import jsonable_encoder
from typing import Union
from fastapi_utils.tasks import repeat_every
import traceback


class TrainerNode(Node):
    trainer: Trainer

    def __init__(self, name: str, uuid: str, trainer: Trainer):
        super().__init__(name, uuid)
        self.trainer = trainer

        @self.sio.on('begin_training')
        async def on_begin_training(organization, project, source_model):
            loop = asyncio.get_event_loop()
            loop.set_debug(True)
            loop.create_task(self.begin_training(organization, project, source_model))
            return True

        @self.sio.on('stop_training')
        async def stop():
            return await self.stop_training()

        @self.sio.on('save')
        def on_save(organization, project, model):
            loop = asyncio.get_event_loop()
            loop.set_debug(True)
            loop.create_task(self.save_model(organization, project, model['id']))
            return True

        @self.on_event("startup")
        @repeat_every(seconds=5, raise_exceptions=True, wait_first=False)
        def check_state():
            self.check_state()

    async def begin_training(self, organization: str, project: str, source_model: dict):
        await self.update_state(State.Preparing)
        try:
            context = Context(organization=organization, project=project)
            downloader = DownloaderFactory.create(self.url, self.headers, context, self.trainer.capability)
            await self.trainer.begin_training(context, source_model, downloader)
        except Exception as e:
            traceback.print_exc()
            self.trainer.stop_training()
            await self.update_state(State.Idle)
            return

        await self.update_state(State.Running)

    async def stop_training(self) -> Union[bool, str]:
        try:
            self.trainer.stop_training()
            await self.update_state(State.Idle)
        except Exception as e:
            traceback.print_exc()
            return str(e)

        return True

    async def save_model(self, organization, project, model_id):
        try:
            await self.trainer.save_model(self.url, organization, project, model_id)
        except Exception as e:
            traceback.print_exc()

    async def check_state(self):
        current_training = self.trainer.training
        if current_training:
            new_model = self.trainer.get_new_model()
            if new_model:
                result = await self.sio.call('update_model', (current_training.context.organization, current_training.context.project, jsonable_encoder(new_model)))
                if result != True:
                    msg = f'could not update model: {str(result)}'
                    print(msg)
                    return msg  # for backdoor
                current_training.last_produced_model = new_model
                await self.send_status()

    async def send_status(self):
        status = TrainingStatus(
            id=self.uuid,
            name=self.name,
            state=self.status.state,
            uptime=self.status.uptime

        )
        if self.trainer.training and self.trainer.training.last_produced_model:
            status.latest_produced_model_id = self.trainer.training.last_produced_model.id

        print('sending status', status, flush=True)
        result = await self.sio.call('update_trainer', jsonable_encoder(status), timeout=1)
        if not result == True:
            raise Exception(result)
        print('status send', flush=True)
