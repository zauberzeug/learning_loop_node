import asyncio
import traceback
from fastapi_utils.tasks import repeat_every
from fastapi.encoders import jsonable_encoder
from typing import Union
from uuid import uuid4
from icecream import ic
from .model import Model
from .trainer import Trainer
from learning_loop_node.status import TrainingStatus
from learning_loop_node.node import Node, State


class TrainerNode(Node):
    trainer: Trainer
    latest_known_model_id: Union[str, None]
    skip_check_state: bool = False

    def __init__(self, name: str, uuid: str, trainer: Trainer):
        super().__init__(name, uuid)
        self.trainer = trainer
        self.latest_known_model_id = None

        @self.sio.on('begin_training')
        async def on_begin_training(organization, project, source_model):
            loop = asyncio.get_event_loop()
            loop.create_task(self.begin_training(organization, project, source_model))
            return True

        @self.sio.on('stop_training')
        async def stop():
            return await self.stop_training()

        @self.sio.on('save')
        def on_save(organization, project, model):
            loop = asyncio.get_event_loop()
            loop.create_task(self.save_model(organization, project, model['id']))
            return True

        @self.on_event("startup")
        @repeat_every(seconds=5, raise_exceptions=True, wait_first=False)
        async def check_state():
            if not self.skip_check_state:
                await self.check_state()

    async def begin_training(self, organization: str, project: str, source_model: dict):
        self.status.latest_error = None
        await self.update_state(State.Preparing)
        try:
            await self.trainer.begin_training(self.url, self.headers, organization, project, source_model)
        except Exception as e:
            traceback.print_exc()
            self.status.latest_error = f'Could not start training: {str(e)})'
            self.trainer.stop_training()
            await self.update_state(State.Idle)
            return
        self.latest_known_model_id = source_model['id']
        await self.update_state(State.Running)

    async def stop_training(self) -> Union[bool, str]:
        try:
            self.trainer.stop_training()
            self.trainer.training = None
            await self.update_state(State.Idle)
        except Exception as e:
            self.status.latest_error = f'Could not stop training: {str(e)})'
            traceback.print_exc()
            await self.send_status()

            return False
        self.latest_known_model_id = None
        await self.send_status()
        return True

    async def save_model(self, organization, project, model_id):
        try:
            await self.trainer.save_model(self.url, self.headers, organization, project, model_id)
        except Exception as e:
            traceback.print_exc()
            await self.update_error_msg(f'Could not save model: {str(e)}')

    async def check_state(self):
        ic(f'checking state: {self.trainer.training != None}, state: {self.status.state}')
        try:
            if self.status.state == State.Running and not self.trainer.is_training_alive():
                raise Exception()
        except:
            await self.update_error_msg(f'Training crashed.')

        await self.try_get_new_model()

    async def try_get_new_model(self) -> None:
        try:
            current_training = self.trainer.training
            if self.status.state == State.Running and current_training:
                model = self.trainer.get_new_model()
                if model:
                    new_model = Model(
                        id=str(uuid4()),
                        confusion_matrix=model.confusion_matrix,
                        parent_id=self.latest_known_model_id,
                        train_image_count=self.trainer.training.data.train_image_count(),
                        test_image_count=self.trainer.training.data.test_image_count(),
                        trainer_id=self.uuid,
                    )

                    result = await self.sio.call('update_model', (current_training.context.organization, current_training.context.project, jsonable_encoder(new_model)))
                    if result != True:
                        await self.update_error_msg(f'Could not update_model: {result}')
                        return

                    ic(f'successfully uploaded model {jsonable_encoder(new_model)}')
                    self.trainer.on_model_published(model, new_model.id)
                    self.latest_known_model_id = new_model.id
                    await self.send_status()
        except Exception as e:
            await self.update_error_msg(f'Could not get new model: {str(e)}')

    async def send_status(self):
        status = TrainingStatus(
            id=self.uuid,
            name=self.name,
            state=self.status.state,
            uptime=self.status.uptime,
            latest_produced_model_id=self.latest_known_model_id,
            latest_error=self.status.latest_error
        )

        print('sending status', status, flush=True)
        result = await self.sio.call('update_trainer', jsonable_encoder(status), timeout=1)
        if not result == True:
            raise Exception(result)
        print('status send', flush=True)

    async def update_error_msg(self, msg: str) -> None:
        self.status.latest_error = msg
        await self.send_status()
