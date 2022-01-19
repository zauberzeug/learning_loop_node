import asyncio
from learning_loop_node.context import Context
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
import logging
from datetime import datetime
from ..socket_response import SocketResponse


class TrainerNode(Node):
    trainer: Trainer
    latest_known_model_id: Union[str, None]
    skip_check_state: bool = False

    def __init__(self, name: str, trainer: Trainer, uuid: str = None):
        super().__init__(name, uuid)
        self.trainer = trainer
        self.latest_known_model_id = None

        @self.sio_client.on('begin_training')
        async def on_begin_training(organization, project, source_model):
            loop = asyncio.get_event_loop()
            loop.create_task(self.begin_training(Context(organization=organization, project=project), source_model))
            return True

        @self.sio_client.on('stop_training')
        async def stop():
            return await self.stop_training()

        @self.sio_client.on('save')
        def on_save(organization, project, model):
            loop = asyncio.get_event_loop()
            loop.create_task(self.save_model(Context(organization=organization, project=project), model['id']))
            return True

        @self.on_event("startup")
        @repeat_every(seconds=5, raise_exceptions=True, wait_first=False)
        async def check_state():
            if self.skip_check_state:
                return
            try:
                await self.check_state()
            except:
                logging.exception('could not check state')

        @self.on_event("shutdown")
        async def shutdown():
            logging.info('shutdown detected, stopping training')
            await self.stop_training()

    async def begin_training(self, context: Context, source_model: dict):
        self.status.reset_error('start_training')
        await self.update_state(State.Preparing)
        try:
            await self.trainer.begin_training(context, source_model)
        except Exception as e:
            self.status.set_error('start_training', f'Could not start training: {str(e)})')

            logging.exception(self.status._errors)
            self.trainer.stop_training()
            await self.update_state(State.Idle)
            return
        self.latest_known_model_id = source_model['id']
        await self.update_state(State.Running)

    async def stop_training(self) -> Union[bool, str]:
        self.status.reset_error('stop_training')
        try:
            result = self.trainer.stop_training()
            self.trainer.training = None
            self.latest_known_model_id = None

            await self.update_state(State.Idle)
            if not result:
                raise Exception('No Training is running')
            self.status.reset_all_errors()
            await self.send_status()

        except Exception as e:
            self.status.set_error('stop_training', f'Could not stop training: {str(e)})')
            await self.send_status()
            return False
        return True

    async def save_model(self, context: Context, model_id: str):
        self.status.reset_error('save_model')
        try:
            await self.trainer.save_model(context, model_id)
        except Exception as e:
            traceback.print_exc()
            self.status.set_error('save_model', f'Could not save model: {str(e)}')

        await self.send_status()

    async def check_state(self):
        logging.debug(f'{self.status.state}')
        self.status.reset_error('training_error')
        error = self.trainer.get_error()

        if error is not None:
            logging.error(error + '\n\n' + self.trainer.get_log()[-1000:])
            self.status.set_error('training_error', error)
            await self.send_status()
            return

        if self.status.state != State.Running:
            return

        if not self.trainer.executor.is_process_running():
            self.status.set_error('training_error', 'Training crashed.')
            logging.info(self.trainer.get_log()[-1000:])
            await self.send_status()
            return

        await self.try_get_new_model()

    async def try_get_new_model(self) -> None:
        self.status.reset_error('get_new_model')

        try:
            current_training = self.trainer.training
            if self.status.state == State.Running and current_training:
                model = self.trainer.get_new_model()
                logging.debug(f'new model {model}')
                if model:
                    new_model = Model(
                        id=str(uuid4()),
                        confusion_matrix=model.confusion_matrix,
                        parent_id=self.latest_known_model_id,
                        train_image_count=current_training.data.train_image_count(),
                        test_image_count=current_training.data.test_image_count(),
                        trainer_id=self.uuid,
                    )

                    result = await self.sio_client.call('update_model', (current_training.context.organization, current_training.context.project, jsonable_encoder(new_model)))
                    response = SocketResponse.from_dict(result)

                    if not response.success:
                        error_msg = f'Error for update_model: Response from loop was : {response.__dict__}'
                        logging.error(error_msg)
                        raise Exception(error_msg)

                    logging.info(f'successfully uploaded model {jsonable_encoder(new_model)}')
                    self.trainer.on_model_published(model, new_model.id)
                    self.latest_known_model_id = new_model.id

        except Exception as e:
            msg = f'Could not get new model: {str(e)}'
            logging.exception(msg)
            self.status.set_error('get_new_model', msg)

        await self.send_status()

    async def send_status(self):

        status = TrainingStatus(
            id=self.uuid,
            name=self.name,
            state=self.status.state,
            uptime=int((datetime.now() - self.startup_time).total_seconds()),
            latest_produced_model_id=self.latest_known_model_id,
            current_error='\n'.join(self.status._errors.values())
        )

        if self.trainer and self.trainer.training:
            status.train_image_count = self.trainer.training.data.train_image_count()
            status.test_image_count = self.trainer.training.data.test_image_count()
            status.skipped_image_count = self.trainer.training.data.skipped_image_count

        logging.info(f'sending status {status}')
        result = await self.sio_client.call('update_trainer', jsonable_encoder(status), timeout=1)
        response = SocketResponse.from_dict(result)

        if not response.success:
            logging.error(f'Error for updating: Response from loop was : {response.__dict__}')

    def get_state(self):
        if self.trainer.executor is not None and self.trainer.executor.is_process_running():
            return State.Running
        return State.Idle
