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
from .training_status import TrainingStatus
from learning_loop_node.node import Node, State
import logging
from datetime import datetime
from ..socket_response import SocketResponse
from .helper import is_valid_uuid4


class TrainerNode(Node):
    trainer: Trainer
    latest_known_model_id: Union[str, None]
    skip_check_state: bool = False

    def __init__(self, name: str, trainer: Trainer, uuid: str = None):
        super().__init__(name, uuid)
        self.trainer = trainer
        self.latest_known_model_id = None

        @self.sio_client.on('begin_training')
        async def on_begin_training(organization: str, project: str, details: dict):
            loop = asyncio.get_event_loop()
            loop.create_task(self.begin_training(Context(organization=organization, project=project), details))
            return True

        @self.sio_client.on('stop_training')
        async def stop():
            logging.debug(f'### on stop_training received. Current state : {self.status.state}')
            loop = asyncio.get_event_loop()
            loop.create_task(self.stop_training())
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
            try:
                self.trainer.executor.stop()
            except:
                logging.exception('could not kill training.')
                pass

    async def begin_training(self, context: Context, details: dict):
        self.status.reset_error('start_training')
        await self.update_state(State.Preparing)
        try:
            await self.trainer.begin_training(context, details)
        except Exception as e:
            self.status.set_error('start_training', f'Could not start training: {str(e)})')

            logging.exception(self.status._errors)
            self.trainer.stop_training()
            await self.update_state(State.Idle)
            return
        # NOTE 'id' can also be a well known pretrained model identifier.
        self.latest_known_model_id = details['id'] if is_valid_uuid4(details['id']) else None
        await self.update_state(State.Running)

    async def stop_training(self, do_detections: bool = True, save_latest_model: bool = True) -> Union[bool, str]:
        if self.status.state != State.Running:
            logging.warning(f'##### stop_training is called but current state is : {self.status.state}')
            return

        await self.update_state(State.Stopping)

        try:
            result = self.trainer.stop_training()

            if self.latest_known_model_id and self.trainer.training.base_model_id != self.latest_known_model_id:
                if save_latest_model:
                    await self.update_state(State.Uploading)
                    await self.save_model(self.trainer.training.context, self.latest_known_model_id)
                if do_detections:
                    await self.update_state(State.Detecting)
                    try:
                        await self.trainer.do_detections(context=self.trainer.training.context,
                                                         model_id=self.latest_known_model_id,
                                                         model_format=self.trainer.model_format)
                    except Exception as e:
                        logging.exception(f'Could not predict detections: {str(e)}')

            await self.clear_training_data(self.trainer.training.training_folder)
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
            logging.exception('could not save model')
            self.status.set_error('save_model', f'Could not save model: {str(e)}')

        await self.send_status()

    async def clear_training_data(self, training_folder: str):
        self.status.reset_error('clear_training_data')
        try:
            await self.trainer.clear_training_data(training_folder)
        except Exception as e:
            traceback.print_exc()
            self.status.set_error('clear_training_data', f'Could not delete training data: {str(e)}')

    async def check_state(self):
        logging.debug(f'{self.status.state}')
        self.status.reset_error('training_error')
        error = self.trainer.get_error()

        if error is not None:
            try:
                # NOTE test_model_should_be_uploaded_when_training_has_error will result in exception without this try/except block.
                logging.error(error + '\n\n' + self.trainer.get_log()[-1000:])
            except:
                pass
            self.status.set_error('training_error', error)
            await self.stop_training()
            await self.send_status()
            return

        if self.status.state != State.Running:
            return

        if not self.trainer.executor.is_process_running():
            self.status.set_error('training_error', 'Training crashed.')
            logging.info(self.trainer.get_log()[-1000:])
            await self.stop_training()
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
                        hyperparameters=self.trainer.hyperparameters
                    )

                    result = await self.sio_client.call('update_model', (current_training.context.organization, current_training.context.project, jsonable_encoder(new_model)))
                    response = SocketResponse.from_dict(result)

                    if not response.success:
                        error_msg = f'Error for update_model: Response from loop was : {response.__dict__}'
                        logging.error(error_msg)
                        raise Exception(error_msg)

                    logging.info(f'successfully uploaded checkpoint {jsonable_encoder(new_model)}')
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
            errors=self.status._errors
        )
        # TODO can self.trainer be None?
        if self.trainer:
            status.pretrained_models = self.trainer.provided_pretrained_models

        if self.trainer and self.trainer.training:
            status.train_image_count = self.trainer.training.data.train_image_count()
            status.test_image_count = self.trainer.training.data.test_image_count()
            status.skipped_image_count = self.trainer.training.data.skipped_image_count
            status.hyperparameters = self.trainer.hyperparameters

        logging.info(f'sending status {status}')
        result = await self.sio_client.call('update_trainer', jsonable_encoder(status), timeout=1)
        response = SocketResponse.from_dict(result)

        if not response.success:
            logging.error(f'Error for updating: Response from loop was : {response.__dict__}')
            logging.error('Going to kill training. ')
            logging.exception('update trainer failed')

            if status.state != State.Idle:
                # TODO was soll passieren, wenn wir z.B. Stopping sind, und das Event von der Loop abgelehnt wird?
                await self.stop_training(do_detections=False, save_latest_model=False)

    def get_state(self):
        if self.trainer.executor is not None and self.trainer.executor.is_process_running():
            return State.Running
        return State.Idle
