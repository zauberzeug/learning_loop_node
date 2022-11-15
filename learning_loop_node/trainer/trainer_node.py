import asyncio
from learning_loop_node.context import Context
import traceback
from fastapi_utils.tasks import repeat_every
from fastapi.encoders import jsonable_encoder
from typing import Union
from uuid import uuid4
from icecream import ic
from learning_loop_node.trainer.training import TrainingOut
from .model import Model
from .trainer import Trainer
from .training_status import TrainingStatus
from learning_loop_node.node import Node, State
import logging
from ..socket_response import SocketResponse
from .rest import controls
from learning_loop_node.trainer import active_training
from learning_loop_node.trainer.training import State as TrainingState
from learning_loop_node.trainer import training_syncronizer


class TrainerNode(Node):
    trainer: Trainer
    skip_check_state: bool = False
    model_published: bool = False

    def __init__(self, name: str, trainer: Trainer, uuid: str = None):
        super().__init__(name, uuid)
        self.trainer = trainer
        self.include_router(controls.router, tags=["controls"])
        self.train_loop_busy = False

        @self.sio_client.on('begin_training')
        async def on_begin_training(organization: str, project: str, details: dict):
            logging.info('received begin_training from server')
            self.trainer.init(Context(organization=organization, project=project), details)
            loop = asyncio.get_event_loop()
            loop.create_task(self.wait_and_train())
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
            await self.shutdown()

        @self.on_event("startup")
        @repeat_every(seconds=1, raise_exceptions=True, wait_first=False)
        async def continous_training():
            try:
                await self.train()
            except:
                logging.exception('Error in continous_training')

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
        await self.update_state(State.Running)

    def stop_training(self, save_and_detect: bool = True) -> Union[bool, str]:
        result = self.trainer.stop()

    async def save_model(self, context: Context):
        self.status.reset_error('save_model')
        uploaded_model = None
        try:
            uploaded_model = await self.trainer.save_model(context)
        except Exception as e:
            logging.exception('could not save model')
            self.status.set_error('save_model', f'Could not save model: {str(e)}')

        await self.send_status()
        return uploaded_model

    async def wait_and_train(self):
        while self.train_loop_busy:
            await asyncio.sleep(0.1)
        logging.info('going to start training')
        await self.train()

    async def train(self):
        if self.train_loop_busy:
            return

        self.train_loop_busy = True
        await self.trainer.train()
        # logging.error('train loop ended')
        self.train_loop_busy = False

    async def check_state(self):
        return
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
        # TODO use new training_syncronizer

        self.status.reset_error('get_new_model')

        try:
            current_training = self.trainer.training
            if self.status.state == State.Running and current_training:
                model = self.trainer.get_new_model()
                logging.debug(f'new model {model}')
                if model:
                    new_training = TrainingOut(
                        trainer_id=self.uuid,
                        confusion_matrix=model.confusion_matrix,
                        train_image_count=current_training.data.train_image_count(),
                        test_image_count=current_training.data.test_image_count(),
                        hyperparameters=self.trainer.hyperparameters
                    )

                    result = await self.sio_client.call('update_training', (current_training.context.organization, current_training.context.project, jsonable_encoder(new_training)))
                    response = SocketResponse.from_dict(result)

                    if not response.success:
                        error_msg = f'Error for update_training: Response from loop was : {response.__dict__}'
                        logging.error(error_msg)
                        raise Exception(error_msg)

                    logging.info(f'successfully updated training {jsonable_encoder(new_training)}')
                    self.trainer.on_model_published(model)
                    self.model_published = True
                    # hier.

        except Exception as e:
            msg = f'Could not get new model: {str(e)}'
            logging.exception(msg)
            self.status.set_error('get_new_model', msg)

        await self.send_status()

    async def send_status(self):
        state_for_learning_loop = TrainerNode.state_for_learning_loop(
            self.trainer.training.training_state) if self.trainer.training else State.Idle
        logging.warning(f'want to send state to learning loop : {state_for_learning_loop}')
        status = TrainingStatus(
            id=self.uuid,
            name=self.name,
            state=TrainerNode.state_for_learning_loop(
                self.trainer.training.training_state) if self.trainer.training else State.Idle,
            uptime=self.training_uptime,
            errors=self.status._errors,
            progress=self.progress
        )
        # TODO can self.trainer be None?
        if self.trainer:
            status.pretrained_models = self.trainer.provided_pretrained_models
            status.architecture = self.trainer.model_architecture

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

    async def shutdown(self):
        logging.info('shutdown detected, stopping training')
        self.trainer.stop()
        if self.trainer.executor:
            try:
                self.trainer.executor.stop()
            except:
                logging.exception('could not kill training.')

    def get_state(self):
        if self.trainer.executor is not None and self.trainer.executor.is_process_running():
            return State.Running
        return State.Idle

    @ property
    def progress(self) -> Union[float, None]:
        return self.trainer.progress if hasattr(self.trainer, 'progress') else None

    @ property
    def training_uptime(self) -> Union[int, None]:
        import time
        now = time.time()
        return now - self.trainer.start_time if self.trainer.start_time else None

    @staticmethod
    def state_for_learning_loop(trainer_state: TrainingState):
        if trainer_state in [TrainingState.Initialized,
                             TrainingState.DataDownloading,
                             TrainingState.DataDownloaded,
                             TrainingState.TrainModelDownloading, TrainingState.TrainModelDownloaded]:
            return State.Preparing
        if trainer_state == TrainingState.TrainingRunning:
            return State.Running
        if trainer_state == TrainingState.TrainingFinished:
            return State.Running
        if trainer_state in [TrainingState.TrainingFinished,
                             TrainingState.ConfusionMatrixSyncing,
                             TrainingState.ConfusionMatrixSynced,
                             TrainingState.TrainModelUploading,
                             TrainingState.TrainModelUploaded]:
            return State.Running
        if trainer_state in [TrainingState.Detecting,
                             TrainingState.Detected]:
            return State.Detecting
        if trainer_state == TrainingState.DetectionUploading:
            return State.Detecting
        if trainer_state == TrainingState.ReadyForCleanup:
            return State.Stopping

        logging.error(trainer_state)
        return 'unknown'
