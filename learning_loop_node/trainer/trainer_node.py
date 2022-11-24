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
    model_published: bool = False

    def __init__(self, name: str, trainer: Trainer, uuid: str = None):
        super().__init__(name, uuid)
        self.trainer = trainer
        self.include_router(controls.router, tags=["controls"])
        self.train_loop_busy = False
        active_training.init(self.uuid)

        @self.sio_client.on('begin_training')
        async def on_begin_training(organization: str, project: str, details: dict):
            logging.info('received begin_training from server')
            self.trainer.init(Context(organization=organization, project=project), details)
            self.start_training_task()
            return True

        @self.sio_client.on('stop_training')
        async def stop():
            logging.info(f'### on stop_training received. Current state : {self.status.state}')
            loop = asyncio.get_event_loop()
            try:
                self.stop_training()
            except:
                logging.exception('error in stop_training')
            return True

        @self.on_event("startup")
        @repeat_every(seconds=5, raise_exceptions=True, wait_first=False)
        async def continous_send_status():
            try:
                await self.send_status()
            except:
                logging.exception('could not send status state')

        @self.on_event("shutdown")
        async def shutdown():
            await self.shutdown()

        @self.on_event("startup")
        async def resume_training_if_exists():
            self.start_training_task()

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

    def start_training_task(self):
        loop = asyncio.get_event_loop()
        loop.create_task(self.trainer.train(self.uuid, self.sio_client))

    async def send_status(self):
        if not self.trainer.training and active_training.exists():
            logging.warning('Found active training, but no its not loaded yet. Skipping this status update.')
            return

        state_for_learning_loop = TrainerNode.state_for_learning_loop(
            self.trainer.training.training_state) if self.trainer.training else State.Idle
        status = TrainingStatus(
            id=self.uuid,
            name=self.name,
            state=state_for_learning_loop,
            errors={},
            uptime=self.training_uptime,
            progress=self.progress
        )

        status.pretrained_models = self.trainer.provided_pretrained_models
        status.architecture = self.trainer.model_architecture

        if self.trainer.training:
            status.train_image_count = self.trainer.training.data.train_image_count()
            status.test_image_count = self.trainer.training.data.test_image_count()
            status.skipped_image_count = self.trainer.training.data.skipped_image_count
            status.hyperparameters = self.trainer.hyperparameters
            status.errors = self.trainer.errors._errors

        logging.info(f'sending status {status}')
        result = await self.sio_client.call('update_trainer', jsonable_encoder(status), timeout=1)
        response = SocketResponse.from_dict(result)

        if not response.success:
            logging.error(f'Error for updating: Response from loop was : {response.__dict__}')
            logging.exception('update trainer failed')

    async def shutdown(self):
        logging.info('shutdown detected, stopping training')
        self.trainer.shutdown()

    def get_state(self):
        if self.trainer.executor is not None and self.trainer.executor.is_process_running():
            return State.Running
        return State.Idle

    @property
    def progress(self) -> Union[float, None]:
        return self.trainer.progress if hasattr(self.trainer, 'progress') else None

    @property
    def training_uptime(self) -> Union[int, None]:
        import time
        now = time.time()
        return now - self.trainer.start_time if self.trainer.start_time else None

    @staticmethod
    def state_for_learning_loop(trainer_state: TrainingState):
        if trainer_state == TrainingState.Initialized:
            return 'Training is initialized'
        if trainer_state == TrainingState.DataDownloading:
            return 'Downloading data'
        if trainer_state == TrainingState.DataDownloaded:
            return 'Data downloaded'
        if trainer_state == TrainingState.TrainModelDownloading:
            return 'Downloading model'
        if trainer_state == TrainingState.TrainModelDownloaded:
            return 'Model downloaded'
        if trainer_state == TrainingState.TrainingRunning:
            return State.Running
        if trainer_state == TrainingState.TrainingFinished:
            return 'Training finished'
        if trainer_state == TrainingState.Detecting:
            return State.Detecting
        if trainer_state == TrainingState.ConfusionMatrixSyncing:
            return 'Syncing confusion matrix'
        if trainer_state == TrainingState.ConfusionMatrixSynced:
            return 'Confusion matrix synced'
        if trainer_state == TrainingState.TrainModelUploading:
            return 'Uploading trained model'
        if trainer_state == TrainingState.TrainModelUploaded:
            return 'Trained model uploaded'
        if trainer_state == TrainingState.Detecting:
            return 'calculating detections'
        if trainer_state == TrainingState.Detected:
            return 'Detections calculated'
        if trainer_state == TrainingState.DetectionUploading:
            return 'Uploading detections'
        if trainer_state == TrainingState.ReadyForCleanup:
            return 'Cleaning training'
        return 'unknown state'
