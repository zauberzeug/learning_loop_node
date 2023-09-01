import asyncio
import time
from typing import Dict, Optional, Union

from fastapi.encoders import jsonable_encoder

from learning_loop_node.data_classes import Context
from learning_loop_node.data_classes import State as TrainingState
from learning_loop_node.data_classes import TrainingStatus
from learning_loop_node.node import Node, State

from ..socket_response import SocketResponse
from .rest import controls
from .trainer import Trainer


class TrainerNode(Node):

    def __init__(self, name: str, trainer: Trainer, uuid: Optional[str] = None):
        super().__init__(name, uuid)
        self.trainer = trainer
        self.train_loop_busy = False
        self.model_published: bool = False
        self.include_router(controls.router, tags=["controls"])

    async def on_repeat(self):
        await super().on_repeat()
        try:
            await self.send_status()
        except Exception as e:
            self.log.exception(f'could not send status state: {e}')

    async def on_shutdown(self):
        await super().on_shutdown()
        self.log.info('shutdown detected, stopping training')
        self.trainer.shutdown()

    async def create_sio_client(self):
        await super().create_sio_client()
        assert self.sio_client is not None

        @self.sio_client.event
        async def begin_training(organization: str, project: str, details: dict):
            self.log.info('received begin_training from server')
            self.trainer.init(Context(organization=organization, project=project), details, self.uuid)
            self.start_training_task()
            return True

        @self.sio_client.event
        async def stop_training():
            self.log.info(f'### on stop_training received. Current state : {self.status.state}')
            try:
                self.trainer.stop()
            except Exception:
                self.log.exception('error in stop_training')
            return True

    def start_training_task(self):
        loop = asyncio.get_event_loop()
        loop.create_task(self.trainer.train(self.uuid, self.sio_client))

    async def send_status(self):
        if self.sio_client is None or not self.sio_client.connected:
            self.log.info('could not send status -- we are not connected to the Learning Loop')
            return

        if not self.trainer.training and self.trainer.active_training_io and self.trainer.active_training_io.exists():
            self.log.warning('Found active training, starting now.')
            self.start_training_task()
            return

        if not self.trainer.training or not self.trainer.training.training_state:
            state_for_learning_loop = 'unknown state'
        else:
            state_for_learning_loop = TrainerNode.state_for_learning_loop(self.trainer.training.training_state)

        status = TrainingStatus(
            uuid=self.uuid,
            name=self.name,
            state=state_for_learning_loop,
            errors={},
            uptime=self.training_uptime,
            progress=self.progress
        )

        status.pretrained_models = self.trainer.provided_pretrained_models
        status.architecture = self.trainer.model_architecture

        if self.trainer.training and self.trainer.training.data:
            status.train_image_count = self.trainer.training.data.train_image_count()
            status.test_image_count = self.trainer.training.data.test_image_count()
            status.skipped_image_count = self.trainer.training.data.skipped_image_count
            status.hyperparameters = self.trainer.hyperparameters
            status.errors = self.trainer.errors.errors

        self.log.info(f'sending status {status}')
        result = await self.sio_client.call('update_trainer', jsonable_encoder(status), timeout=1)
        assert isinstance(result, Dict)
        response = SocketResponse.from_dict(result)

        if not response.success:
            self.log.error(f'Error for updating: Response from loop was : {response.__dict__}')
            self.log.exception('update trainer failed')

    async def get_state(self):
        if self.trainer.executor is not None and self.trainer.executor.is_process_running():
            return State.Running
        return State.Idle

    @property
    def progress(self) -> Union[float, None]:
        return self.trainer.progress if (self.trainer is not None and hasattr(self.trainer, 'progress')) else None

    @property
    def training_uptime(self) -> Union[float, None]:
        return time.time() - self.trainer.start_time if self.trainer.start_time else None

    @staticmethod
    def state_for_learning_loop(trainer_state: TrainingState) -> str:
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

    def get_node_type(self):
        return 'trainer'
