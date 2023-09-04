import asyncio
import logging
import time
from typing import Dict, Optional, Union

from fastapi import APIRouter, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from socketio import AsyncClient

from learning_loop_node.data_classes import (Context, NodeState, TrainingState,
                                             TrainingStatus)
from learning_loop_node.loop_communication import glc
from learning_loop_node.node import Node
from learning_loop_node.trainer.trainer import Trainer
from learning_loop_node.trainer.training_io_helpers import LastTrainingIO

from ..socket_response import SocketResponse
from .rest import controls

router = APIRouter()


@router.post("/controls/detect/{organization}/{project}/{version}")
async def operation_mode(organization: str, project: str, version: str, request: Request):
    '''
    Example Usage
        curl -X POST localhost/controls/detect/<organization>/<project>/<model_version>
    '''
    path = f'/{organization}/projects/{project}/models'
    response = await glc.get(path)
    if response.status_code != 200:
        raise HTTPException(404, 'could not load latest model')
    models = response.json()['models']
    model_id = next(m for m in models if m['version'] == version)['id']
    logging.info(model_id)
    trainer: Trainer = request.app.trainer
    await trainer.do_detections()
    return "OK"


class TrainerNode(Node):

    def __init__(self, name: str, trainer: Trainer, uuid: Optional[str] = None):
        super().__init__(name, uuid)
        self.trainer = trainer
        self.train_loop_busy = False
        self.model_published: bool = False
        self.last_training_io = LastTrainingIO(self.uuid)
        self.include_router(controls.router, tags=["controls"])

    async def on_startup(self):
        pass

    async def on_repeat(self):
        try:
            await self.send_status()
        except Exception as e:
            self.log.exception(f'could not send status state: {e}')

    async def on_shutdown(self):
        self.log.info('shutdown detected, stopping training')
        self.trainer.shutdown()

    def register_sio_events(self, sio_client: AsyncClient):

        @sio_client.event
        async def begin_training(organization: str, project: str, details: dict):
            assert self._sio_client is not None
            self.log.info('received begin_training from server')
            self.trainer.init(Context(organization=organization, project=project),
                              details, self.uuid, self._sio_client, self.last_training_io)
            self.start_training_task()
            return True

        @sio_client.event
        async def stop_training():
            self.log.info(f'### on stop_training received. Current state : {self.status.state}')
            try:
                self.trainer.stop()
            except Exception:
                self.log.exception('error in stop_training')
            return True

    def start_training_task(self):
        loop = asyncio.get_event_loop()
        loop.create_task(self.trainer.train())

    async def send_status(self):
        # NOTE: the send status is used to potentially start an existing training P?

        if self._sio_client is None or not self._sio_client.connected:
            self.log.info('could not send status -- we are not connected to the Learning Loop')
            return

        if not self.trainer.is_initialized() and self.last_training_io.exists():
            self.log.warning('Found active training, starting now.')
            self.start_training_task()
            return

        if not self.trainer.is_initialized():
            state_for_learning_loop = 'unknown state'
        else:
            assert self.trainer.training.training_state is not None
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

        if self.trainer.is_initialized() and self.trainer.training.data:
            status.train_image_count = self.trainer.training.data.train_image_count()
            status.test_image_count = self.trainer.training.data.test_image_count()
            status.skipped_image_count = self.trainer.training.data.skipped_image_count
            status.hyperparameters = self.trainer.hyperparameters
            status.errors = self.trainer.errors.errors

        self.log.info(f'sending status {status}')
        result = await self._sio_client.call('update_trainer', jsonable_encoder(status), timeout=1)
        assert isinstance(result, Dict)
        response = SocketResponse.from_dict(result)

        if not response.success:
            self.log.error(f'Error for updating: Response from loop was : {response.__dict__}')
            self.log.exception('update trainer failed')

    async def get_state(self):
        if self.trainer.executor is not None and self.trainer.executor.is_process_running():
            return NodeState.Running
        return NodeState.Idle

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
            return NodeState.Running
        if trainer_state == TrainingState.TrainingFinished:
            return 'Training finished'
        if trainer_state == TrainingState.Detecting:
            return NodeState.Detecting
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
