import asyncio
import time
from dataclasses import asdict
from typing import Dict, Optional, Union

from dacite import from_dict
from fastapi.encoders import jsonable_encoder
from socketio import AsyncClient

from ..data_classes import Context, NodeState, TrainingState, TrainingStatus
from ..node import Node
from ..socket_response import SocketResponse
from .io_helpers import LastTrainingIO
from .rest import controls
from .trainer_logic import TrainerLogic


class TrainerNode(Node):

    def __init__(self, name: str, trainer_logic: TrainerLogic, uuid: Optional[str] = None):
        super().__init__(name, uuid)
        self.trainer_logic = trainer_logic
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
        await self.trainer_logic.shutdown()

    def register_sio_events(self, sio_client: AsyncClient):

        @sio_client.event
        async def begin_training(organization: str, project: str, details: Dict):
            assert self._sio_client is not None
            self.log.info('received begin_training from server')
            self.trainer_logic.init(Context(organization=organization, project=project), details, self)
            self.start_training_task()
            return True

        @sio_client.event
        async def stop_training():
            self.log.info(f'### on stop_training received. Current state : {self.status.state}')
            try:
                await self.trainer_logic.stop()
            except Exception:
                self.log.exception('error in stop_training')
            return True

    def start_training_task(self):
        loop = asyncio.get_event_loop()
        loop.create_task(self.trainer_logic.train())

    async def send_status(self):  # TODO rename?

        if self._sio_client is None or not self._sio_client.connected:
            self.log.info('could not send status -- we are not connected to the Learning Loop')
            return

        if not self.trainer_logic.is_initialized and self.last_training_io.exists():
            self.log.warning('Found active training, starting now.')
            self.start_training_task()
            return

        if not self.trainer_logic.is_initialized:
            state_for_learning_loop = 'unknown state'
        else:
            assert self.trainer_logic.training.training_state is not None
            state_for_learning_loop = TrainerNode.state_for_learning_loop(self.trainer_logic.training.training_state)

        status = TrainingStatus(
            uuid=self.uuid,
            name=self.name,
            state=state_for_learning_loop,
            errors={},
            uptime=self.training_uptime,
            progress=self.progress
        )

        status.pretrained_models = self.trainer_logic.provided_pretrained_models
        status.architecture = self.trainer_logic.model_architecture

        if self.trainer_logic.is_initialized and self.trainer_logic.training.data:
            status.train_image_count = self.trainer_logic.training.data.train_image_count()
            status.test_image_count = self.trainer_logic.training.data.test_image_count()
            status.skipped_image_count = self.trainer_logic.training.data.skipped_image_count
            status.hyperparameters = self.trainer_logic.hyperparameters
            status.errors = self.trainer_logic.errors.errors

        self.log.info(f'sending status {status}')
        result = await self._sio_client.call('update_trainer', jsonable_encoder(asdict(status)), timeout=1)
        assert isinstance(result, Dict)
        response = from_dict(data_class=SocketResponse, data=result)

        if not response.success:
            self.log.error(f'Error for updating: Response from loop was : {response.__dict__}')
            self.log.exception('update trainer failed')

    async def get_state(self):
        if self.trainer_logic._executor is not None and self.trainer_logic._executor.is_process_running():  # pylint: disable=protected-access
            return NodeState.Running
        return NodeState.Idle

    @property
    def progress(self) -> Union[float, None]:
        return self.trainer_logic.progress if (self.trainer_logic is not None and
                                               hasattr(self.trainer_logic, 'progress')) else None

    @property
    def training_uptime(self) -> Union[float, None]:
        return time.time() - self.trainer_logic.start_time if self.trainer_logic.start_time else None

    @staticmethod
    def state_for_learning_loop(trainer_state: Union[TrainingState, str]) -> str:
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
