import asyncio
import time
from dataclasses import asdict
from typing import Dict, Optional, Union

from dacite import from_dict
from fastapi.encoders import jsonable_encoder
from socketio import AsyncClient

from ..data_classes import Context, NodeState, TrainingState, TrainingStatus
from ..data_classes.socket_response import SocketResponse
from ..node import Node
from .io_helpers import LastTrainingIO
from .rest import backdoor_controls, controls
from .trainer_logic import TrainerLogic


class TrainerNode(Node):

    def __init__(self, name: str, trainer_logic: TrainerLogic, uuid: Optional[str] = None, use_backdoor_controls: bool = False):
        super().__init__(name, uuid)
        trainer_logic._node = self  # pylint: disable=protected-access
        self.trainer_logic = trainer_logic
        self.last_training_io = LastTrainingIO(self.uuid)
        self.include_router(controls.router, tags=["controls"])
        if use_backdoor_controls:
            self.include_router(backdoor_controls.router, tags=["controls"])

    # --------------------------------------------------- STATUS ---------------------------------------------------

    @property
    def progress(self) -> Union[float, None]:
        return self.trainer_logic.general_progress if (self.trainer_logic is not None and
                                                       hasattr(self.trainer_logic, 'general_progress')) else None

    @property
    def training_uptime(self) -> Union[float, None]:
        return time.time() - self.trainer_logic.start_time if self.trainer_logic.start_time else None

    # ----------------------------------- LIVECYCLE: ABSTRACT NODE METHODS --------------------------

    async def on_startup(self):
        pass

    async def on_shutdown(self):
        self.log.info('shutdown detected, stopping training')
        await self.trainer_logic.shutdown()

    async def on_repeat(self):
        try:
            await self.send_status()
            await self.continue_run_if_incomplete()
        except Exception as e:
            if isinstance(e, asyncio.TimeoutError):
                self.log.warning('timeout when sending status to learning loop, reconnecting sio_client')
                await self.sio_client.disconnect()
                # NOTE: reconnect happens in node._on_repeat
            else:
                self.log.exception(f'could not send status state: {e}')

    # ---------------------------------------------- NODE ABSTRACT METHODS ---------------------------------------------------

    def register_sio_events(self, sio_client: AsyncClient):

        @sio_client.event
        async def begin_training(organization: str, project: str, details: Dict):
            assert self._sio_client is not None
            self.log.info('received begin_training from server')
            self.trainer_logic.init(Context(organization=organization, project=project), details, self)
            asyncio.get_event_loop().create_task(self.trainer_logic.run())
            return True

        @sio_client.event
        async def stop_training():
            self.log.info(f'stop_training received. Current state : {self.status.state}')
            try:
                await self.trainer_logic.stop()
            except Exception:
                self.log.exception('error in stop_training. Exception:')
            return True

    async def send_status(self):
        if self._sio_client is None or not self._sio_client.connected:
            self.log.warning('cannot send status - not connected to the Learning Loop')
            return

        if not self.trainer_logic.is_initialized:
            state_for_learning_loop = str(NodeState.Idle.value)
        else:
            assert self.trainer_logic.training.training_state is not None
            state_for_learning_loop = TrainerNode.state_for_learning_loop(
                self.trainer_logic.training.training_state)

        status = TrainingStatus(id=self.uuid,
                                name=self.name,
                                state=state_for_learning_loop,
                                errors={},
                                uptime=self.training_uptime,
                                progress=self.progress)

        status.pretrained_models = self.trainer_logic.provided_pretrained_models
        status.architecture = self.trainer_logic.model_architecture

        if self.trainer_logic.is_initialized and self.trainer_logic.training.data:
            status.train_image_count = self.trainer_logic.training.data.train_image_count()
            status.test_image_count = self.trainer_logic.training.data.test_image_count()
            status.skipped_image_count = self.trainer_logic.training.data.skipped_image_count
            status.hyperparameters = self.trainer_logic.hyperparameters
            status.errors = self.trainer_logic.errors.errors
            status.context = self.trainer_logic.training.context

        self.log.info(f'sending status: {status.short_str()}')
        result = await self._sio_client.call('update_trainer', jsonable_encoder(asdict(status)), timeout=30)
        assert isinstance(result, Dict)
        response = from_dict(data_class=SocketResponse, data=result)

        if not response.success:
            self.log.error(f'Error when sending status update: Response from loop was:\n {asdict(response)}')

    async def continue_run_if_incomplete(self):
        if not self.trainer_logic.is_initialized and self.last_training_io.exists():
            self.log.info('found incomplete training, continuing now.')
            asyncio.get_event_loop().create_task(self.trainer_logic.run())

    async def get_state(self):
        if self.trainer_logic._executor is not None and self.trainer_logic._executor.is_process_running():  # pylint: disable=protected-access
            return NodeState.Running
        return NodeState.Idle

    def get_node_type(self):
        return 'trainer'

    # --------------------------------------------------- HELPER ---------------------------------------------------

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
