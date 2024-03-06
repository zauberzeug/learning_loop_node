import asyncio
import time
from dataclasses import asdict
from typing import Dict, Optional, Union

from dacite import from_dict
from fastapi.encoders import jsonable_encoder
from socketio import AsyncClient

from ..data_classes import Context, NodeState, TrainerState, TrainingStatus
from ..data_classes.socket_response import SocketResponse
from ..node import Node
from .io_helpers import LastTrainingIO
from .rest import backdoor_controls, controls
from .trainer_logic import TrainerLogic


class TrainerNode(Node):

    def __init__(self, name: str, trainer_logic: TrainerLogic, uuid: Optional[str] = None, use_backdoor_controls: bool = False):
        super().__init__(name, uuid, 'trainer')
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
            if await self.continue_run_if_incomplete():
                return  # NOTE: we prevent sending idle status after starting a continuation
            await self.send_status()
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
            self.log.info('received begin_training from server')
            self.trainer_logic.init_new_training(Context(organization=organization, project=project), details)
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
        if not self.sio_client.connected:
            self.log.warning('cannot send status - not connected to the Learning Loop')
            return

        status = TrainingStatus(id=self.uuid,
                                name=self.name,
                                state=self.trainer_logic.state,
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
        result = await self.sio_client.call('update_trainer', jsonable_encoder(asdict(status)), timeout=30)
        assert isinstance(result, Dict)
        if not result['success']:
            self.log.error(f'Error when sending status update: Response from loop was:\n {result}')

    async def continue_run_if_incomplete(self) -> bool:
        if not self.trainer_logic.is_initialized and self.last_training_io.exists():
            self.log.info('found incomplete training, continuing now.')
            self.trainer_logic.init_from_last_training()
            asyncio.get_event_loop().create_task(self.trainer_logic.run())
            return True
        return False

    # --------------------------------------------------- HELPER ---------------------------------------------------

    @staticmethod
    def state_for_learning_loop(trainer_state: Union[TrainerState, str]) -> str:
        if trainer_state == TrainerState.Initialized:
            return 'Training is initialized'
        if trainer_state == TrainerState.DataDownloading:
            return 'Downloading data'
        if trainer_state == TrainerState.DataDownloaded:
            return 'Data downloaded'
        if trainer_state == TrainerState.TrainModelDownloading:
            return 'Downloading model'
        if trainer_state == TrainerState.TrainModelDownloaded:
            return 'Model downloaded'
        if trainer_state == TrainerState.TrainingRunning:
            return NodeState.Running
        if trainer_state == TrainerState.TrainingFinished:
            return 'Training finished'
        if trainer_state == TrainerState.Detecting:
            return NodeState.Detecting
        if trainer_state == TrainerState.ConfusionMatrixSyncing:
            return 'Syncing confusion matrix'
        if trainer_state == TrainerState.ConfusionMatrixSynced:
            return 'Confusion matrix synced'
        if trainer_state == TrainerState.TrainModelUploading:
            return 'Uploading trained model'
        if trainer_state == TrainerState.TrainModelUploaded:
            return 'Trained model uploaded'
        if trainer_state == TrainerState.Detecting:
            return 'calculating detections'
        if trainer_state == TrainerState.Detected:
            return 'Detections calculated'
        if trainer_state == TrainerState.DetectionUploading:
            return 'Uploading detections'
        if trainer_state == TrainerState.ReadyForCleanup:
            return 'Cleaning training'
        return 'unknown state'
