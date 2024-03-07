import os
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional

from socketio import AsyncClient

from ..data_classes import Context, Errors, PretrainedModel, TrainerState, Training, TrainingData
from ..data_exchanger import DataExchanger
from ..loop_communication import LoopCommunicator
from .io_helpers import ActiveTrainingIO, LastTrainingIO

if TYPE_CHECKING:
    from .trainer_node import TrainerNode


class TrainerLogicAbstraction(ABC):

    def __init__(self, model_format: str):

        # NOTE: String to be used in the file path for the model on the server:
        # '/{context.organization}/projects/{context.project}/models/{model_id}/{model_format}/file'
        self.model_format: str = model_format

        self._node: Optional['TrainerNode'] = None  # type: ignore
        self._last_training_io: Optional[LastTrainingIO] = None  # type: ignore
        self.errors = Errors()

        self._training: Optional[Training] = None
        self._active_training_io: Optional[ActiveTrainingIO] = None

        self.restart_after_training = os.environ.get('RESTART_AFTER_TRAINING', 'FALSE').lower() in ['true', '1']
        self.keep_old_trainings = os.environ.get('KEEP_OLD_TRAININGS', 'FALSE').lower() in ['true', '1']
        self.inference_batch_size = int(os.environ.get('INFERENCE_BATCH_SIZE', '10'))

    @property
    def node(self) -> 'TrainerNode':
        assert self._node is not None, 'node should be set by TrainerNode before initialization'
        return self._node

    @property
    def last_training_io(self) -> LastTrainingIO:
        assert self._last_training_io is not None, 'last_training_io should be set by TrainerNode before initialization'
        return self._last_training_io

    @property
    def data_exchanger(self) -> DataExchanger:
        return self.node.data_exchanger

    @property
    def loop_communicator(self) -> LoopCommunicator:
        return self.node.loop_communicator

    @property
    def node_uuid(self) -> str:
        return self.node.uuid

    @property
    def sio_client(self) -> AsyncClient:
        return self.node.sio_client

    @property
    def active_training_io(self) -> ActiveTrainingIO:
        assert self._active_training_io is not None, 'active_training_io must be set, call `init` first'
        return self._active_training_io

    @property
    def training_active(self) -> bool:
        """_training and _active_training_io are set in 'init_new_training' or 'init_from_last_training'"""
        return self._training is not None and self._active_training_io is not None

    @property
    def state(self) -> str:
        if (not self.training_active) or (self.active_training.training_state is None):
            return TrainerState.Idle.value
        else:
            return self.active_training.training_state

    @property
    def active_training(self) -> Training:
        assert self._training is not None, 'training must be initialized, call `init` first'
        return self._training

    @property
    def training_uptime(self) -> Optional[float]:
        if self.active_training:
            return time.time() - self.active_training.start_time
        return None

    @property
    def training_data(self) -> Optional[TrainingData]:
        if self.training_active and self.active_training.data:
            return self.active_training.data
        return None

    @property
    def training_context(self) -> Optional[Context]:
        if self.training_active:
            return self.active_training.context
        return None

    # --- ABSTRACT PROPERTIES
    # --------- implemented in TrainerLogicGeneric

    @property
    @abstractmethod
    def general_progress(self) -> Optional[float]:
        """Returns the general progress of the training per state or None if idle"""

    # --------- implemented in TrainerLogic(with Executor)
    @property
    @abstractmethod
    def hyperparameters(self) -> Optional[Dict]:
        """Returns the currently used hyperparameters if available"""

    # --------- not implemented in any abstract class
    @property
    @abstractmethod
    def model_architecture(self) -> Optional[str]:
        """Returns the architecture name of the model if available"""

    @property
    @abstractmethod
    def provided_pretrained_models(self) -> List[PretrainedModel]:
        """Returns the list of provided pretrained models"""

    # --- ABSTRACT METHODS -----
    # --------- implemented in TrainerLogicGeneric ---

    @abstractmethod
    async def on_shutdown(self):
        """Called when the trainer is shut down"""

    @abstractmethod
    async def begin_training(self, organization: str, project: str, details: dict):
        """Starts the training process"""

    @abstractmethod
    async def try_continue_run_if_incomplete(self) -> bool:
        """Start training continuation if possible, returns True if continuation started"""

    # --- implemented in TrainerLogic(with Executor) ---

    @abstractmethod
    async def stop(self):
        """Stops the training process"""
