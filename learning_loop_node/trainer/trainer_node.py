import os
import sys
import time
from dataclasses import asdict
from typing import Dict, Optional

from fastapi.encoders import jsonable_encoder
from socketio import AsyncClient, exceptions

from ..data_classes import TrainingStatus
from ..node import Node
from .io_helpers import LastTrainingIO
from .rest import backdoor_controls
from .trainer_logic_generic import TrainerLogicGeneric


class TrainerNode(Node):

    def __init__(self, name: str, trainer_logic: TrainerLogicGeneric, uuid: Optional[str] = None, use_backdoor_controls: bool = False):
        super().__init__(name, uuid, 'trainer')
        trainer_logic._node = self
        self.trainer_logic = trainer_logic
        self.last_training_io = LastTrainingIO(self.uuid)
        self.trainer_logic._last_training_io = self.last_training_io

        self.first_idle_time: float | None = None
        if os.environ.get('TRAINER_IDLE_TIMEOUT_SEC', 0.0):
            self.idle_timeout = float(os.environ.get('TRAINER_IDLE_TIMEOUT_SEC', 0.0))
        else:
            self.idle_timeout = 0.0
        if self.idle_timeout:
            self.log.info(
                f'Trainer started with an idle_timeout of {self.idle_timeout} seconds. Note that shutdown does not work if docker container has the restart policy set to always')

        if use_backdoor_controls or os.environ.get('USE_BACKDOOR_CONTROLS', '0').lower() in ('1', 'true'):
            self.include_router(backdoor_controls.router, tags=["controls"])

    # ----------------------------------- NODE LIVECYCLE METHODS --------------------------

    async def on_startup(self):
        pass

    async def on_shutdown(self):
        self.log.info('shutdown detected, stopping training')
        await self.trainer_logic.on_shutdown()

    async def on_repeat(self):
        try:
            if await self.trainer_logic.try_continue_run_if_incomplete():
                return  # NOTE: we prevent sending idle status after starting a continuation
            await self.send_status()
            self.check_idle_timeout()
        except exceptions.TimeoutError:
            self.log.warning('timeout when sending status to learning loop, reconnecting sio_client')
            await self.sio_client.disconnect()  # NOTE: reconnect happens in node._on_repeat
        except Exception as e:
            self.log.exception(f'could not send status state: {e}')

    # ---------------------------------------------- NODE METHODS ---------------------------------------------------

    def register_sio_events(self, sio_client: AsyncClient):

        @sio_client.event
        async def begin_training(organization: str, project: str, details: Dict):
            self.log.info('received begin_training from server')
            await self.trainer_logic.begin_training(organization, project, details)
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
            self.log.debug('cannot send status - not connected to the Learning Loop')
            return

        status = TrainingStatus(id=self.uuid,
                                name=self.name,
                                state=self.trainer_logic.state,
                                errors={},
                                uptime=self.trainer_logic.training_uptime,
                                progress=self.trainer_logic.general_progress)

        status.pretrained_models = self.trainer_logic.provided_pretrained_models
        status.architecture = self.trainer_logic.model_architecture

        if data := self.trainer_logic.training_data:
            status.train_image_count = data.train_image_count()
            status.test_image_count = data.test_image_count()
            status.skipped_image_count = data.skipped_image_count
            status.hyperparameters = self.trainer_logic.hyperparameters_for_state_sync
            status.errors = self.trainer_logic.errors.errors
            status.context = self.trainer_logic.training_context

        self.log.debug('sending status: %s', status.short_str())
        result = await self.sio_client.call('update_trainer', jsonable_encoder(asdict(status)), timeout=30)
        if isinstance(result, Dict) and not result['success']:
            self.socket_connection_broken = True
            self.log.error('Error when sending status update: Response from loop was:\n %s', result)

    def check_idle_timeout(self):
        if not self.idle_timeout:
            return

        if self.trainer_logic.state == 'idle':
            if self.first_idle_time is None:
                self.first_idle_time = time.time()
            idle_time = time.time() - self.first_idle_time
            if idle_time > self.idle_timeout:
                self.log.info('Trainer has been idle for %.2f s (with timeout %.2f s). Shutting down.',
                              idle_time, self.idle_timeout)
                sys.exit(0)
            self.log.debug('idle time: %.2f s / %.2f s', idle_time, self.idle_timeout)
        else:
            self.first_idle_time = None
