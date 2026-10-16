import asyncio
import json
import logging
import shutil
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Callable, Coroutine, Dict, List, Optional

from fastapi.encoders import jsonable_encoder

from ..data_classes import (
    Context,
    Errors,
    PretrainedModel,
    Training,
    TrainingOut,
    TrainingStateData,
    TrainingStatus,
)
from ..enums import TrainerState
from ..helpers.misc import (
    create_project_folder,
    delete_all_training_folders,
    is_valid_uuid4,
)
from .downloader import TrainingsDownloader
from .exceptions import CriticalError, ModelAlreadyUploadedError, NodeNeedsRestartError, TrainingRefusedError
from .io_helpers import ActiveTrainingIO, EnvironmentVars, LastTrainingIO
from .training_slot import TrainingSlot, training_slot_from_env
from .validation_set import assign_validation_set

if TYPE_CHECKING:
    from .trainer_node import TrainerNode

STATE_RETRY_DELAY = 5  # seconds before the first retry of a failed state, doubled for every further one
MAX_STATE_RETRY_DELAY = 300  # seconds

logger = logging.getLogger('learning_loop_node.trainer_logic_generic')

# The states after the training has run, which upload its model and detections.
_WIND_DOWN_STATES = frozenset({TrainerState.TrainingFinished, TrainerState.ConfusionMatrixSyncing,
                               TrainerState.ConfusionMatrixSynced, TrainerState.TrainModelUploading,
                               TrainerState.TrainModelUploaded, TrainerState.Detecting, TrainerState.Detected,
                               TrainerState.DetectionUploading, TrainerState.ReadyForCleanup})


def _retry_delay(failures: int) -> float:
    """The wait before the next attempt of a state that has failed `failures` times in a row."""
    return min(STATE_RETRY_DELAY * 2 ** min(failures - 1, 16), MAX_STATE_RETRY_DELAY)


class TrainerLogicGeneric(ABC):

    def __init__(self, model_format: str, training_slot: TrainingSlot | None = None) -> None:

        # NOTE: model_format is used in the file path for the model on the server:
        # It acts as a key for list of files (cf. _get_latest_model_files)
        # '/{context.organization}/projects/{context.project}/models/{model_id}/{model_format}/file'
        self.model_format: str = model_format
        self.errors = Errors()
        self._state_failures: Dict[str, int] = {}  # failed attempts in a row, by error key

        self.training_task: Optional[asyncio.Task] = None
        self.shutdown_event: asyncio.Event = asyncio.Event()

        self._node: Optional['TrainerNode'] = None  # type: ignore
        self._last_training_io: Optional[LastTrainingIO] = None  # type: ignore

        self._training: Optional[Training] = None
        self._active_training_io: Optional[ActiveTrainingIO] = None
        self._environment_vars = EnvironmentVars()
        # whether the last attempt of the current state failed, so it is being retried
        self._state_failed: bool = False
        self.training_slot = training_slot or training_slot_from_env()

    # ---------------------------------------- PROPERTIES TO AVOID CHECKING FOR NONE ----------------------------------------

    @property
    def node(self) -> 'TrainerNode':
        assert self._node is not None, 'node should be set by TrainerNode before initialization'
        return self._node

    @property
    def last_training_io(self) -> LastTrainingIO:
        assert self._last_training_io is not None, 'last_training_io should be set by TrainerNode before initialization'
        return self._last_training_io

    @property
    def active_training_io(self) -> ActiveTrainingIO:
        assert self._active_training_io is not None, 'active_training_io must be set, call `init` first'
        return self._active_training_io

    @property
    def training(self) -> Training:
        assert self._training is not None, 'training must be initialized, call `init` first'
        return self._training

    @property
    def hyperparameters(self) -> Dict[str, Any]:
        assert self._training is not None, 'Training should have data'
        return self._training.hyperparameters

    # ---------------------------------------- PROPERTIES ----------------------------------------

    @property
    def training_context(self) -> Optional[Context]:
        if self.training_active:
            return self.training.context
        return None

    @property
    def training_active(self) -> bool:
        """_training and _active_training_io are set in 'init_new_training' or 'init_from_last_training'.
        """
        return self._training is not None and self._active_training_io is not None

    @property
    def state(self) -> str:
        """Returns the current state of the training.

        Without a training the state is `idle`, or `blocked` while a sibling node holds the training slot.
        A training resumed after a restart reports `waiting_for_slot` until it gets the slot: it has a
        training, so it is never `blocked`.
        """
        if not self.training_active:
            return TrainerState.Blocked.value if self.training_slot.taken_by_sibling else TrainerState.Idle.value
        if not self.training_slot.held:
            return TrainerState.WaitingForSlot.value
        if self.training.training_state is None:
            return TrainerState.Idle.value
        return self.training.training_state

    @property
    def training_uptime(self) -> Optional[float]:
        """Livetime of current Training object. Start time is set during initialization of Training object.
        """
        if self.training_active:
            return time.time() - self.training.start_time
        return None

    @property
    def hyperparameters_for_state_sync(self) -> Optional[Dict]:
        """Used in sync_confusion_matrix and send_status to provide information about the training configuration.
        """
        if self._training:
            return self._training.hyperparameters
        return None

    @property
    def general_progress(self) -> Optional[float]:
        """Represents the progress for different states, should run from 0 to 100 for each state.
        Note that training_progress and detection_progress need to be implemented in the specific trainer.
        """
        if not self.training_active:
            return None

        t_state = self.training.training_state
        if t_state == TrainerState.DataDownloading:
            return self.node.data_exchanger.progress
        if t_state == TrainerState.TrainingRunning:
            return self.training_progress
        if t_state == TrainerState.Detecting:
            return self.detection_progress

        return None

    # ---------------------------------------- ABSTRACT PROPERTIES ----------------------------------------

    @property
    @abstractmethod
    def training_progress(self) -> Optional[float]:
        """Represents the training progress."""
        raise NotImplementedError

    @property
    @abstractmethod
    def detection_progress(self) -> Optional[float]:
        """Represents the detection progress."""
        raise NotImplementedError

    @property
    @abstractmethod
    def model_architecture(self) -> Optional[str]:
        """Returns the architecture name of the model if available"""
        raise NotImplementedError

    @property
    @abstractmethod
    def provided_pretrained_models(self) -> List[PretrainedModel]:
        """Returns the list of provided pretrained models.
        The names of the models will come back as model_uuid_or_name in the training details.
        """
        raise NotImplementedError

    # ---------------------------------------- METHODS ----------------------------------------

    # NOTE: Trainings are started by the Learning Loop via the begin_training event
        # or by the trainer itself via try_continue_run_if_incomplete.
        # The trainer will then initialize a new training object and start the training loop.
        # Initializing a new training object will create the folder structure for the training.
        # The training loop will then run through the states of the training.

    def generate_status_for_loop(self, trainer_uuid: str, trainer_name: str) -> TrainingStatus:

        status = TrainingStatus(id=trainer_uuid,
                                name=trainer_name,
                                state=self.state,
                                errors={},
                                uptime=self.training_uptime,
                                progress=self.general_progress)

        status.pretrained_models = self.provided_pretrained_models
        status.architecture = self.model_architecture

        if self._training:
            status.errors = self.errors.errors
            status.context = self.training_context

        return status

    async def try_continue_run_if_incomplete(self) -> bool:
        """Tries to continue a training if the last training was not finished.
        """
        if not self.training_active and self.last_training_io.exists():
            self._init_from_last_training()
            logger.info('found incomplete training, continuing now.')
            self._begin_training_task()
            return True
        return False

    def _init_from_last_training(self) -> None:
        """Initializes a new training object from the last training saved on disc via last_training_io.
        """
        self._training = self.last_training_io.load()
        assert self._training is not None and self._training.training_folder is not None, 'could not restore training folder'
        logger.info('restored training: \n%s', self._training)
        self._active_training_io = ActiveTrainingIO(
            self._training.training_folder, self.node.loop_communicator, self._training.context)

    async def begin_training(self, organization: str, project: str, training_config: Dict) -> None:
        """Called on `begin_training` event from the Learning Loop.

        The training slot is taken before anything is set up, so a refused training leaves the
        previous training folders in place.

        :raise TrainingRefusedError: if a training is active or a sibling node holds the training slot.
        """
        if self.training_active:
            raise TrainingRefusedError(f'training {self.training.id} is still active')
        context = Context(organization=organization, project=project)
        self.training_slot.claim(self._slot_claim(context, training_config.get('training_number')))
        try:
            self._init_new_training(context, training_config)
        except Exception:
            self.training_slot.release()
            raise
        self._begin_training_task()

    def _begin_training_task(self) -> None:
        # NOTE: Task object is used to potentially cancel the task
        self.training_task = asyncio.get_event_loop().create_task(self._run())

    def _init_new_training(self, context: Context, training_config: Dict) -> None:
        """Called on `begin_training` event from the Learning Loop.
        Note that training_config needs the entries 'categories', 'model_variant' and 'training_number',
        but also the hyperparameter entries.
        'base_model_uuid' is optional if the training is continued from a previous training.
        """
        project_folder = create_project_folder(context)
        if not self._environment_vars.keep_old_trainings:
            delete_all_training_folders(project_folder)
        self._training = Training.generate_training(project_folder, context, training_config)

        self._active_training_io = ActiveTrainingIO(
            self._training.training_folder, self.node.loop_communicator, context)
        logger.info('new training initialized: %s', self._training)

    async def _run(self) -> None:
        """Called on `begin_training` event from the Learning Loop. 
        Either via `begin_training` or `try_continue_run_if_incomplete`.
        """
        self.errors.reset_all()
        self._state_failures.clear()
        self._state_failed = False
        try:
            await self.training_slot.wait_for(self._slot_claim(self.training.context, self.training.training_number))
            await self._training_loop()
        except asyncio.CancelledError:
            if not self.shutdown_event.is_set():
                logger.info('CancelledError in _run - training task was cancelled but not by shutdown event')
                self.training.training_state = TrainerState.ReadyForCleanup
                self.last_training_io.save(self.training)
                await self._clear_training()
                self._may_restart()
            else:
                logger.info('CancelledError in _run - shutting down')
        except Exception:
            logger.exception('(Ignored) exception in trainer_logic._run:')
        finally:
            self._training = None
            self.training_slot.release()

    def _slot_claim(self, context: Context, training_number: Any) -> str:
        return f'node {self.node.uuid}, training #{training_number} of {context.organization}/{context.project}'

    # ---------------------------------------- TRAINING STATES ----------------------------------------

    async def _training_loop(self) -> None:
        """Cycle through the training states until the training is finished or 
        a critical error occurs (asyncio.CancelledError or CriticalError).
        """
        assert self.training_active

        while self._training is not None:
            tstate = self.training.training_state
            await asyncio.sleep(0.6)  # Note: Required for pytests!

            if tstate == TrainerState.Initialized:  # -> DataDownloading -> DataDownloaded
                await self._perform_state('prepare', TrainerState.DataDownloading, TrainerState.DataDownloaded, self._prepare)
            elif tstate == TrainerState.DataDownloaded:  # -> TrainModelDownloading -> TrainModelDownloaded
                await self._perform_state('download_model', TrainerState.TrainModelDownloading, TrainerState.TrainModelDownloaded, self._download_model)
            elif tstate == TrainerState.TrainModelDownloaded:  # -> TrainingRunning -> TrainingFinished
                await self._perform_state('run_training', TrainerState.TrainingRunning, TrainerState.TrainingFinished, self._train)
            elif tstate == TrainerState.TrainingFinished:  # -> ConfusionMatrixSyncing -> ConfusionMatrixSynced
                await self._perform_state('sync_confusion_matrix', TrainerState.ConfusionMatrixSyncing, TrainerState.ConfusionMatrixSynced, self._sync_training)
            elif tstate == TrainerState.ConfusionMatrixSynced:  # -> TrainModelUploading -> TrainModelUploaded
                await self._perform_state('upload_model', TrainerState.TrainModelUploading, TrainerState.TrainModelUploaded, self._upload_model)
            elif tstate == TrainerState.TrainModelUploaded:  # -> Detecting -> Detected
                await self._perform_state('detecting', TrainerState.Detecting, TrainerState.Detected, self._do_detections)
            elif tstate == TrainerState.Detected:  # -> DetectionUploading -> ReadyForCleanup
                await self._perform_state('upload_detections', TrainerState.DetectionUploading, TrainerState.ReadyForCleanup, self.active_training_io.upload_detetions)
            elif tstate == TrainerState.ReadyForCleanup:  # -> Idle (RESTART or _training = None)
                await self._clear_training()
                self._may_restart()

    async def _perform_state(self, error_key: str, state_during: TrainerState, state_after: TrainerState, action: Callable[[], Coroutine], reset_early=False):
        '''
        Perform a training state and handle errors.
        - If the training is aborted, the CancelledError moves it to ReadyForCleanup; on shutdown it is re-raised.
        - States can raise a CriticalError indicating that there is no point in retrying the state.
        - If any other error occurs, the error is stored in the errors object and the state is reset to the previous state.
          The next attempt waits `STATE_RETRY_DELAY`, twice as long after every further failure up to `MAX_STATE_RETRY_DELAY`,
          so a loop that cannot answer is not asked over and over. The state is retried until it succeeds or the training
          is aborted, so a finished training is not dropped because the loop is down for a while.
          It stays marked as failed until an attempt succeeds, so a stop during its retries aborts.
        '''

        if failures := self._state_failures.get(error_key):
            await asyncio.sleep(_retry_delay(failures))
        await asyncio.sleep(0.1)
        logger.info('Performing state: %s', state_during)
        previous_state = self.training.training_state
        self.training.training_state = state_during
        await asyncio.sleep(0.1)
        if reset_early:
            self.errors.reset(error_key)

        try:
            await action()

        except asyncio.CancelledError:
            if self.shutdown_event.is_set():
                logger.info('CancelledError in %s - shutdown event set', state_during)
                raise
            logger.info('CancelledError in %s - cleaning up', state_during)
            self.training.training_state = TrainerState.ReadyForCleanup
        except CriticalError as e:
            logger.error('CriticalError in %s - Exception: %s', state_during, e)
            self.errors.set(error_key, str(e))
            self.training.training_state = TrainerState.ReadyForCleanup
        except NodeNeedsRestartError:
            logger.error('Node Restart Requested')
            sys.exit(0)
        except Exception as e:
            print('Error in %s - Exception: %s', state_during, e, flush=True)
            self.errors.set(error_key, str(e))
            logger.exception('Error in %s - Exception: %s', state_during, e)
            self._state_failures[error_key] = self._state_failures.get(error_key, 0) + 1
            self.training.training_state = previous_state
            self._state_failed = True
            return
        else:
            logger.info('Successfully finished state: %s', state_during)
            if not reset_early:
                self.errors.reset(error_key)
            self.training.training_state = state_after

        self._state_failures.pop(error_key, None)
        self._state_failed = False
        self.last_training_io.save(self.training)

    async def _prepare(self) -> None:
        """Downloads images to the images_folder and saves annotations to training.data.image_data.
        Moves part of the train images to `val` if `separate_testset` asks for it (see `validation_set`).
        """
        self.node.data_exchanger.set_context(self.training.context)
        downloader = TrainingsDownloader(self.node.data_exchanger)
        image_data, skipped_image_count = await downloader.download_training_data(self.training.images_folder)
        assign_validation_set(image_data, self.training.hyperparameters)

        self.training.image_data = image_data
        self.training.skipped_image_count = skipped_image_count

    async def _download_model(self) -> None:
        """If training is continued, the model is downloaded from the Learning Loop to the training_folder.
        The downloaded model.json file is renamed to base_model.json because a new model.json will be created during training.
        """
        base_model_uuid = self.training.base_model_uuid
        base_model_uuid_is_none = base_model_uuid is None
        base_model_uuid_is_valid = is_valid_uuid4(base_model_uuid)

        if not base_model_uuid_is_none and not base_model_uuid_is_valid:
            logger.warning(
                'base model uuid was provided but was not valid (base_model_uuid: %s).\nSkipping download and starting training from scratch.', base_model_uuid)
            return

        if base_model_uuid_is_none:
            logger.info('No base model provided (base_model_uuid: %s).\nStarting training from scratch.', base_model_uuid)
            return

        logger.info('loading model from Learning Loop')
        logger.info('downloading model %s as %s', base_model_uuid, self.model_format)
        assert base_model_uuid is not None
        await self.node.data_exchanger.download_model(self.training.training_folder, self.training.context, base_model_uuid, self.model_format)
        shutil.move(f'{self.training.training_folder}/model.json',
                    f'{self.training.training_folder}/base_model.json')

    async def _sync_training(self) -> None:
        """Syncronizes the training with the Learning Loop via the update_training endpoint.
        NOTE: This stage sets the errors explicitly because it may be used inside the training stage.
        """
        if not self.node.sio_client or not self.node.sio_client.connected:
            raise ConnectionError('SocketIO client is not connected')

        error_key = 'sync_confusion_matrix'
        try:
            new_best_model = self._get_new_best_training_state()
            if new_best_model:
                # NOTE: the loop has no validation set yet, so the val images split off the train images
                # (see `validation_set`) are reported as train images, which they are in the loop.
                # TODO: rework once the loop offers a dedicated validation set: report a separate
                # `val_image_count` then, which the loop stores and shows next to the train and test counts.
                train_image_count = len(self.training.train_images()) + len(self.training.val_images())
                new_training = TrainingOut(trainer_id=self.node.uuid,
                                           trainer_name=self.node.name,
                                           confusion_matrix=new_best_model.confusion_matrix,
                                           train_image_count=train_image_count,
                                           test_image_count=self.training.test_image_count(),
                                           hyperparameters=self.hyperparameters_for_state_sync,
                                           best_epoch=new_best_model.epoch)
                await asyncio.sleep(0.1)  # NOTE needed for tests.

                result = await self.node.sio_client.call('update_training', (
                    self.training.context.organization, self.training.context.project, jsonable_encoder(new_training)))
                if isinstance(result,  dict) and result['success']:
                    logger.info('successfully updated training %s', asdict(new_training))
                    self._on_metrics_published(new_best_model)
                else:
                    raise Exception(f'Error for update_training: Response from loop was : {result}')
        except Exception as e:
            logger.exception('Error during confusion matrix syncronization')
            self.errors.set(error_key, str(e))
            raise
        self.errors.reset(error_key)

    async def _upload_model(self) -> None:
        """Uploads the latest model to the Learning Loop.
        """
        new_model_uuid = await self._upload_model_return_new_model_uuid(self.training.context)
        logger.info('Successfully uploaded model and received new model id: %s', new_model_uuid)
        self.training.model_uuid_for_detecting = new_model_uuid

    async def _upload_model_return_new_model_uuid(self, context: Context) -> str:
        """Upload model files, usually pytorch model (.pt) hyp.yaml and the converted .wts file.
        Note that with the latest trainers the conversion to (.wts) is done by the trainer.
        The conversion from .wts to .engine is done by the detector (needs to be done on target hardware).
        Note that trainer may train with different classes, which is why we send an initial model.json file.
        A format the loop already stores counts as uploaded.
        The model UUID is kept beside the upload progress, so a retry after a restart still knows it.

        :return: The new model UUID.
        :raise CriticalError: If the latest model files cannot be obtained,
            or the model is uploaded but its UUID is unknown.
        """

        files = await self._get_latest_model_files()
        if files is None or len(files) == 0:
            raise CriticalError('Could not get latest model files. Training might have failed.')

        if isinstance(files, List):
            files = {self.model_format: files}
        assert isinstance(files, Dict), f'can only upload model as list or dict, but was {files}'

        already_uploaded_formats = self.active_training_io.load_model_upload_progress()

        for file_format in [f for f in files if f not in already_uploaded_formats]:
            _files = files[file_format] + [self._dump_model_json()]
            assert len([f for f in _files if 'model.json' in f]) == 1, "model.json must be included exactly once"

            try:
                model_uuid = await self.node.data_exchanger.upload_model_get_uuid(context, _files, self.training.training_number, file_format)
                self.active_training_io.save_uploaded_model_uuid(model_uuid)
            except ModelAlreadyUploadedError:
                logger.warning('The loop already stores the model in format %s, counting it as uploaded', file_format)

            already_uploaded_formats.append(file_format)
            self.active_training_io.save_model_upload_progress(already_uploaded_formats)

        try:
            return self.active_training_io.load_uploaded_model_uuid()
        except FileNotFoundError as e:
            raise CriticalError('The model is uploaded, but the loop never answered with its id') from e

    def _dump_model_json(self) -> str:
        """Dumps the categories and the trainer's model metadata to a json file and returns the path to the file.
        """
        content = {'categories': [asdict(c) for c in self._training.categories],
                   **self._get_model_metadata()} if self._training else None
        json_path = '/tmp/model.json'
        with open(json_path, 'w') as f:
            json.dump(content, f)
        return json_path

    async def _clear_training(self):
        """Clears the training data after a training has finished.
        """
        self.active_training_io.delete_detections()
        self.active_training_io.delete_detection_upload_progress()
        self.active_training_io.delete_detections_upload_file_index()
        await self._clear_training_data(self.training.training_folder)
        self.last_training_io.delete()

        await self.node.send_status()
        self._training = None

    # ---------------------------------------- OTHER METHODS ----------------------------------------

    async def on_shutdown(self) -> None:
        self.shutdown_event.set()
        await self.abort()

    async def stop(self) -> None:
        """Stop the training but keep its model.

        Once the training has run, the states that upload its model and detections are left to finish.
        Before that this base class aborts, as it cannot end a running training; `TrainerLogic` ends its
        training process instead. A state that is failing and being retried would never finish, and a
        shutdown must not wait for the upload, so both abort in any state.
        """
        if not self.training_active:
            return
        if (self.training.training_state in _WIND_DOWN_STATES and not self._state_failed
                and not self.shutdown_event.is_set()):
            logger.info('Stop in state %s: the training has already run, letting its upload finish',
                        self.training.training_state)
            return
        await self.abort()

    async def abort(self):
        """Abort the training: cancel the task immediately and discards in-progress model.
        """
        if not self.training_active:
            return
        if self.training_task:
            logger.info('aborting training task')
            if self.training_task.cancel():
                try:
                    await self.training_task
                except asyncio.CancelledError:
                    pass
                logger.info('aborted training task')
                self._may_restart()

    def _may_restart(self) -> None:
        """If the environment variable RESTART_AFTER_TRAINING is set, the trainer will restart after a training.
        """
        if self._environment_vars.restart_after_training:
            logger.info('restarting')
            sys.exit(0)
        else:
            logger.info('not restarting')
    # ---------------------------------------- ABSTRACT METHODS ----------------------------------------

    @abstractmethod
    async def _train(self) -> None:
        """Should be used to execute a training.
        At this point, images are already downloaded to the images_folder and annotations are saved in training.data.image_data.
        If a training is continued, the model is already downloaded.
        The model should be synchronized with the Learning Loop via self._sync_confusion_matrix() every now and then.
        asyncio.CancelledError should be catched and re-raised.
        """
        raise NotImplementedError

    @abstractmethod
    async def _do_detections(self) -> None:
        """Should be used to infer detections of all images and save them to drive.
        active_training_io.save_detections(...) should be used to store the detections.
        asyncio.CancelledError should be catched and re-raised.
        """
        raise NotImplementedError

    @abstractmethod
    def _get_new_best_training_state(self) -> Optional[TrainingStateData]:
        """Is called frequently by `_sync_training` during training to check if a new "best" model is availabe.
        Returns None if no new model could be found. Otherwise TrainingStateData(confusion_matrix, meta_information).
        `confusion_matrix` contains a dict of all classes:
            - The classes must be identified by their uuid, not their name.
            - For each class a dict with tp, fp, fn is provided (true positives, false positives, false negatives).
        `meta_information` can hold any data which is helpful for self._on_metrics_published to store weight file etc for later upload via self.get_model_files
        `epoch` is the epoch number of the best model.
        """
        raise NotImplementedError

    @abstractmethod
    def _on_metrics_published(self, training_state_data: TrainingStateData) -> None:
        """Called after the metrics corresponding to TrainingStateData have been successfully send to the Learning Loop.
        Receives the TrainingStateData object which was returned by self._get_new_best_training_state. 
        If above function returns None, this function is not called.
        The respective files for this model should be stored so they can be later uploaded in get_latest_model_files.
        """
        raise NotImplementedError

    @abstractmethod
    async def _get_latest_model_files(self) -> Dict[str, List[str]]:
        """Called when the Learning Loop requests to backup the latest model for the training.
        This function is used to __generate and gather__ all files needed for transfering the actual data from the trainer node to the Learning Loop.
        In the simplest implementation this method just renames the weight file (e.g. stored in TrainingStateData.meta_information) into a file name like latest_published_model

        The function should return a list of file paths which describe the model per format.
        These files must contain all data neccessary for the trainer to resume a training (eg. weight file, hyperparameters, etc.)
        and will be stored in the Learning Loop unter the format of this trainer.
        Note: by convention the weightfile should be named "model.<extension>" where extension is the file format of the weightfile.
        For example "model.pt" for pytorch or "model.weights" for darknet/yolo.

        If a trainer can also generate other formats (for example for an detector),
        a dictionary mapping format -> list of files can be returned.

        If the function returns an empty dict, something went wrong and the model upload will be skipped.
        """
        raise NotImplementedError

    def _get_model_metadata(self) -> dict[str, Any]:
        """Called when the files of `_get_latest_model_files` are uploaded.
        Returns values the trainer determined for this model, merged into its model.json.
        The keys must be fields of `ModelInformation` (e.g. `conf_threshold`) so detectors can read them.
        """
        return {}

    @abstractmethod
    async def _clear_training_data(self, training_folder: str) -> None:
        """Called after a training has finished. Deletes all data that is not needed anymore after a training run. 
        This can be old weightfiles or any additional files.
        """
        raise NotImplementedError
