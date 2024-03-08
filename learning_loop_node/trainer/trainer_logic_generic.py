import asyncio
import json
import logging
import shutil
import sys
from abc import abstractmethod
from dataclasses import asdict
from typing import Callable, Coroutine, Dict, List, Optional, Union

from dacite import from_dict
from fastapi.encoders import jsonable_encoder

from ..data_classes import BasicModel, Category, Context, Hyperparameter, TrainerState, TrainingData, TrainingOut
from ..helpers.misc import create_project_folder, delete_all_training_folders, generate_training, is_valid_uuid4
from .downloader import TrainingsDownloader
from .io_helpers import ActiveTrainingIO
from .trainer_logic_abstraction import TrainerLogicAbstraction


class TrainerLogicGeneric(TrainerLogicAbstraction):

    def __init__(self, model_format: str) -> None:
        super().__init__(model_format)
        self.training_task: Optional[asyncio.Task] = None
        self.detection_progress = 0.0
        self.shutdown_event: asyncio.Event = asyncio.Event()

    @property
    def general_progress(self) -> Optional[float]:
        """Represents the progress for different states."""
        if not self.training_active:
            return None

        t_state = self.active_training.training_state
        if t_state == TrainerState.DataDownloading:
            return self.data_exchanger.progress
        if t_state == TrainerState.TrainingRunning:
            return self.training_progress
        if t_state == TrainerState.Detecting:
            return self.detection_progress

        return None

    def init_new_training(self, context: Context, details: Dict) -> None:
        """Called on `begin_training` event from the Learning Loop.
        Note that details needs the entries 'categories' and 'training_number'"""

        project_folder = create_project_folder(context)
        if not self.keep_old_trainings:
            # NOTE: We delete all existing training folders because they are not needed anymore.
            delete_all_training_folders(project_folder)
        self._training = generate_training(project_folder, context)
        self._training.data = TrainingData(categories=Category.from_list(details['categories']))
        self._training.data.hyperparameter = from_dict(data_class=Hyperparameter, data=details)
        self._training.training_number = details['training_number']
        self._training.base_model_id = details['id']
        self._training.training_state = TrainerState.Initialized
        self._active_training_io = ActiveTrainingIO(
            self._training.training_folder, self.loop_communicator, context)
        logging.info(f'training initialized: {self._training}')

    async def try_continue_run_if_incomplete(self) -> bool:
        if not self.training_active and self.last_training_io.exists():
            logging.info('found incomplete training, continuing now.')
            self.init_from_last_training()
            asyncio.get_event_loop().create_task(self.run())
            return True
        return False

    def init_from_last_training(self) -> None:
        self._training = self.last_training_io.load()
        assert self._training is not None and self._training.training_folder is not None, 'could not restore training folder'
        self._active_training_io = ActiveTrainingIO(
            self._training.training_folder, self.loop_communicator, self._training.context)

    async def begin_training(self, organization: str, project: str, details: Dict) -> None:
        """Called on `begin_training` event from the Learning Loop."""

        self.init_new_training(Context(organization=organization, project=project), details)
        asyncio.get_event_loop().create_task(self.run())

    async def run(self) -> None:
        self.errors.reset_all()
        try:
            self.training_task = asyncio.get_running_loop().create_task(self._training_loop())
            await self.training_task  # NOTE: Task object is used to potentially cancel the task
        except asyncio.CancelledError:
            if not self.shutdown_event.is_set():
                logging.info('training task was cancelled but not by shutdown event')
                self.active_training.training_state = TrainerState.ReadyForCleanup
                self.last_training_io.save(self.active_training)
                await self.clear_training()
        except Exception as e:
            logging.exception(f'Error in train: {e}')

    # ---------------------------------------- TRAINING STATES ----------------------------------------

    async def _training_loop(self) -> None:
        """asyncio.CancelledError is catched in run"""

        assert self.training_active

        while self._training is not None:
            tstate = self.active_training.training_state
            logging.info(f'STATE LOOP: {tstate}, eerrors: {self.errors.errors}')
            await asyncio.sleep(0.6)  # Note: Required for pytests!
            if tstate == TrainerState.Initialized:  # -> DataDownloading -> DataDownloaded
                await self.perform_state('prepare', TrainerState.DataDownloading, TrainerState.DataDownloaded, self._prepare)
            elif tstate == TrainerState.DataDownloaded:  # -> TrainModelDownloading -> TrainModelDownloaded
                await self.perform_state('download_model', TrainerState.TrainModelDownloading, TrainerState.TrainModelDownloaded, self._download_model)
            elif tstate == TrainerState.TrainModelDownloaded:  # -> TrainingRunning -> TrainingFinished
                await self.perform_state('run_training', TrainerState.TrainingRunning, TrainerState.TrainingFinished, self._train)
            elif tstate == TrainerState.TrainingFinished:  # -> ConfusionMatrixSyncing -> ConfusionMatrixSynced
                await self.perform_state('sync_confusion_matrix', TrainerState.ConfusionMatrixSyncing, TrainerState.ConfusionMatrixSynced, self._sync_confusion_matrix)
            elif tstate == TrainerState.ConfusionMatrixSynced:  # -> TrainModelUploading -> TrainModelUploaded
                await self.perform_state('upload_model', TrainerState.TrainModelUploading, TrainerState.TrainModelUploaded, self._upload_model)
            elif tstate == TrainerState.TrainModelUploaded:  # -> Detecting -> Detected
                await self.perform_state('detecting', TrainerState.Detecting, TrainerState.Detected, self._do_detections)
            elif tstate == TrainerState.Detected:  # -> DetectionUploading -> ReadyForCleanup
                await self.perform_state('upload_detections', TrainerState.DetectionUploading, TrainerState.ReadyForCleanup, self.active_training_io.upload_detetions)
            elif tstate == TrainerState.ReadyForCleanup:  # -> RESTART or TrainingFinished
                await self.clear_training()
                self.may_restart()

    async def perform_state(self, error_key: str, state_during: TrainerState, state_after: TrainerState, action: Callable[[], Coroutine], reset_early=False):
        await asyncio.sleep(0.1)
        logging.info(f'Performing state: {state_during}')
        previous_state = self.active_training.training_state
        self.active_training.training_state = state_during
        await asyncio.sleep(0.1)
        if reset_early:
            self.errors.reset(error_key)

        try:
            if await action():
                logging.error('Something went really bad.. cleaning up')
                state_after = TrainerState.ReadyForCleanup
        except asyncio.CancelledError:
            logging.warning(f'CancelledError in {state_during}')
            raise
        except Exception as e:
            self.errors.set(error_key, str(e))
            logging.exception(f'Error in {state_during} - Exception:')
            self.active_training.training_state = previous_state
        else:
            if not reset_early:
                self.errors.reset(error_key)
            self.active_training.training_state = state_after
            self.last_training_io.save(self.active_training)

    async def _prepare(self) -> None:
        self.data_exchanger.set_context(self.active_training.context)
        downloader = TrainingsDownloader(self.data_exchanger)
        image_data, skipped_image_count = await downloader.download_training_data(self.active_training.images_folder)
        assert self.active_training.data is not None, 'training.data must be set'
        self.active_training.data.image_data = image_data
        self.active_training.data.skipped_image_count = skipped_image_count

    async def _download_model(self) -> None:
        model_id = self.active_training.base_model_id
        assert model_id is not None, 'model_id must be set'
        if is_valid_uuid4(
                self.active_training.base_model_id):  # TODO this checks if we continue a training -> make more explicit
            logging.info('loading model from Learning Loop')
            logging.info(f'downloading model {model_id} as {self.model_format}')
            await self.data_exchanger.download_model(self.active_training.training_folder, self.active_training.context, model_id, self.model_format)
            shutil.move(f'{self.active_training.training_folder}/model.json',
                        f'{self.active_training.training_folder}/base_model.json')
        else:
            logging.info(f'base_model_id {model_id} is not a valid uuid4, skipping download')

    async def _sync_confusion_matrix(self):
        '''NOTE: This stage sets the errors explicitly because it may be used inside the training stage.'''
        error_key = 'sync_confusion_matrix'
        try:
            new_best_model = self.get_new_best_model()
            if new_best_model and self.active_training.data:
                new_training = TrainingOut(trainer_id=self.node_uuid,
                                           confusion_matrix=new_best_model.confusion_matrix,
                                           train_image_count=self.active_training.data.train_image_count(),
                                           test_image_count=self.active_training.data.test_image_count(),
                                           hyperparameters=self.hyperparameters)
                await asyncio.sleep(0.1)  # NOTE needed for tests.

                result = await self.sio_client.call('update_training', (
                    self.active_training.context.organization, self.active_training.context.project, jsonable_encoder(new_training)))
                if isinstance(result,  dict) and result['success']:
                    logging.info(f'successfully updated training {asdict(new_training)}')
                    self.on_model_published(new_best_model)
                else:
                    raise Exception(f'Error for update_training: Response from loop was : {result}')
        except Exception as e:
            logging.exception('Error during confusion matrix syncronization')
            self.errors.set(error_key, str(e))
            raise
        self.errors.reset(error_key)

    async def _upload_model(self) -> None | bool:
        """Returns True if the training should be cleaned up."""

        new_model_id = await self._upload_model_return_new_model_uuid(self.active_training.context)
        if new_model_id is None:
            self.active_training.training_state = TrainerState.ReadyForCleanup
            logging.error('could not upload model - maybe training failed.. cleaning up')
            return True
        logging.info(f'Successfully uploaded model and received new model id: {new_model_id}')
        self.active_training.model_id_for_detecting = new_model_id
        return None

    async def _upload_model_return_new_model_uuid(self, context: Context) -> Optional[str]:
        """Upload model files, usually pytorch model (.pt) hyp.yaml and the converted .wts file.
        Note that with the latest trainers the conversion to (.wts) is done by the trainer.
        The conversion from .wts to .engine is done by the detector (needs to be done on target hardware).
        Note that trainer may train with different classes, which is why we send an initial model.json file.
        """
        files = await asyncio.get_running_loop().run_in_executor(None, self.get_latest_model_files)
        if files is None:
            return None

        if isinstance(files, List):
            files = {self.model_format: files}
        assert isinstance(files, Dict), f'can only upload model as list or dict, but was {files}'

        already_uploaded_formats = self.active_training_io.load_model_upload_progress()

        model_uuid = None
        for file_format in [f for f in files if f not in already_uploaded_formats]:
            _files = files[file_format] + [self.dump_categories_to_json()]
            assert len([f for f in _files if 'model.json' in f]) == 1, "model.json must be included exactly once"

            model_uuid = await self.data_exchanger.upload_model_get_uuid(context, _files, self.active_training.training_number, file_format)
            if model_uuid is None:
                return None

            already_uploaded_formats.append(file_format)
            self.active_training_io.save_model_upload_progress(already_uploaded_formats)

        return model_uuid

    def dump_categories_to_json(self) -> str:
        content = {'categories': [asdict(c) for c in self.training_data.categories], } if self.training_data else None
        json_path = '/tmp/model.json'
        with open(json_path, 'w') as f:
            json.dump(content, f)
        return json_path

    async def clear_training(self):
        self.active_training_io.delete_detections()
        self.active_training_io.delete_detection_upload_progress()
        self.active_training_io.delete_detections_upload_file_index()
        await self.clear_training_data(self.active_training.training_folder)
        self.last_training_io.delete()
        # self.training.training_state = TrainingState.TrainingFinished

        await self.node.send_status()
        self._training = None

    # ---------------------------------------- OTHER METHODS ----------------------------------------

    def may_restart(self) -> None:
        if self.restart_after_training:
            logging.info('restarting')
            sys.exit(0)
        else:
            logging.info('not restarting')

    async def on_shutdown(self) -> None:
        self.shutdown_event.set()
        await self.stop()
        await self.stop()

    # ---------------------------------------- ABSTRACT PROPERTIES ----------------------------------------

    @property
    @abstractmethod
    def training_progress(self) -> Optional[float]:
        """Represents the training progress."""
        raise NotImplementedError

    # ---------------------------------------- ABSTRACT METHODS ----------------------------------------

    @abstractmethod
    async def _train(self) -> None:
        '''Should be used to execute a training.
        The model should be synchronized with the Learning Loop via self._sync_confusion_matrix() every now and then.
        asyncio.CancelledError should be catched and re-raised.'''

    @abstractmethod
    async def _do_detections(self) -> None:
        '''Should be used to execute detections.
        active_training_io.save_detections(...) should be used to store the detections.
        asyncio.CancelledError should be catched and re-raised.'''

    @abstractmethod
    def get_new_best_model(self) -> Optional[BasicModel]:
        '''Is called frequently in `_sync_confusion_matrix` to check if a new "best" model is availabe.
        Returns None if no new model could be found. Otherwise BasicModel(confusion_matrix, meta_information).
        `confusion_matrix` contains a dict of all classes:
            - The classes must be identified by their id, not their name.
            - For each class a dict with tp, fp, fn is provided (true positives, false positives, false negatives).
        `meta_information` can hold any data which is helpful for self.on_model_published to store weight file etc for later upload via self.get_model_files
        '''

    @abstractmethod
    def on_model_published(self, basic_model: BasicModel) -> None:
        '''Called after a BasicModel has been successfully send to the Learning Loop.
        The files for this model should be stored.
        self.get_latest_model_files is used to gather all files needed for transfering the actual data from the trainer node to the Learning Loop.
        In the simplest implementation this method just renames the weight file (encoded in BasicModel.meta_information) into a file name like latest_published_model
        '''

    @abstractmethod
    def get_latest_model_files(self) -> Optional[Union[List[str], Dict[str, List[str]]]]:
        '''Called when the Learning Loop requests to backup the latest model for the training.
        Should return a list of file paths which describe the model.
        These files must contain all data neccessary for the trainer to resume a training (eg. weight file, hyperparameters, etc.)
        and will be stored in the Learning Loop unter the format of this trainer.
        Note: by convention the weightfile should be named "model.<extension>" where extension is the file format of the weightfile.
        For example "model.pt" for pytorch or "model.weights" for darknet/yolo.

        If a trainer can also generate other formats (for example for an detector),
        a dictionary mapping format -> list of files can be returned.'''

    @abstractmethod
    async def clear_training_data(self, training_folder: str) -> None:
        '''Called after a training has finished. Deletes all data that is not needed anymore after a training run. 
        This can be old weightfiles or any additional files.'''
