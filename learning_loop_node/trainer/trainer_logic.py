import asyncio
import json
import logging
import os
import shutil
from abc import abstractmethod
from datetime import datetime
from typing import Coroutine, List, Optional

from dacite import from_dict

from ..data_classes import Detections, ModelInformation, TrainerState, TrainingError
from ..helpers.misc import create_image_folder, create_project_folder, images_for_ids, is_valid_uuid4
from .executor import Executor
from .trainer_logic_generic import TrainerLogicGeneric


class TrainerLogic(TrainerLogicGeneric):

    def __init__(self, model_format: str) -> None:
        super().__init__(model_format)
        self.model_format: str = model_format
        # NOTE: String to be used in the file path for the model on the server:
        # '/{context.organization}/projects/{context.project}/models/{model_id}/{model_format}/file'
        self._detection_progress: Optional[float] = None
        self._executor: Optional[Executor] = None
        self.start_training_task: Optional[Coroutine] = None

    @property
    def detection_progress(self) -> Optional[float]:
        return self._detection_progress

    @property
    def executor(self) -> Executor:
        assert self._executor is not None, 'executor must be set, call `run_training` first'
        return self._executor

    async def _train(self) -> None:
        previous_state = TrainerState.TrainModelDownloaded
        error_key = 'run_training'
        self._executor = Executor(self.training.training_folder)
        self.training.training_state = TrainerState.TrainingRunning

        try:
            await self._start_training()

            last_sync_time = datetime.now()
            while True:
                if not self.executor.is_process_running():
                    break
                if (datetime.now() - last_sync_time).total_seconds() > 5:
                    last_sync_time = datetime.now()
                    if self.get_executor_error_from_log():
                        break
                    self.errors.reset(error_key)
                    try:
                        await self._sync_confusion_matrix()
                    except asyncio.CancelledError:
                        logging.warning('CancelledError in run_training')
                        raise
                    except Exception:
                        pass
                else:
                    await asyncio.sleep(0.1)

            error = self.get_executor_error_from_log()
            if error:
                raise TrainingError(cause=error)

            # TODO check if this works to catch errors from the executor:
            # if self.executor.return_code != 0:
            #     self.errors.set(error_key, f'Executor return code was {self.executor.return_code}')
            #     raise TrainingError(cause=f'Executor return code was {self.executor.return_code}')

        except TrainingError:
            logging.exception('Error in TrainingProcess')
            if self.executor.is_process_running():
                self.executor.stop()
            self.training.training_state = previous_state
            raise

    async def _start_training(self):
        self.start_training_task = None  # NOTE: this is used i.e. by tests
        if self.can_resume():
            self.start_training_task = self.resume()
        else:
            base_model_id = self.training.base_model_id
            if not is_valid_uuid4(base_model_id):  # TODO this check was done earlier!
                self.start_training_task = self.start_training_from_scratch()
            else:
                self.start_training_task = self.start_training()
        await self.start_training_task

    async def _do_detections(self) -> None:
        context = self.training.context
        model_id = self.training.model_id_for_detecting
        assert model_id, 'model_id must be set'
        tmp_folder = f'/tmp/model_for_auto_detections_{model_id}_{self.model_format}'

        shutil.rmtree(tmp_folder, ignore_errors=True)
        os.makedirs(tmp_folder)
        logging.info(f'downloading detection model to {tmp_folder}')

        await self.node.data_exchanger.download_model(tmp_folder, context, model_id, self.model_format)
        with open(f'{tmp_folder}/model.json', 'r') as f:
            model_information = from_dict(data_class=ModelInformation, data=json.load(f))

        project_folder = create_project_folder(context)
        image_folder = create_image_folder(project_folder)
        self.node.data_exchanger.set_context(context)
        image_ids = []
        for state, p in zip(['inbox', 'annotate', 'review', 'complete'], [0.1, 0.2, 0.3, 0.4]):
            self._detection_progress = p
            logging.info(f'fetching image ids of {state}')
            new_ids = await self.node.data_exchanger.fetch_image_uuids(query_params=f'state={state}')
            image_ids += new_ids
            logging.info(f'downloading {len(new_ids)} images')
            await self.node.data_exchanger.download_images(new_ids, image_folder)
        self._detection_progress = 0.42
        # await delete_corrupt_images(image_folder)

        images = await asyncio.get_event_loop().run_in_executor(None, images_for_ids, image_ids, image_folder)
        if not images:
            self.active_training_io.save_detections([], 0)
        num_images = len(images)

        batch_size = 200
        for idx, i in enumerate(range(0, num_images, batch_size)):
            self._detection_progress = 0.5 + (i/num_images)*0.5
            batch_images = images[i:i+batch_size]
            batch_detections = await self._detect(model_information, batch_images, tmp_folder)
            self.active_training_io.save_detections(batch_detections, idx)

    async def stop(self) -> None:
        """If executor is running, stop it. Else cancel training task."""
        if not self.training_active:
            return
        if self._executor and self._executor.is_process_running():
            self.executor.stop()
        elif self.training_task:
            logging.info('cancelling training task')
            if self.training_task.cancel():
                try:
                    await self.training_task
                except asyncio.CancelledError:
                    pass
                logging.info('cancelled training task')
                self._may_restart()

    def get_log(self) -> str:
        return self.executor.get_log()

    # ---------------------------------------- ABSTRACT METHODS ----------------------------------------

    @abstractmethod
    async def start_training(self) -> None:
        '''Should be used to start a training on executer, e.g. self.executor.start(cmd).'''

    @abstractmethod
    async def start_training_from_scratch(self) -> None:
        '''Should be used to start a training from scratch on executer, e.g. self.executor.start(cmd).
        NOTE base_model_id is now accessible via self.training.base_model_id 
        the id of a pretrained model provided by self.provided_pretrained_models.'''

    @abstractmethod
    def can_resume(self) -> bool:
        '''Override this method to return True if the trainer can resume training.'''

    @abstractmethod
    async def resume(self) -> None:
        '''Is called when self.can_resume() returns True.
        One may resume the training on a previously trained model stored by self.on_model_published(basic_model).'''

    @abstractmethod
    def get_executor_error_from_log(self) -> Optional[str]:
        '''Should be used to provide error informations to the Learning Loop by extracting data from self.executor.get_log().'''

    @abstractmethod
    async def _detect(self, model_information: ModelInformation, images: List[str], model_folder: str) -> List[Detections]:
        '''Called to run detections on a list of images.'''
