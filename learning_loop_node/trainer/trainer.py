from abc import abstractmethod
import asyncio
import os
from threading import Thread
from typing import Dict, List, Optional, Union
from uuid import uuid4
from learning_loop_node.rest.downloader import DataDownloader
from learning_loop_node.loop import loop
from tqdm import tqdm
from ..model_information import ModelInformation
from .executor import Executor
from .training import Training
from .model import BasicModel, Model, PretrainedModel
from ..context import Context
from ..node import Node
from .downloader import TrainingsDownloader
from ..rest import downloads, uploads
from .. import node_helper
import logging
from icecream import ic
from .helper import is_valid_uuid4
from glob import glob
import json
from fastapi.encoders import jsonable_encoder
import shutil
from learning_loop_node.data_classes.category import Category
from learning_loop_node.trainer.hyperparameter import Hyperparameter
import time
from time import perf_counter
from learning_loop_node.trainer.training import State as TrainingState
from learning_loop_node.trainer.training_data import TrainingData
from learning_loop_node.trainer import active_training
from learning_loop_node.rest.downloads import DownloadError
from learning_loop_node.trainer import training_syncronizer
import socketio
from aiohttp import ClientResponseError
from asyncio import coroutine


class Errors():
    def __init__(self):
        self._errors: Optional[dict] = {}

    def set(self, key: str, value: str):
        self._errors[key] = value

    def reset(self, key: str):
        try:
            del self._errors[key]
        except AttributeError:
            pass
        except KeyError:
            pass

    def reset_all(self):
        for key in list(self._errors.keys()):
            self.reset_error(key)

    def has_error_for(self, key: str) -> bool:
        return key in self._errors

    def has_error(self) -> bool:
        return self._errors == {}


class Trainer():

    def __init__(self, model_format: str) -> None:
        self.model_format: str = model_format
        self.training: Optional[Training] = None
        self.executor: Optional[Executor] = None
        self.start_time: Optional[int] = None
        self.training_task: Optional[asyncio.Task] = None
        self.train_task: Optional[asyncio.Task] = None
        self.errors = Errors()

    def init(self, context: Context, details: dict) -> None:
        try:
            self.training = Trainer.generate_training(context)
            self.training.data = TrainingData(categories=Category.from_list(details['categories']))
            self.training.data.hyperparameter = Hyperparameter.from_dict(details)
            self.training.training_number = details['training_number']
            self.training.base_model_id = details['id']
            self.training.training_state = TrainingState.Initialized
            active_training.save(self.training)
            logging.info(f'init training: {self.training}')
        except:
            logging.exception('Error in init')

    async def train(self, uuid, sio_client) -> None:
        try:
            self.training_task = asyncio.get_running_loop().create_task(self._train(uuid, sio_client))
            await self.training_task

        except asyncio.CancelledError:
            logging.info('cancelled training task')
            self.training.training_state = TrainingState.ReadyForCleanup
            active_training.save(self.training)
            await self.clear_training()
        except BaseException:
            logging.exception('Error in train')

    async def _train(self, uuid, sio_client) -> None:
        training = None
        if self.training:
            training = self.training
        elif active_training.exists():
            logging.warning('found active training on hd')
            training = active_training.load()
            logging.warning(jsonable_encoder(training))
            self.training = training

        while self.training or active_training.exists():
            if training and training.training_state == TrainingState.Initialized:
                await self.prepare()
            if training and training.training_state == TrainingState.DataDownloaded:
                await self.download_model()
            if training and training.training_state == TrainingState.TrainModelDownloaded:
                await self.run_training()
            if training and training.training_state == TrainingState.TrainingFinished:
                await self.ensure_confusion_matrix_synced(uuid, sio_client)
            if training and training.training_state == TrainingState.ConfusionMatrixSynced:
                await self.upload_model()
            if training and training.training_state == TrainingState.TrainModelUploaded:
                await self.do_detections()
            if training and training.training_state == TrainingState.Detected:
                await self.upload_detections()
            if training and training.training_state == TrainingState.ReadyForCleanup:
                await self.clear_training()
            else:
                await asyncio.sleep(1)

    async def prepare(self) -> None:
        previous_state = self.training.training_state
        self.training.training_state = TrainingState.DataDownloading
        error_key = 'prepare'
        try:
            await self._prepare()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logging.exception("Unknown error in 'prepare'")
            self.training.training_state = previous_state
            self.errors.set(error_key, str(e))
        else:
            self.errors.reset(error_key)
            self.training.training_state = TrainingState.DataDownloaded
            active_training.save(self.training)

    async def _prepare(self) -> None:
        downloader = TrainingsDownloader(self.training.context)
        image_data, skipped_image_count = await downloader.download_training_data(self.training.images_folder)
        self.training.data.image_data = image_data
        self.training.data.skipped_image_count = skipped_image_count

    async def download_model(self) -> None:
        previous_state = self.training.training_state
        self.training.training_state = TrainingState.TrainModelDownloading
        error_key = 'download_model'
        try:
            await self._download_model()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logging.exception('download_model failed')
            self.training.training_state = previous_state
            self.errors.set(error_key, str(e))
        else:
            self.errors.reset(error_key)
            logging.info('download_model_task finished')
            self.training.training_state = TrainingState.TrainModelDownloaded
            active_training.save(self.training)

    async def _download_model(self) -> None:
        model_id = self.training.base_model_id
        if is_valid_uuid4(self.training.base_model_id):
            logging.debug('loading model from Learning Loop')
            logging.info(f'downloading model {model_id} as {self.model_format}')
            try:
                await downloads.download_model(self.training.training_folder, self.training.context, model_id, self.model_format)
                shutil.move(f'{self.training.training_folder}/model.json',
                            f'{self.training.training_folder}/base_model.json')
            except DownloadError:
                # TODO what should be done here? Maybe the Model does not exist in the format?
                raise

    async def run_training(self):
        error_key = 'run_training'
        previous_state = self.training.training_state

        try:
            self.executor = Executor(self.training.training_folder)
            self.training.training_state = TrainingState.TrainingRunning
            self.train_task = None
            try:
                if self.can_resume():
                    self.train_task = self.resume()
            except NotImplementedError:
                pass

            if not self.train_task:
                model_id = self.training.base_model_id
                if not is_valid_uuid4(model_id):
                    self.train_task = self.start_training_from_scratch(model_id)
                else:
                    self.train_task = self.start_training()

            await self.train_task
            # TODO catch normal errors?

            while True:
                if not self.executor.is_process_running():
                    # TODO how  / where to check for error?
                    break
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.errors.set(error_key, f'Could not start training {str(e)}')
            self.training.training_state = previous_state
            logging.exception('Error in run_training')
        else:
            self.errors.reset(error_key)
            self.training.training_state = TrainingState.TrainingFinished
            active_training.save(self.training)

    async def ensure_confusion_matrix_synced(self, trainer_node_uuid: str, sio_client: socketio.AsyncClient):
        previous_state = self.training.training_state
        self.training.training_state = TrainingState.ConfusionMatrixSyncing
        error_key = 'ensure_confusion_matrix_synced'
        try:
            await training_syncronizer.try_sync_model(self, trainer_node_uuid, sio_client)
        except socketio.exceptions.BadNamespaceError as e:
            logging.error('Error during confusion matrix syncronization. BadNamespaceError')

            self.errors.set(error_key, str(e))
            self.training.training_state = previous_state
        except Exception as e:
            # TODO maybe we should handle {'success:False'} separately?
            logging.exception('Error during confusion matrix syncronization')

            self.errors.set(error_key, str(e))
            self.training.training_state = previous_state
        else:
            self.errors.reset(error_key)
            self.training.training_state = TrainingState.ConfusionMatrixSynced
            active_training.save(self.training)

    async def upload_model(self) -> None:
        # TODO is it correct, that this can not be aborted?
        error_key = 'upload_model'
        previous_state = self.training.training_state
        self.training.training_state = TrainingState.TrainModelUploading

        try:
            uploaded_model = await self._upload_model(self.training.context)
            self.training.model_id_for_detecting = uploaded_model['id']
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logging.exception('Error in upload_model')
            self.errors.set(error_key, str(e))
            self.training.training_state = previous_state
        else:
            self.errors.reset(error_key)
            self.training.training_state = TrainingState.TrainModelUploaded
            active_training.save(self.training)

    async def _upload_model(self, context: Context) -> dict:
        files = await asyncio.get_running_loop().run_in_executor(None, self.get_latest_model_files)
        model_json_content = self.create_model_json_content()
        model_json_path = '/tmp/model.json'
        with open(model_json_path, 'w') as f:
            json.dump(model_json_content, f)

        if isinstance(files, list):
            files = {self.model_format: files}
        uploaded_model = None
        if isinstance(files, dict):
            for format in files:
                # model.json was mandatory in previous versions. Now its forbidden to provide an own model.json file.
                assert len([file for file in files[format] if 'model.json' in file]) == 0, \
                    "It is not allowed to provide a 'model.json' file."
                _files = files[format]
                _files.append(model_json_path)
                uploaded_model = await uploads.upload_model_for_training(context, _files, self.training.training_number, format)

        else:
            raise TypeError(f'can only save model as list or dict, but was {files}')
        return uploaded_model

    async def do_detections(self):
        error_key = 'detecting'
        previous_state = self.training.training_state
        try:
            self.training.training_state = TrainingState.Detecting
            detections = await self._do_detections()
            active_training.save_detections(self.training, jsonable_encoder(detections))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.errors.set(error_key, str(e))
            logging.exception('Error in do_detections')
            self.training.training_state = previous_state
        else:
            self.errors.reset(error_key)
            self.training.training_state = TrainingState.Detected
            active_training.save(self.training)

    async def _do_detections(self) -> List:
        context = self.training.context
        model_id = self.training.model_id_for_detecting
        tmp_folder = f'/tmp/model_for_auto_detections_{model_id}_{self.model_format}'

        shutil.rmtree(tmp_folder, ignore_errors=True)
        os.makedirs(tmp_folder)
        logging.info('downloading model for detecting')

        await downloads.download_model(tmp_folder, context, model_id, self.model_format)
        with open(f'{tmp_folder}/model.json', 'r') as f:
            content = json.load(f)
            model_information = ModelInformation.parse_obj(content)

        project_folder = Node.create_project_folder(context)
        image_folder = node_helper.create_image_folder(project_folder)
        downloader = DataDownloader(context)
        image_ids = []
        for state in ['inbox', 'annotate', 'review', 'complete']:
            logging.info(f'fetching image ids of {state}')
            new_ids = await downloader.fetch_image_ids(query_params=f'state={state}')
            image_ids += new_ids
            logging.info(f'downloading {len(new_ids)} images')
            await downloader.download_images(new_ids, image_folder)
        images = await asyncio.get_event_loop().run_in_executor(None, Trainer.images_for_ids, image_ids, image_folder)
        logging.info(f'running detections on {len(images)} images')
        detections = await self._detect(model_information, images, tmp_folder)
        return detections

    async def upload_detections(self):
        error_key = 'upload_detections'
        previous_state = self.training.training_state
        self.training.training_state = TrainingState.DetectionUploading
        context = self.training.context
        await asyncio.sleep(0.1)  # TODO NOTE for tests

        try:
            detections = active_training.load_detections(self.training)
            await self._upload_detections(context, detections)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.errors.set(error_key, str(e))
            logging.exception('Error in upload_detections')
            self.training.training_state = previous_state
        else:
            self.errors.reset(error_key)
            self.training.training_state = TrainingState.ReadyForCleanup
            active_training.save(self.training)

    async def _upload_detections(self, context: Context, detections: List[dict]):
        logging.info('uploading detections')
        batch_size = 500

        for i in tqdm(range(0, len(detections), batch_size), position=0, leave=True):
            batch_detections = detections[i:i+batch_size]
            logging.info(f'uploading detections. File size : {len(json.dumps(batch_detections))}')
            try:
                async with loop.post(f'api/{context.organization}/projects/{context.project}/detections', json=batch_detections) as response:
                    if response.status != 200:
                        logging.error(f'could not upload detections. {str(response)}')
                        # TODO how to handle already uploaded detections?
                        raise Exception(f'could not upload detections. {str(response)}')
                    else:
                        logging.info('successfully uploaded detections')
            except:
                logging.exception('error uploading detections.')
                # TODO where to catch exceptions?
                raise

    async def clear_training(self):
        active_training.delete_detections(self.training)
        try:
            await self.clear_training_data(self.training.training_folder)
        except NotImplementedError:
            logging.exception('clear_training_data not implemented')
            pass
        else:
            self.training = None
            active_training.delete()

    def can_resume(self) -> bool:
        return False

    async def resume(self) -> None:
        raise NotImplementedError()

    def stop(self) -> None:
        if not self.training:
            return
        if self.executor and self.executor.is_process_running():
            self.executor.stop()
        elif self.training_task:
            self.training_task.cancel()

    async def start_training(self) -> None:
        raise NotImplementedError()

    async def start_training_from_scratch(self, identifier: str) -> None:
        raise NotImplementedError()

    def get_error(self) -> Optional[Union[None, str]]:
        '''Should be used to provide error informations to the Learning Loop by extracting data from self.executor.get_log().'''
        pass

    def get_log(self) -> str:
        return self.executor.get_log()

    def get_new_model(self) -> Optional[BasicModel]:
        '''Is called frequently to check if a new "best" model is availabe.
        Returns None if no new model could be found. Otherwise BasicModel(confusion_matrix, meta_information).
        `confusion_matrix` contains a dict of all classes:
            - The classes must be identified by their id, not their name.
            - For each class a dict with tp, fp, fn is provided (true positives, false positives, false negatives).
        `meta_information` can hold any data which is helpful for self.on_model_published to store weight file etc for later upload via self.get_model_files
        '''
        raise NotImplementedError()

    def on_model_published(self, basic_model: BasicModel) -> None:
        '''Called after a BasicModel has been successfully send to the Learning Loop.
        The files for this model should be stored.
        self.get_latest_model_files is used to gather all files needed for transfering the actual data from the trainer node to the Learning Loop.
        In the simplest implementation this method just renames the weight file (encoded in BasicModel.meta_information) into a file name like latest_published_model
        '''
        raise NotImplementedError()

    def get_latest_model_files(self) -> Union[List[str], Dict[str, List[str]]]:
        '''Called when the Learning Loop requests to backup the latest model for the training.
        Should return a list of file paths which describe the model.
        These files must contain all data neccessary for the trainer to resume a training (eg. weight file, hyperparameters, etc.)
        and will be stored in the Learning Loop unter the format of this trainer.
        Note: by convention the weightfile should be named "model.<extension>" where extension is the file format of the weightfile.
        For example "model.pt" for pytorch or "model.weights" for darknet/yolo.

        If a trainer can also generate other formats (for example for an detector),
        a dictionary mapping format -> list of files can be returned.
        '''
        raise NotImplementedError()

    @ staticmethod
    def images_for_ids(image_ids, image_folder) -> List[str]:
        logging.info(f'### Going to get images for {len(image_ids)} images ids')
        start = perf_counter()
        images = [img for img in glob(f'{image_folder}/**/*.*', recursive=True)
                  if os.path.splitext(os.path.basename(img))[0] in image_ids]
        end = perf_counter()
        logging.info(f'found {len(images)} images for {len(image_ids)} image ids, which took {end-start:0.2f} seconds')
        return images

    async def _detect(self, model_information: ModelInformation, images:  List[str], model_folder: str) -> List:
        raise NotImplementedError()

    async def clear_training_data(self, training_folder: str) -> None:
        '''Called after a training has finished. Deletes all data that is not needed anymore after a training run. This can be old
        weightfiles or any additional files.
        '''
        raise NotImplementedError()

    @ property
    @ abstractmethod
    def provided_pretrained_models(self) -> List[PretrainedModel]:
        raise NotImplementedError()

    @ staticmethod
    def generate_training(context: Context) -> Training:
        training_uuid = str(uuid4())
        project_folder = Node.create_project_folder(context)
        return Training(
            id=training_uuid,
            context=context,
            project_folder=project_folder,
            images_folder=node_helper.create_image_folder(project_folder),
            training_folder=Trainer.create_training_folder(project_folder, training_uuid)
        )

    @ staticmethod
    def create_training_folder(project_folder: str, trainings_id: str) -> str:
        training_folder = f'{project_folder}/trainings/{trainings_id}'
        os.makedirs(training_folder, exist_ok=True)
        return training_folder

    @ property
    def hyperparameters(self) -> dict:
        if self.training and self.training.data:
            information = {}
            information['resolution'] = self.training.data.hyperparameter.resolution
            information['flipRl'] = self.training.data.hyperparameter.flip_rl
            information['flipUd'] = self.training.data.hyperparameter.flip_ud
            return information
        else:
            return None

    @ property
    def model_architecture(self) -> Union[str, None]:
        return None

    def create_model_json_content(self):
        content = {
            'categories': [c.dict() for c in self.training.data.categories],
            'resolution': self.training.data.hyperparameter.resolution
        }
        return content
