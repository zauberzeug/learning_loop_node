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


class Trainer():

    def __init__(self, model_format: str) -> None:
        self.model_format: str = model_format
        self.training: Optional[Training] = None
        self.executor: Optional[Executor] = None
        self.start_time: Optional[int] = None
        self.prepare_task = None
        self.download_model_task = None
        self.training_task = None

    def init(self, context: Context, details: dict) -> None:
        try:
            self.training = Trainer.generate_training(context)
            self.training.data = TrainingData(categories=Category.from_list(details['categories']))
            self.training.data.hyperparameter = Hyperparameter.from_dict(details)
            self.training.training_number = details['training_number']
            self.training.base_model_id = details['id']
            self.training.training_state = TrainingState.Initialized
            active_training.save(self.training)
        except:
            logging.exception('Error in init')

    async def prepare(self) -> None:
        if self.training.training_state != 'initialized':
            raise Exception(f"Training must be in state 'initialized', but was '{self.training.training_state}'")
        try:
            self.prepare_task = asyncio.get_running_loop().create_task(self._prepare())
            try:
                await self.prepare_task
            except asyncio.CancelledError:
                logging.info('cancelled prepare task')
                self.training = None
                active_training.delete()
                return False
            except GeneratorExit:
                logging.info('cancelled prepare task')
                self.training = None
                active_training.delete()
                return False
            except Exception:
                logging.exception("Unknown error in 'prepare'")
            finally:
                logging.info('setting prepare_task to None')
                self.prepare_task = None
        except:
            logging.exception('error in prepare training')

        return True

    async def _prepare(self) -> None:
        self.training.training_state = TrainingState.DataDownloading
        downloader = TrainingsDownloader(self.training.context)
        image_data, skipped_image_count = await downloader.download_training_data(self.training.images_folder)
        self.training.data.image_data = image_data
        self.training.data.skipped_image_count = skipped_image_count

        self.training.training_state = TrainingState.DataDownloaded
        active_training.save(self.training)

    async def download_model(self) -> None:
        self.download_model_task = asyncio.get_running_loop().create_task(self._download_model())

        try:
            await self.download_model_task
            logging.info('download_model_task finished')
        except asyncio.CancelledError:
            logging.info('cancelled download model task')
            self.training = None
            active_training.delete()
            return
        finally:
            logging.info(f'setting download_model_task to None')
            self.download_model_task = None

    async def _download_model(self) -> None:
        self.training.training_state = TrainingState.TrainModelDownloading
        model_id = self.training.base_model_id
        if is_valid_uuid4(self.training.base_model_id):
            logging.debug('loading model from Learning Loop')
            logging.info(f'downloading model {model_id} as {self.model_format}')
            try:
                await downloads.download_model(self.training.training_folder, self.training.context, model_id, self.model_format)
                shutil.move(f'{self.training.training_folder}/model.json',
                            f'{self.training.training_folder}/base_model.json')
            except DownloadError:
                logging.exception('DownloadError')
                # TODO what should be done here? Maybe the Model does not exist in the format?

        self.training.training_state = TrainingState.TrainModelDownloaded
        active_training.save(self.training)

    async def run_training(self):
        try:
            self.executor = Executor(self.training.training_folder)
            self.training.training_state = TrainingState.TrainingRunning
            try:
                if self.can_resume():
                    self.training_task = asyncio.get_running_loop().create_task(self.resume())
            except NotImplementedError:
                pass

            if not self.training_task:
                model_id = self.training.base_model_id
                if not is_valid_uuid4(model_id):
                    self.training_task = asyncio.get_running_loop().create_task(self.start_training_from_scratch(model_id))
                else:
                    self.training_task = asyncio.get_running_loop().create_task(self.start_training())

            await self.training_task
            # TODO abort training before executor is started

            while True:
                if not self.executor.is_process_running():
                    # TODO how  / where to check for error?
                    break
                if self.training_task.cancelled():
                    break
                await asyncio.sleep(0.1)
            self.training.training_state = TrainingState.TrainingFinished
            active_training.save(self.training)

        except:
            logging.exception('Error in run_training')

    async def ensure_confusion_matrix_synced(self, trainer_node_uuid: str, sio_client: socketio.AsyncClient):
        try:
            await training_syncronizer.try_sync_model(self, trainer_node_uuid, sio_client)
        except socketio.exceptions.BadNamespaceError:
            logging.error('Error during confusion matrix syncronization. BadNamespaceError')
            return
        except:
            # TODO maybe we should handle {'success:False'} separately?
            logging.exception('Error during confusion matrix syncronization')
            return

        self.training.training_state = TrainingState.ConfusionMatrixSynced
        active_training.save(self.training)

    async def upload_model(self) -> None:
        # TODO is it correct, that this can not be aborted?

        self.training.training_state = TrainingState.TrainModelUploading
        uploaded_model = await self._upload_model(self.training.context)
        # TODO where to catch errors?
        if not uploaded_model:
            raise Exception('Something went wrong while uploading the model')
            # uploaded model can be None - but we need the id for detecting.
        self.training.model_id_for_detecting = uploaded_model['id']
        self.training.training_state = TrainingState.TrainModelUploaded
        active_training.save(self.training)
        return uploaded_model

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

    def can_resume(self) -> bool:
        return False

    async def resume(self) -> None:
        raise NotImplementedError()

    def stop(self) -> None:
        if not self.training:
            return

        if self.training.training_state == TrainingState.DataDownloading:
            if self.prepare_task:
                self.prepare_task.cancel()
        elif self.training.training_state == TrainingState.TrainModelDownloading:
            if self.download_model_task:
                self.download_model_task.cancel()
        elif self.training.training_state == TrainingState.TrainingRunning:
            self.training_task.cancel()
            self.executor.stop()
        else:
            raise NotImplementedError(f'can not stop training: {self.training.training_state}')

    async def start_training(self) -> None:
        raise NotImplementedError()

    async def start_training_from_scratch(self, identifier: str) -> None:
        raise NotImplementedError()

    def stop_training(self) -> bool:
        if self.executor:
            self.executor.stop()
            self.executor = None
            self.start_time = None
            return True
        else:
            logging.info('could not stop training, executor is None')
            return False

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

    async def do_detections(self, context: Context, model_id: str):

        tmp_folder = f'/tmp/model_for_auto_detections_{model_id}_{self.model_format}'

        shutil.rmtree(tmp_folder, ignore_errors=True)
        os.makedirs(tmp_folder)
        logging.info('downloading model for detecting')
        try:
            await downloads.download_model(tmp_folder, context, model_id, self.model_format)
        except:
            logging.exception('download error')
            return
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
        logging.info(f'uploading {len(detections)} detections')
        await self._upload_detections(context, jsonable_encoder(detections))
        return detections

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
                    else:
                        logging.info('successfully uploaded detections')
            except:
                logging.exception('error uploading detections.')

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
