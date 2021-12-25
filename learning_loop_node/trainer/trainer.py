import os
from typing import List, Optional, Union
from uuid import uuid4
from learning_loop_node.trainer.executor import Executor
from learning_loop_node.trainer.training import Training
from learning_loop_node.trainer.model import BasicModel
from learning_loop_node.context import Context
from learning_loop_node.node import Node
from learning_loop_node.trainer.downloader import TrainingsDownloader
from learning_loop_node.rest import downloads, uploads
from learning_loop_node import node_helper
import logging
from icecream import ic


class Trainer():

    def __init__(self, model_format: str) -> None:
        self.model_format: str = model_format
        self.training: Optional[Training] = None
        self.executor: Optional[Executor] = None

    async def begin_training(self, context: Context, source_model: dict) -> None:
        downloader = TrainingsDownloader(context)

        self.training = Trainer.generate_training(context, source_model)
        self.training.data = await downloader.download_training_data(self.training.images_folder)
        self.executor = Executor(self.training.training_folder)

        await downloads.download_model(self.training.training_folder, context, source_model['id'], self.model_format)

        await self.start_training()

    async def start_training(self) -> None:
        raise NotImplementedError()

    def stop_training(self) -> None:
        if self.executor:
            self.executor.stop()
        else:
            logging.info('could not stop training -- ')

    def get_error(self) -> Optional[Union[None, str]]:
        '''Should be used to provide error informations to the Learning Loop by extracting data from self.executor.get_log().'''
        pass

    def get_log(self) -> str:
        return self.executor.get_log()

    async def save_model(self,  context: Context, model_id: str) -> None:
        files = self.get_model_files(model_id)
        await uploads.upload_model(context, files, model_id, self.model_format)

    def get_new_model(self) -> Optional[BasicModel]:
        '''Is called frequently to check if a new "best" model is availabe.
        Returns None if no new model could be found. Otherwise BasicModel(confusion_matrix, meta_information).
        confusion_matrix contains a dict of all classes. For each class a dict with tp, fp, fn is provided (true positives, false positives, false negatives).
        meta_information can hold any data which is helpful for self.on_model_published to store weight file etc for later upload via self.get_model_files
        '''
        raise NotImplementedError()

    def on_model_published(self, basic_model: BasicModel, model_id: str) -> None:
        '''Called after a BasicModel has been successfully send to the Learning Loop.
        The model_id is an uuid to identify the model within the Learning Loop.
        self.get_model_files uses this id to gather all files needed for transfering the actual data from the trainer node to the Learning Loop.
        In the simplest implementation this method just renames the weight file (encoded in BasicModel.meta_information) into a file name containing the model_id.
        '''
        raise NotImplementedError()

    def get_model_files(self, model_id) -> List[str]:
        '''Called when the Learning Loop requests to backup a specific model. 
        Returns a list of file paths which describe the model (eg. weight file, hyperparameters, categories etc.)
        '''
        raise NotImplementedError()

    @staticmethod
    def generate_training(context: Context, source_model: dict) -> Training:
        training_uuid = str(uuid4())
        project_folder = Node.create_project_folder(context)
        return Training(
            id=training_uuid,
            base_model=source_model,
            context=context,
            project_folder=project_folder,
            images_folder=node_helper.create_image_folder(project_folder),
            training_folder=Trainer.create_training_folder(project_folder, training_uuid)
        )

    @staticmethod
    def create_training_folder(project_folder: str, trainings_id: str) -> str:
        training_folder = f'{project_folder}/trainings/{trainings_id}'
        os.makedirs(training_folder, exist_ok=True)
        return training_folder
