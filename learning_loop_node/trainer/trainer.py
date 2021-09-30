from learning_loop_node.trainer.executor import Executor
from learning_loop_node.trainer.downloader_factory import DownloaderFactory
from learning_loop_node import node_helper
from typing import List, Optional
from uuid import uuid4
import os
from learning_loop_node.trainer.training import Training
from learning_loop_node.trainer.model import BasicModel
from learning_loop_node.context import Context
from learning_loop_node.node import Node
from icecream import ic


class Trainer():

    def __init__(self, model_format: str) -> None:
        self.model_format: str = model_format
        self.training: Optional[Training] = None
        self.executor: Optional[Executor] = None

    async def begin_training(self, context: Context, source_model: dict) -> None:
        downloader = DownloaderFactory.create(context)

        self.training = Trainer.generate_training(context, source_model)
        self.training.data = await downloader.download_data(self.training.images_folder)
        self.executor = Executor(self.training.training_folder)

        await node_helper.download_model(self.training.training_folder, context, source_model['id'], self.model_format)

        await self.start_training()

    async def start_training(self) -> None:
        raise NotImplementedError()

    def stop_training(self) -> None:
        raise NotImplementedError()

    def get_error(self) -> str:
        raise NotImplementedError()

    def get_log(self) -> str:
        return self.executor.get_log()

    async def save_model(self,  context: Context, model_id: str) -> None:
        files = self.get_model_files(model_id)
        await node_helper.upload_model(context, files, model_id, self.model_format)

    def get_model_files(self, model_id) -> List[str]:
        raise NotImplementedError()

    def get_new_model(self) -> Optional[BasicModel]:
        raise NotImplementedError()

    def on_model_published(self, basic_model: BasicModel, uuid: str) -> None:
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
            images_folder=Trainer.create_image_folder(project_folder),
            training_folder=Trainer.create_training_folder(project_folder, training_uuid)
        )

    @staticmethod
    def create_image_folder(project_folder: str) -> str:
        image_folder = f'{project_folder}/images'
        os.makedirs(image_folder, exist_ok=True)
        return image_folder

    @staticmethod
    def create_training_folder(project_folder: str, trainings_id: str) -> str:
        training_folder = f'{project_folder}/trainings/{trainings_id}'
        os.makedirs(training_folder, exist_ok=True)
        return training_folder
