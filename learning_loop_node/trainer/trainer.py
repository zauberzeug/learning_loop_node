from typing import List, Optional
from uuid import uuid4
import os
from pydantic.main import BaseModel
import requests
from learning_loop_node.trainer.downloader import Downloader
from learning_loop_node.trainer.capability import Capability
from learning_loop_node.trainer.training import Training
from learning_loop_node.trainer.model import BasicModel
from learning_loop_node.context import Context
from learning_loop_node.node import Node
from icecream import ic


class Trainer(BaseModel):
    training: Optional[Training]
    capability: Capability
    downloader: Optional[Downloader]

    async def begin_training(self, context: Context, source_model: dict, downloader: Downloader) -> None:
        self.downloader = downloader
        self.training = Trainer.generate_training(context, source_model)
        self.training.data = await self.downloader.download_data(self.training.images_folder, self.training.training_folder, source_model['id'])

        await self.start_training()

    @staticmethod
    def generate_training(context: Context, source_model: dict) -> Training:
        training_uuid = str(uuid4())
        project_folder = Node.create_project_folder(context.organization, context.project)
        return Training(
            id=training_uuid,
            base_model=source_model,
            context=context,
            project_folder=project_folder,
            images_folder=Trainer.create_image_folder(project_folder),
            training_folder=Trainer.create_training_folder(project_folder, training_uuid)
        )

    async def start_training(self) -> None:
        raise NotImplementedError()

    def stop_training(self) -> None:
        raise NotImplementedError()

    def is_training_alive(self) -> bool:
        raise NotImplementedError()

    async def save_model(self, host_url, headers, organization, project, model_id) -> bool:
        # TODO remove the need of host_url. Create an uploader?

        uri_base = f'{host_url}/api/{organization}/projects/{project}'
        data = []
        for file_name in self.get_model_files(model_id):
            data.append(('files',  open(file_name, 'rb')))

        response = requests.put(
            f'{uri_base}/models/{model_id}/file',
            files=data, headers=headers
        )
        if response.status_code != 200:
            msg = f'---- could not save model with id {model_id}'
            raise Exception(msg)
        else:
            ic(f'---- uploaded model with id {model_id}')

    def get_model_files(self, model_id) -> List[str]:
        raise NotImplementedError()

    def get_new_model(self) -> Optional[BasicModel]:
        raise NotImplementedError()

    def on_model_published(self, basic_model: BasicModel, uuid: str) -> None:
        raise NotImplementedError()

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
