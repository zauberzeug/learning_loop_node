from pydantic.main import BaseModel
from learning_loop_node.trainer.training_data import TrainingData
from learning_loop_node.node import Node
from learning_loop_node.trainer.training import Training
import requests
import asyncio
from status import State
import os
from uuid import uuid4
from fastapi.encoders import jsonable_encoder
from learning_loop_node.status import Status
from learning_loop_node.status import TrainingStatus


class Trainer(Node):
    training: Training

    def __init__(self, name: str, uuid: str):
        super().__init__(name, uuid)
        self.training = None

        @self.sio.on('begin_training')
        async def on_begin_training(organization, project, source_model):
            if not hasattr(self, '_begin_training'):
                msg = 'node does not provide a begin_training function'
                raise Exception(msg)

            print(f'---- running training with source model {source_model} for {organization}.{project}', flush=True)
            uri_base = f'{self.url}/api/{ organization}/projects/{ project}'

            response = requests.get(uri_base + '/data/data2?state=complete', headers=self.headers)
            assert response.status_code == 200
            data = response.json()

            training_uuid = str(uuid4())
            project_folder = Node.create_project_folder(organization, project)
            self.training = Training(
                id=training_uuid,
                base_model=source_model,
                organization=organization,
                project=project,
                project_folder=project_folder,
                images_folder=Node.create_project_folder(organization, project),
                training_folder=Trainer.create_training_folder(project_folder, training_uuid)
            )
            loop = asyncio.get_event_loop()

            loop.set_debug(True)
            loop.create_task(self._begin_training(data))
            await self.update_state(State.Running)
            return True

        @self.sio.on('stop_training')
        async def stop():
            print('---- stopping', flush=True)
            if hasattr(self, '_stop_training'):
                await self._stop_training()
            await self.update_state(State.Idle)
            return True

        @self.sio.on('save')
        def on_save(organization, project, model):
            print('---- saving model', model['id'], flush=True)
            if not hasattr(self, '_get_model_files'):
                return 'node does not provide a get_model_files function'
            # NOTE: Do not use self.status.organization here. The requested model maybe not belongs to the currently running training.

            uri_base = f'{self.url}/api/{organization}/projects/{project}'
            data = []
            for file_name in self._get_model_files(organization, project, model['id']):
                data.append(('files',  open(file_name, 'rb')))

            response = requests.put(
                f'{uri_base}/models/{model["id"]}/file',
                files=data
            )
            if response.status_code == 200:
                print('---- model saved', flush=True)
                return True
            else:
                print('---- could not save model', flush=True)
                return response.json()['detail']

    async def send_status(self):
        status = TrainingStatus(
            id=self.uuid,
            name=self.name,
            state=self.status.state,
            uptime=self.status.uptime
        )
        if self.training:
            status.latest_produced_model_id = self.training.last_known_model.id

        print('sending status', status, flush=True)
        result = await self.sio.call('update_trainer', jsonable_encoder(status), timeout=1)
        if not result == True:
            raise Exception(result)
        print('status send', flush=True)

    def get_model_files(self, func):
        self._get_model_files = func
        return func

    def begin_training(self, func):
        self._begin_training = func
        return func

    def stop_training(self, func):
        self._stop_training = func
        return func

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
