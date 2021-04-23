from learning_loop_node.node import Node
import requests
import asyncio
from status import State


class Trainer(Node):
    def __init__(self, name: str, uuid: str):
        super().__init__(name, uuid)

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
                return True
            else:
                return response.json()['detail']

        @self.sio.on('begin_training')
        async def on_begin_training(organization, project, source_model):
            if not hasattr(self, '_begin_training'):
                msg = 'node does not provide a begin_training function'
                raise Exception(msg)

            print(f'---- running training with source model {source_model} for {organization}.{project}', flush=True)
            self.status.model = source_model
            self.status.organization = organization
            self.status.project = project

            uri_base = f'{self.url}/api/{ self.status.organization}/projects/{ self.status.project}'

            response = requests.get(uri_base + '/data/data2?state=complete', headers=self.headers)
            assert response.status_code == 200
            data = response.json()

            loop = asyncio.get_event_loop()

            loop.set_debug(True)
            loop.create_task(self._begin_training(data))
            await self.update_state(State.Running)
            return True

        @self.sio.on('stop_training')
        async def stop():
            print('---- stopping', flush=True)
            if hasattr(self, '_stop_training'):
                self._stop_training()
            await self.update_state(State.Idle)
            return True
