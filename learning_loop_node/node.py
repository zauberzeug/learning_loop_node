from fastapi import FastAPI
import socketio
import asyncio
from status import Status, State
import asyncio
import requests


class Node(FastAPI):

    def __init__(self, hostname: str, name: str, uuid: str):
        super().__init__()
        self.hostname = hostname
        self.sio = socketio.AsyncClient(
            reconnection_delay=0,
            request_timeout=0.5,
            # logger=True, engineio_logger=True
        )

        def reset():
            self.status = Status(id=uuid, name=name, state=State.Offline)
        reset()

        @self.on_event("startup")
        async def startup():
            print('startup', flush=True)
            await self.connect()

        @self.on_event("shutdown")
        async def shutdown():
            print('shutting down', flush=True)
            await self.sio.disconnect()

        @self.sio.on('save')
        def on_save(organization, project, model):
            print('---- saving model', model['id'], flush=True)
            if not hasattr(self, '_get_weightfile'):
                return 'node does not provide a get_weightfile function'
            # NOTE: Do not use self.status.organization here. The requested model maybe not belongs to the currently running training.
            uri_base = f'http://{self.hostname}/api/{organization}/projects/{project}'
            data = [('files', self._get_weightfile(organization, project, model['id']))]
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
                return 'node does not provide a begin_training function'

            print(f'---- running training with source model {source_model} for {organization}.{project}', flush=True)
            self.status.model = source_model
            self.status.organization = organization
            self.status.project = project

            uri_base = f'http://{self.hostname}/api/{organization}/projects/{project}'
            data = requests.get(uri_base + '/data?state=complete&mode=boxes').json()

            self._begin_training(data)

            await self.update_state(State.Running)
            return True

        @self.sio.on('stop_training')
        async def stop():
            print('---- stopping', flush=True)
            if hasattr(self, '_stop_training'):
                self._stop_training()
            await self.update_state(State.Idle)
            return True

        @self.sio.on('connect')
        async def on_connect():
            reset()
            await self.update_state(State.Idle)

        @self.sio.on('disconnect')
        async def on_disconnect():
            await self.update_state(State.Offline)

    async def connect(self):
        await self.sio.disconnect()
        print('connecting to Learning Loop', flush=True)

        try:
            await self.sio.connect(f"ws://{self.hostname}", socketio_path="/ws/socket.io")
            print('my sid is', self.sio.sid, flush=True)
        except:
            await asyncio.sleep(0.2)
            await self.connect()
        print('connected to Learning Loop', flush=True)

    def get_weightfile(self, func):
        self._get_weightfile = func

    def begin_training(self, func):
        self._begin_training = func

    def stop_training(self, func):
        self._stop_training = func

    async def update_state(self, state: State):
        self.status.state = state
        if self.status.state != State.Offline:
            await self.send_status()

    async def update_status(self, new_status: Status):
        self.status.id = new_status.id
        self.status.name = new_status.name
        self.status.uptime = new_status.uptime
        self.status.model = new_status.model
        self.status.hyperparameters = new_status.hyperparameters
        self.status.box_categories = new_status.box_categories
        self.status.train_images = new_status.train_images
        self.status.test_images = new_status.test_images

        if self.status.state != State.Offline:
            self.status.state = State.Idle
            await self.send_status()

    async def send_status(self):
        content = self.status.dict()
        if self.status.model:
            content['latest_produced_model_id'] = self.status.model['id']
        del content['model']
        del content['train_images']
        del content['test_images']

        print('sending status', content, flush=True)
        result = await self.sio.call('update_trainer', content)
        if not result == True:
            raise Exception(result)
