from pydantic.main import BaseModel
import requests
from learning_loop_node.converter.converter import Converter
from learning_loop_node.status import TrainingStatus, State
from learning_loop_node.context import Context
from learning_loop_node.node import Node
import asyncio
from fastapi.encoders import jsonable_encoder
from fastapi_utils.tasks import repeat_every
from icecream import ic


class ModelInformation(BaseModel):
    organization: str
    project: str
    model_id: str


class ConverterNode(Node):
    converter: Converter
    skip_check_state: bool = False

    def __init__(self, name: str, uuid: str, converter: Converter):
        super().__init__(name, uuid)
        self.converter = converter

        @self.on_event("startup")
        @repeat_every(seconds=5, raise_exceptions=True, wait_first=False)
        async def check_state():
            if not self.skip_check_state:
                await self.check_state()

    async def convert_model(self, organization: str, project: str, model_id: str):
        await self.converter.convert(self.url, self.headers, organization, project, model_id)
        await self.converter.upload_model(self.url, self.headers, organization, project, model_id)

    async def check_state(self):
        ic(f'checking state: {self.status.state}')

        if self.status.state == State.Running:
            return
        self.status.state = State.Running

        model = self.find_model_to_convert()
        if model:
            await self.convert_model(model.organization, model.project, model.model_id)
        self.status.state = State.Idle

    def find_model_to_convert(self) -> ModelInformation:
        response = requests.get(f'{self.url}/api/projects', headers=self.headers)
        assert response.status_code == 200

        projects = response.json()['projects']

        for project in projects:
            organization_id = project['organization_id']
            project_id = project['project_id']
            url = f'{self.url}/api{project["resource"]}/models'

            models_response = requests.get(url, headers=self.headers)
            assert models_response.status_code == 200
            models = models_response.json()['models']

            for model in models:
                if model['version']:
                    ic(model)
                    if self.converter.source_format in model['formats'] and not self.converter.target_format in model['formats']:
                        return ModelInformation(organization=organization_id, project=project_id, model_id=model['id'])

    async def send_status(self):
        # NOTE not yet implemented
        pass