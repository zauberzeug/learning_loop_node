from pydantic.main import BaseModel
from ..converter.converter import Converter
from ..status import State
from ..node import Node
from fastapi_utils.tasks import repeat_every
from icecream import ic
from ..loop import loop
import logging
from ..converter.model_information import ModelInformation


class ConverterNode(Node):
    converter: Converter
    skip_check_state: bool = False

    def __init__(self, name: str, uuid: str, converter: Converter):
        super().__init__(name, uuid)
        self.converter = converter

        @self.on_event("startup")
        @repeat_every(seconds=5, raise_exceptions=True, wait_first=True)
        async def check_state():
            if not self.skip_check_state:
                try:
                    await self.check_state()
                except:
                    logging.error('could not check state. Is loop reachable?')

    async def convert_model(self, model_information: ModelInformation):
        try:
            await self.converter.convert(model_information)
            await self.converter.upload_model(model_information.context, model_information.model_id)
        except Exception as e:
            logging.error(
                f'could not convert model {model_information.model_id} for {model_information.context.organization}/{model_information.context.project}. Details: {str(e)}.')

    async def check_state(self):
        logging.debug(f'checking state: {self.status.state}')

        if self.status.state == State.Running:
            return
        self.status.state = State.Running
        try:
            await self.convert_models()
        except Exception as e:
            logging.error(str(e))

        self.status.state = State.Idle

    async def convert_models(self) -> None:

        async with loop.get('api/projects') as response:
            assert response.status == 200, f'Assert statuscode 200, but was {response.status}.'
            content = await response.json()
            projects = content['projects']

        for project in projects:
            organization_id = project['organization_id']
            project_id = project['project_id']

            async with loop.get(f'api{project["resource"]}') as response:
                project_categories = (await response.json())['box_categories']

            path = f'api{project["resource"]}/models'
            async with loop.get(path) as models_response:
                assert models_response.status == 200
                content = await models_response.json()
                models = content['models']

                for model in models:
                    if model['version']:
                        if self.converter.source_format in model['formats'] and not self.converter.target_format in model['formats']:
                            model_information = ModelInformation(
                                organization=organization_id, project=project_id, model_id=model['id'], project_categories=project_categories, version=model['version'])
                            await self.convert_model(model_information)

    async def send_status(self):
        # NOTE not yet implemented
        pass

    async def get_state(self):
        return State.Idle
