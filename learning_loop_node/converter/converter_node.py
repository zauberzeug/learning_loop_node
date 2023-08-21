from ..converter.converter import Converter
from ..status import State
from ..node import Node
from fastapi_utils.tasks import repeat_every
from icecream import ic
from ..loop import loop
import logging
from ..model_information import ModelInformation
from http import HTTPStatus
from fastapi.encoders import jsonable_encoder


class ConverterNode(Node):
    converter: Converter
    skip_check_state: bool = False
    bad_model_ids = []

    def __init__(self, name: str, converter: Converter, uuid: str = None):
        super().__init__(name, uuid)
        self.converter = converter

    async def create_sio_client(self):
        await super().create_sio_client()

        @self.on_event("startup")
        @repeat_every(seconds=60, raise_exceptions=True, wait_first=False)
        async def check_state():
            if not self.skip_check_state:
                try:
                    await self.check_state()
                except:
                    logging.error('could not check state. Is loop reachable?')

    async def convert_model(self, model_information: ModelInformation):
        if model_information.id in self.bad_model_ids:
            logging.info(
                f'skipping bad model model {model_information.id} for {model_information.context.organization}/{model_information.context.project}.')
            return
        try:
            logging.info(
                f'converting model {jsonable_encoder(model_information)}')
            await self.converter.convert(model_information)
            logging.info('uploading model ')
            await self.converter.upload_model(model_information.context, model_information.id)
        except Exception as e:
            self.bad_model_ids.append(model_information.id)
            logging.error(
                f'could not convert model {model_information.id} for {model_information.context.organization}/{model_information.context.project}. Details: {str(e)}.')

    async def check_state(self):
        logging.info(f'checking state: {self.status.state}')

        if self.status.state == State.Running:
            return
        self.status.state = State.Running
        try:
            await self.convert_models()
        except Exception as e:
            logging.error(str(e))

        self.status.state = State.Idle

    async def convert_models(self) -> None:
        try:
            response = await loop.get('/projects')
            assert response.status_code == 200, f'Assert statuscode 200, but was {response.status}.'
            content = response.json()
            projects = content['projects']

            for project in projects:
                organization_id = project['organization_id']
                project_id = project['project_id']

                response = await loop.get(f'{project["resource"]}')
                if response.status_code != HTTPStatus.OK:
                    logging.error(
                        f'got bad response for {response.url}: {response.status_code}, {response.content}')
                    continue
                project_categories = response.json()['categories']

                path = f'{project["resource"]}/models'
                models_response = await loop.get(path)
                assert models_response.status_code == 200
                content = models_response.json()
                models = content['models']

                for model in models:
                    if (model['version']
                            and self.converter.source_format in model['formats']
                            and self.converter.target_format not in model['formats']
                        ):
                        # if self.converter.source_format in model['formats'] and project_id == 'drawingbot' and model['version'] == "6.0":
                        model_information = ModelInformation(
                            host=loop.web.base_url,
                            organization=organization_id,
                            project=project_id,
                            id=model['id'],
                            categories=project_categories,
                            version=model['version'],
                        )
                        await self.convert_model(model_information)
        except Exception as e:
            import traceback
            logging.error(str(e))
            print(traceback.format_exc())

    async def send_status(self):
        # NOTE not yet implemented
        pass

    def get_state(self):
        return State.Idle
