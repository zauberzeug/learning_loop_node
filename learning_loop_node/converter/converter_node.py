import logging
import traceback
from http import HTTPStatus
from typing import Optional

from fastapi.encoders import jsonable_encoder
from fastapi_utils.tasks import repeat_every
from socketio import AsyncClient

from learning_loop_node.converter.converter_logic import ConverterLogic
from learning_loop_node.data_classes import ModelInformation
from learning_loop_node.loop_communication import glc
from learning_loop_node.node import Node
from learning_loop_node.status import NodeState


class ConverterNode(Node):
    converter: ConverterLogic
    skip_check_state: bool = False
    bad_model_ids = []

    def __init__(self, name: str, converter: ConverterLogic, uuid: Optional[str] = None):
        super().__init__(name, uuid)
        self.converter = converter

        @self.on_event("startup")
        @repeat_every(seconds=60, raise_exceptions=True, wait_first=False)
        async def check_state():
            if not self.skip_check_state:
                try:
                    await self.check_state()
                except Exception:
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

        if self.status.state == NodeState.Running:
            return
        self.status.state = NodeState.Running
        try:
            await self.convert_models()
        except Exception as exc:
            logging.error(str(exc))

        self.status.state = NodeState.Idle

    async def convert_models(self) -> None:
        try:
            response = await glc.get('/projects')
            assert response.status_code == 200, f'Assert statuscode 200, but was {response.status_code}.'
            content = response.json()
            projects = content['projects']

            for project in projects:
                organization_id = project['organization_id']
                project_id = project['project_id']

                response = await glc.get(f'{project["resource"]}')
                if response.status_code != HTTPStatus.OK:
                    logging.error(
                        f'got bad response for {response.url}: {response.status_code}, {response.content}')
                    continue
                project_categories = response.json()['categories']

                path = f'{project["resource"]}/models'
                models_response = await glc.get(path)
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
                            host=glc.base_url,
                            organization=organization_id,
                            project=project_id,
                            id=model['id'],
                            categories=project_categories,
                            version=model['version'],
                        )
                        await self.convert_model(model_information)
        except Exception as e:

            logging.error(str(e))
            print(traceback.format_exc())

    async def send_status(self):
        # NOTE not yet implemented
        pass

    async def on_startup(self):
        pass

    async def on_shutdown(self):
        pass

    async def on_repeat(self):
        pass

    def register_sio_events(self, sio_client: AsyncClient):
        pass

    async def get_state(self):
        return NodeState.Idle  # NOTE unused for this node type

    def get_node_type(self):
        return 'converter'
