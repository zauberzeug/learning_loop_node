import asyncio
import time
from typing import Dict, List, Optional, Union

from learning_loop_node.data_classes import BasicModel, Context, Detections, ModelInformation, PretrainedModel
from learning_loop_node.trainer.trainer_logic import TrainerLogic


class TestingTrainerLogic(TrainerLogic):
    __test__ = False

    def __init__(self, can_resume: bool = False) -> None:
        super().__init__('mocked')
        self._can_resume: bool = can_resume
        self.has_new_model: bool = False
        self.error_msg: Optional[str] = None

    @property
    def training_progress(self) -> float:
        return 1.0

    @property
    def model_architecture(self) -> str:
        return 'mocked'

    @property
    def provided_pretrained_models(self) -> List[PretrainedModel]:
        return [
            PretrainedModel(name='small', label='Small', description='a small model'),
            PretrainedModel(name='medium', label='Medium', description='a medium model'),
            PretrainedModel(name='large', label='Large', description='a large model')]

    # pylint: disable=unused-argument
    async def start_training(self, model: str = 'model.model') -> None:
        assert self._executor is not None
        self._executor.start('while true; do sleep 1; done')

    async def start_training_from_scratch(self, base_model_id: str) -> None:
        await self.start_training(model=f'model_{base_model_id}.pt')

    def get_new_model(self) -> Optional[BasicModel]:
        if self.has_new_model:
            return BasicModel(confusion_matrix={})
        return None

    def on_model_published(self, basic_model: BasicModel) -> None:
        pass

    async def _prepare(self) -> None:
        await super()._prepare()
        await asyncio.sleep(0.1)  # give tests a bit time to to check for the state

    async def _download_model(self) -> None:
        await super()._download_model()
        await asyncio.sleep(0.1)  # give tests a bit time to to check for the state

    async def ensure_confusion_matrix_synced(self):
        await asyncio.sleep(0.1)  # give tests a bit time to to check for the state
        await super().ensure_confusion_matrix_synced()
        await asyncio.sleep(0.1)  # give tests a bit time to to check for the state

    async def upload_model(self) -> None:
        await asyncio.sleep(0.1)  # give tests a bit time to to check for the state
        await super().upload_model()
        await asyncio.sleep(0.1)  # give tests a bit time to to check for the state

    async def _upload_model_return_new_id(self, context: Context) -> Optional[str]:
        await asyncio.sleep(0.1)  # give tests a bit time to to check for the state
        result = await super()._upload_model_return_new_id(context)
        await asyncio.sleep(0.1)  # give tests a bit time to to check for the state
        assert isinstance(result, str)
        return result

    def get_latest_model_files(self) -> Union[List[str], Dict[str, List[str]]]:
        time.sleep(1)  # NOTE reduce flakyness in Backend tests du to wrong order of events.
        fake_weight_file = '/tmp/weightfile.weights'
        with open(fake_weight_file, 'wb') as f:
            f.write(b'\x42')

        more_data_file = '/tmp/some_more_data.txt'
        with open(more_data_file, 'w') as f:
            f.write('zweiundvierzig')
        return {'mocked': [fake_weight_file, more_data_file], 'mocked_2': [fake_weight_file, more_data_file]}

    def can_resume(self) -> bool:
        return self._can_resume

    async def resume(self) -> None:
        return await self.start_training()

    async def _detect(self, model_information: ModelInformation, images:  List[str], model_folder: str) -> List[Detections]:
        detections: List[Detections] = []
        return detections

    async def clear_training_data(self, training_folder: str) -> None:
        return

    def get_executor_error_from_log(self) -> Optional[str]:
        return self.error_msg
