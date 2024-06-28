import asyncio
import time
from typing import Dict, List, Optional

from ...data_classes import Context, Detections, ModelInformation, PretrainedModel, TrainingStateData
from ...trainer.trainer_logic import TrainerLogic


class TestingTrainerLogic(TrainerLogic):
    __test__ = False

    def __init__(self, can_resume: bool = False) -> None:
        super().__init__('mocked')
        self._can_resume_flag: bool = can_resume
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
        return [PretrainedModel(name='small', label='Small', description='a small model'),
                PretrainedModel(name='medium', label='Medium', description='a medium model'),
                PretrainedModel(name='large', label='Large', description='a large model')]

    # pylint: disable=unused-argument
    async def _start_training_from_base_model(self, model: str = 'model.model') -> None:
        assert self._executor is not None
        await self._executor.start('/bin/bash -c "while true; do sleep 1; done"')

    async def _start_training_from_scratch(self) -> None:
        assert self.training.base_model_uuid_or_name is not None, 'base_model_uuid_or_name must be set'
        await self._start_training_from_base_model(model=f'model_{self.training.base_model_uuid_or_name}.pt')

    def _get_new_best_training_state(self) -> Optional[TrainingStateData]:
        if self.has_new_model:
            return TrainingStateData(confusion_matrix={})
        return None

    def _on_metrics_published(self, training_state_data: TrainingStateData) -> None:
        pass

    async def _prepare(self) -> None:
        await super()._prepare()
        await asyncio.sleep(0.1)  # give tests a bit time to to check for the state

    async def _download_model(self) -> None:
        await super()._download_model()
        await asyncio.sleep(0.1)  # give tests a bit time to to check for the state

    async def _upload_model(self) -> None:
        await asyncio.sleep(0.1)  # give tests a bit time to to check for the state
        await super()._upload_model()
        await asyncio.sleep(0.1)  # give tests a bit time to to check for the state

    async def _upload_model_return_new_model_uuid(self, context: Context) -> str:
        await asyncio.sleep(0.1)  # give tests a bit time to to check for the state
        result = await super()._upload_model_return_new_model_uuid(context)
        await asyncio.sleep(0.1)  # give tests a bit time to to check for the state
        assert isinstance(result, str)
        return result

    async def _get_latest_model_files(self) -> Dict[str, List[str]]:
        time.sleep(1)  # NOTE reduce flakyness in Backend tests du to wrong order of events.
        fake_weight_file = '/tmp/weightfile.weights'
        with open(fake_weight_file, 'wb') as f:
            f.write(b'\x42')

        more_data_file = '/tmp/some_more_data.txt'
        with open(more_data_file, 'w') as f:
            f.write('zweiundvierzig')
        return {'mocked': [fake_weight_file, more_data_file], 'mocked_2': [fake_weight_file, more_data_file]}

    def _can_resume(self) -> bool:
        return self._can_resume_flag

    async def _resume(self) -> None:
        return await self._start_training_from_base_model()

    async def _detect(self, model_information: ModelInformation, images:  List[str], model_folder: str) -> List[Detections]:
        detections: List[Detections] = []
        return detections

    async def _clear_training_data(self, training_folder: str) -> None:
        return

    def _get_executor_error_from_log(self) -> Optional[str]:
        return self.error_msg
