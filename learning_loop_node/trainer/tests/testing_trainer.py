from typing import List
from learning_loop_node.trainer import Trainer
from learning_loop_node.trainer.model import PretrainedModel


class TestingTrainer(Trainer):
    __test__ = False

    def __init__(self, ) -> None:
        super().__init__('mocked')

    async def start_training(self) -> None:
        self.executor.start('while true; do sleep 1; done')

    @property
    def provided_pretrained_models(self) -> List[PretrainedModel]:
        return [
            PretrainedModel(name='small', label='Small', description='a small model'),
            PretrainedModel(name='medium', label='Medium', description='a medium model'),
            PretrainedModel(name='large', label='Large', description='a large model')]
