from learning_loop_node.trainer import Trainer


class TestingTrainer(Trainer):
    __test__ = False

    def __init__(self, ) -> None:
        super().__init__('mocked')

    async def start_training(self) -> None:
        self.executor.start('while true; do sleep 1; done')
