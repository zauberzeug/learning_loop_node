from learning_loop_node.trainer import Trainer


class TestingTrainer(Trainer):
    __test__ = False

    def __init__(self, ) -> None:
        super().__init__('mocked')
