from learning_loop_node.trainer.trainer_node import TrainerNode
from learning_loop_node.trainer.tests.testing_trainer import TestingTrainer


def test_fixture_trainer_node(test_trainer_node):
    assert isinstance(test_trainer_node, TrainerNode)
    assert isinstance(test_trainer_node.trainer, TestingTrainer)
