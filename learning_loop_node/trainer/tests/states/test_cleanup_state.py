from learning_loop_node.trainer.tests.testing_trainer import TestingTrainer
from learning_loop_node.trainer.tests.states import state_helper
from learning_loop_node.trainer import active_training
from learning_loop_node.trainer.training import Training

import os


def create_detection_file(training: Training):
    active_training.save_detections(training, [])


async def test_cleanup_successfull():
    state_helper.create_active_training_file(training_state='ready_for_cleanup')
    trainer = TestingTrainer()
    trainer.training = active_training.load()  # normally done by node
    create_detection_file(trainer.training)

    training_path = trainer.training.training_folder
    detections_path = f'{training_path}/detections.json'

    assert trainer.training is not None
    assert active_training.exists() is True
    assert os.path.exists(detections_path) is True

    await trainer.clear_training()

    assert trainer.training is None
    assert active_training.exists() is False
    assert os.path.exists(detections_path) is False
