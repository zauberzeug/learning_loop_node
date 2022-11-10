from learning_loop_node.trainer.tests.testing_trainer import TestingTrainer
from learning_loop_node.trainer.tests.states import state_helper
from learning_loop_node.trainer import active_training
from learning_loop_node.trainer.training import Training
from learning_loop_node.context import Context


def create_detection_file(training: Training):
    active_training.save_detections(training, [])


async def test_upload_successfull():
    state_helper.create_active_training_file(training_state='some_previous_state')
    trainer = TestingTrainer()
    trainer.training = active_training.load()  # normally done by node
    create_detection_file(trainer.training)

    await trainer.upload_detections()

    assert trainer.training.training_state == 'ready_for_cleanup'
    assert active_training.load() == trainer.training


async def test_bad_status_from_LearningLoop():
    state_helper.create_active_training_file(training_state='some_previous_state', context=Context(
        organization='zauberzeug', project='some_bad_project'))
    trainer = TestingTrainer()
    trainer.training = active_training.load()  # normally done by node

    await trainer.upload_detections()

    assert trainer.training.training_state == 'some_previous_state'
    assert active_training.load() == trainer.training


async def test_other_errors():
    # e.g. missing detection file
    state_helper.create_active_training_file(training_state='some_previous_state')
    trainer = TestingTrainer()
    trainer.training = active_training.load()  # normally done by node

    await trainer.upload_detections()

    assert trainer.training.training_state == 'some_previous_state'
    assert active_training.load() == trainer.training
