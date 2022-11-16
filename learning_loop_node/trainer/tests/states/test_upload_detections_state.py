from learning_loop_node.trainer.tests.testing_trainer import TestingTrainer
from learning_loop_node.trainer.tests.states import state_helper
from learning_loop_node.trainer.tests.states.state_helper import assert_training_state
from learning_loop_node.trainer import active_training
from learning_loop_node.trainer.trainer import Trainer
from learning_loop_node.trainer.training import Training
from learning_loop_node.context import Context
import asyncio

error_key = 'upload_detections'


def trainer_has_error(trainer: Trainer):
    return trainer.errors.has_error_for(error_key)


def create_detection_file(training: Training):
    active_training.save_detections(training, [{'some_bad_data': 'some_bad_data'}])


async def test_upload_successfull():
    # TODO How can we create detections for uploading?
    # Or should we mock it?
    return
    state_helper.create_active_training_file(training_state='detected')
    trainer = TestingTrainer()
    trainer.training = active_training.load()  # normally done by node
    await create_detection_file(trainer.training)

    await trainer.upload_detections()

    assert trainer.training.training_state == 'ready_for_cleanup'
    assert active_training.load() == trainer.training


async def test_bad_status_from_LearningLoop():
    state_helper.create_active_training_file(training_state='detected', context=Context(
        organization='zauberzeug', project='some_bad_project'))
    trainer = TestingTrainer()
    trainer.training = active_training.load()  # normally done by node
    create_detection_file(trainer.training)

    train_task = asyncio.get_running_loop().create_task(trainer.train(None, None))
    await assert_training_state(trainer.training, 'detection_uploading', timeout=1, interval=0.001)
    await assert_training_state(trainer.training, 'detected', timeout=1, interval=0.001)

    assert trainer_has_error(trainer)
    assert trainer.training.training_state == 'detected'
    assert active_training.load() == trainer.training


async def test_other_errors():
    # e.g. missing detection file
    state_helper.create_active_training_file(training_state='detected')
    trainer = TestingTrainer()
    trainer.training = active_training.load()  # normally done by node

    train_task = asyncio.get_running_loop().create_task(trainer.train(None, None))
    await assert_training_state(trainer.training, 'detection_uploading', timeout=1, interval=0.001)
    await assert_training_state(trainer.training, 'detected', timeout=1, interval=0.001)

    assert trainer_has_error(trainer)
    assert trainer.training.training_state == 'detected'
    assert active_training.load() == trainer.training
