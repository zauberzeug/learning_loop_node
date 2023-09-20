import asyncio

from learning_loop_node.conftest import get_dummy_detections
from learning_loop_node.data_classes import TrainingState
from learning_loop_node.trainer.tests.state_helper import (
    assert_training_state, create_active_training_file)
from learning_loop_node.trainer.tests.testing_trainer_logic import \
    TestingTrainerLogic
from learning_loop_node.trainer.trainer_logic import TrainerLogic

error_key = 'detecting'


def trainer_has_error(trainer: TrainerLogic):
    return trainer.errors.has_error_for(error_key)


async def test_successful_detecting(test_initialized_trainer: TestingTrainerLogic):  # TODO Flaky test
    trainer = test_initialized_trainer
    create_active_training_file(trainer, training_state='train_model_uploaded',
                                model_id_for_detecting='917d5c7f-403d-7e92-f95f-577f79c2273a')
    # trainer.load_active_training()
    _ = asyncio.get_running_loop().create_task(trainer.do_detections())

    await assert_training_state(trainer.training, 'detecting', timeout=1, interval=0.001)
    await assert_training_state(trainer.training, 'detected', timeout=10, interval=0.001)

    assert trainer_has_error(trainer) is False
    assert trainer.training.training_state == 'detected'
    assert trainer.node.last_training_io.load() == trainer.training
    assert trainer.active_training_io.detections_exist()


async def test_detecting_can_be_aborted(test_initialized_trainer: TestingTrainerLogic):
    trainer = test_initialized_trainer
    create_active_training_file(trainer, training_state=TrainingState.TrainModelUploaded)
    trainer.load_last_training()
    trainer.training.model_id_for_detecting = '12345678-bobo-7e92-f95f-424242424242'

    _ = asyncio.get_running_loop().create_task(trainer.run())

    await assert_training_state(trainer.training, 'detecting', timeout=5, interval=0.001)
    await trainer.stop()
    await asyncio.sleep(0.1)

    assert trainer._training is None  # pylint: disable=protected-access
    assert trainer.active_training_io.detections_exist() is False
    assert trainer.node.last_training_io.exists() is False


async def test_model_not_downloadable_error(test_initialized_trainer: TestingTrainerLogic):
    trainer = test_initialized_trainer
    create_active_training_file(trainer, training_state='train_model_uploaded',
                                model_id_for_detecting='00000000-0000-0000-0000-000000000000')  # bad model id
    trainer.load_last_training()

    _ = asyncio.get_running_loop().create_task(trainer.run())

    await assert_training_state(trainer.training, 'detecting', timeout=1, interval=0.001)
    await assert_training_state(trainer.training, 'train_model_uploaded', timeout=1, interval=0.001)

    assert trainer_has_error(trainer)
    assert trainer.training.training_state == 'train_model_uploaded'
    assert trainer.training.model_id_for_detecting == '00000000-0000-0000-0000-000000000000'
    assert trainer.node.last_training_io.load() == trainer.training


def test_save_load_detections(test_initialized_trainer: TestingTrainerLogic):
    trainer = test_initialized_trainer
    create_active_training_file(trainer)
    trainer.load_last_training()

    detections = [get_dummy_detections(), get_dummy_detections()]

    trainer.active_training_io.save_detections(detections)
    assert trainer.active_training_io.detections_exist()

    stored_detections = trainer.active_training_io.load_detections()
    assert stored_detections == detections

    trainer.active_training_io.delete_detections()
    assert trainer.active_training_io.detections_exist() is False
