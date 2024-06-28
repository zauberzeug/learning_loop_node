import asyncio

from ....data_classes import TrainerState
from ....trainer.trainer_logic import TrainerLogic
from ...test_helper import get_dummy_detections
from ..state_helper import assert_training_state, create_active_training_file
from ..testing_trainer_logic import TestingTrainerLogic

# pylint: disable=protected-access
error_key = 'detecting'


def trainer_has_error(trainer: TrainerLogic):
    return trainer.errors.has_error_for(error_key)


async def test_successful_detecting(test_initialized_trainer: TestingTrainerLogic):
    trainer = test_initialized_trainer
    create_active_training_file(trainer, training_state='train_model_uploaded',
                                model_uuid_for_detecting='00000000-0000-0000-0000-000000000011')  # NOTE: this is the hard coded model uuid for zauberzeug/demo (model version 1.1)

    _ = asyncio.get_running_loop().create_task(
        trainer._perform_state('detecting', TrainerState.Detecting, TrainerState.Detected, trainer._do_detections))

    await assert_training_state(trainer.training, TrainerState.Detecting, timeout=1, interval=0.001)
    await assert_training_state(trainer.training, TrainerState.Detected, timeout=10, interval=0.001)

    assert trainer_has_error(trainer) is False
    assert trainer.training.training_state == TrainerState.Detected
    assert trainer.node.last_training_io.load() == trainer.training
    assert trainer.active_training_io.detections_exist()


async def test_detecting_can_be_aborted(test_initialized_trainer: TestingTrainerLogic):
    trainer = test_initialized_trainer
    create_active_training_file(trainer, training_state=TrainerState.TrainModelUploaded)
    trainer._init_from_last_training()
    trainer.training.model_uuid_for_detecting = '12345678-bobo-7e92-f95f-424242424242'

    _ = asyncio.get_running_loop().create_task(trainer._run())

    await assert_training_state(trainer.training, TrainerState.Detecting, timeout=5, interval=0.001)
    await trainer.stop()
    await asyncio.sleep(0.1)

    assert trainer._training is None
    assert trainer.active_training_io.detections_exist() is False
    assert trainer.node.last_training_io.exists() is False


async def test_model_not_downloadable_error(test_initialized_trainer: TestingTrainerLogic):
    trainer = test_initialized_trainer
    create_active_training_file(trainer, training_state=TrainerState.TrainModelUploaded,
                                model_uuid_for_detecting='00000000-0000-0000-0000-000000000000')  # bad model id
    trainer._init_from_last_training()

    _ = asyncio.get_running_loop().create_task(trainer._run())

    await assert_training_state(trainer.training, 'detecting', timeout=1, interval=0.001)
    await assert_training_state(trainer.training, 'train_model_uploaded', timeout=1, interval=0.001)
    await asyncio.sleep(0.1)

    assert trainer_has_error(trainer)
    assert trainer.training.training_state == TrainerState.TrainModelUploaded
    assert trainer.training.model_uuid_for_detecting == '00000000-0000-0000-0000-000000000000'
    assert trainer.node.last_training_io.load() == trainer.training


def test_save_load_detections(test_initialized_trainer: TestingTrainerLogic):
    trainer = test_initialized_trainer
    create_active_training_file(trainer)
    trainer._init_from_last_training()

    detections = [get_dummy_detections(), get_dummy_detections()]

    trainer.active_training_io.save_detections(detections)
    assert trainer.active_training_io.detections_exist()

    stored_detections = trainer.active_training_io.load_detections()
    assert stored_detections == detections

    trainer.active_training_io.delete_detections()
    assert trainer.active_training_io.detections_exist() is False
