import asyncio

from ....data_classes import Context, TrainerState
from ....trainer.trainer_logic import TrainerLogic
from ..state_helper import assert_training_state, create_active_training_file
from ..testing_trainer_logic import TestingTrainerLogic

# pylint: disable=protected-access
error_key = 'prepare'


def trainer_has_error(trainer: TrainerLogic):
    return trainer.errors.has_error_for(error_key)


async def test_preparing_is_successful(test_initialized_trainer: TestingTrainerLogic):
    trainer = test_initialized_trainer
    create_active_training_file(trainer)
    trainer._init_from_last_training()

    await trainer._perform_state('prepare', TrainerState.DataDownloading, TrainerState.DataDownloaded, trainer._prepare)
    assert trainer_has_error(trainer) is False
    assert trainer.training.training_state == TrainerState.DataDownloaded
    assert trainer.training.data is not None
    assert trainer.node.last_training_io.load() == trainer.training


async def test_abort_preparing(test_initialized_trainer: TestingTrainerLogic):
    trainer = test_initialized_trainer
    create_active_training_file(trainer)
    trainer._init_from_last_training()

    _ = asyncio.get_running_loop().create_task(trainer._run())
    await assert_training_state(trainer.training, TrainerState.DataDownloading, timeout=1, interval=0.001)

    await trainer.stop()
    await asyncio.sleep(0.1)

    assert trainer._training is None  # pylint: disable=protected-access
    assert trainer.node.last_training_io.exists() is False


async def test_request_error(test_initialized_trainer: TestingTrainerLogic):
    trainer = test_initialized_trainer
    create_active_training_file(trainer, context=Context(
        organization='zauberzeug', project='some_bad_project'))
    trainer._init_from_last_training()

    _ = asyncio.get_running_loop().create_task(trainer._run())
    await assert_training_state(trainer.training, TrainerState.DataDownloading, timeout=3, interval=0.001)
    await assert_training_state(trainer.training, TrainerState.Initialized, timeout=3, interval=0.001)

    assert trainer_has_error(trainer)
    assert trainer._training is not None  # pylint: disable=protected-access
    assert trainer.training.training_state == TrainerState.Initialized
    assert trainer.node.last_training_io.load() == trainer.training
