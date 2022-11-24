from learning_loop_node.trainer.tests.testing_trainer import TestingTrainer
from learning_loop_node.trainer.trainer import Trainer
import asyncio
from learning_loop_node.trainer.tests.states.state_helper import assert_training_state
from learning_loop_node.trainer.tests.states import state_helper
from learning_loop_node.trainer import active_training
from learning_loop_node.context import Context

error_key = 'prepare'


def trainer_has_error(trainer: Trainer):
    return trainer.errors.has_error_for(error_key)


async def test_preparing_is_successful():
    state_helper.create_active_training_file()
    trainer = Trainer(model_format='mocked')
    trainer.training = active_training.load()  # normally done by node

    await trainer.prepare()

    assert trainer_has_error(trainer) == False
    assert trainer.training.training_state == 'data_downloaded'
    assert trainer.training.data is not None
    assert active_training.load() == trainer.training


async def test_abort_preparing():
    state_helper.create_active_training_file()
    trainer = TestingTrainer()
    trainer.training = active_training.load()  # normally done by node

    train_task = asyncio.get_running_loop().create_task(trainer.train(None, None))
    await assert_training_state(trainer.training, 'data_downloading', timeout=1, interval=0.001)

    trainer.stop()
    await asyncio.sleep(0.1)

    assert trainer.training == None
    assert active_training.exists() == False


async def test_request_error():
    state_helper.create_active_training_file(context=Context(
        organization='zauberzeug', project='some_bad_project'))
    trainer = Trainer(model_format='mocked')
    trainer.training = active_training.load()  # normally done by node

    train_task = asyncio.get_running_loop().create_task(trainer.train(None, None))
    await assert_training_state(trainer.training, 'data_downloading', timeout=3, interval=0.001)
    await assert_training_state(trainer.training, 'initialized', timeout=3, interval=0.001)

    assert trainer_has_error(trainer)
    assert trainer.training is not None
    assert trainer.training.training_state == 'initialized'
    assert active_training.load() == trainer.training
