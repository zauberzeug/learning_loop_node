from learning_loop_node.trainer.tests.testing_trainer import TestingTrainer
import asyncio
from learning_loop_node.trainer.tests.states.state_helper import assert_training_state
from learning_loop_node.trainer.tests.states import state_helper
from learning_loop_node.trainer import active_training
from learning_loop_node.tests.test_helper import condition
from asyncio import coroutine


async def test_successful_training():
    state_helper.create_active_training_file(training_state='train_model_downloaded')
    trainer = TestingTrainer()
    trainer.training = active_training.load()  # normally done by node

    train_task = asyncio.get_running_loop().create_task(trainer.train(None, None))

    await assert_training_state(trainer.training, 'training_running', timeout=1, interval=0.001)
    assert trainer.train_task.__name__ == 'start_training'

    trainer.executor.stop()  # NOTE normally a training terminates itself e.g
    await assert_training_state(trainer.training, 'training_finished', timeout=1, interval=0.001)

    assert trainer.training.training_state == 'training_finished'
    assert active_training.load() == trainer.training


async def test_stop_running_training():
    state_helper.create_active_training_file(training_state='train_model_downloaded')
    trainer = TestingTrainer()
    trainer.training = active_training.load()  # normally done by node

    train_task = asyncio.get_running_loop().create_task(trainer.train(None, None))

    await condition(lambda: trainer.executor and trainer.executor.is_process_running(), timeout=1, interval=0.01)
    await assert_training_state(trainer.training, 'training_running', timeout=1, interval=0.001)
    assert trainer.train_task.__name__ == 'start_training'

    trainer.stop()
    await assert_training_state(trainer.training, 'training_finished', timeout=1, interval=0.001)

    assert trainer.training.training_state == 'training_finished'
    assert active_training.load() == trainer.training


async def test_training_can_maybe_resumed():
    # NOTE e.g. when a node-computer is restarted
    state_helper.create_active_training_file(training_state='train_model_downloaded')
    trainer = TestingTrainer(can_resume=True)
    trainer.training = active_training.load()  # normally done by node

    train_task = asyncio.get_running_loop().create_task(trainer.train(None, None))

    await condition(lambda: trainer.executor and trainer.executor.is_process_running(), timeout=1, interval=0.01)
    await assert_training_state(trainer.training, 'training_running', timeout=1, interval=0.001)
    assert trainer.train_task.__name__ == 'resume'

    trainer.executor.stop()  # NOTE normally a training terminates itself e.g
    await assert_training_state(trainer.training, 'training_finished', timeout=1, interval=0.001)

    assert trainer.training.training_state == 'training_finished'
    assert active_training.load() == trainer.training
