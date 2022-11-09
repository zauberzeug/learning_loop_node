
from learning_loop_node.trainer.tests.testing_trainer import TestingTrainer
from learning_loop_node.trainer.trainer import Trainer
import asyncio
from learning_loop_node.trainer.tests.states.state_helper import assert_training_file, assert_training_state
from learning_loop_node.trainer.tests.states import state_helper
from learning_loop_node.trainer import active_training
from learning_loop_node.trainer.training import Training
import os
import logging


def get_coro_name(task: asyncio.Task) -> str:
    # TODO In python 3.8 there is a Tast.get_coro() method, see https://docs.python.org/3/library/asyncio-task.html#asyncio.Task.get_coro
    return str(task).split('coro=<')[1].split('()')[0]


async def test_successful_training():
    def _assert_training_contains_all_data(training: Training) -> None:
        assert trainer.training.training_state == 'training_finished'

    state_helper.create_active_training_file()
    trainer = TestingTrainer()
    trainer.training = active_training.load()  # normally done by node
    await trainer.prepare()
    await trainer.download_model()

    train_task = asyncio.get_running_loop().create_task(trainer.run_training())

    await assert_training_state(trainer.training, 'training_running', timeout=1, interval=0.001)
    assert get_coro_name(trainer.training_task) == 'TestingTrainer.start_training'

    trainer.executor.stop()  # NOTE normally a training terminates itself e.g
    await asyncio.sleep(0.1)

    _assert_training_contains_all_data(trainer.training)
    assert_training_file(exists=True)

    loaded_training = active_training.load()
    _assert_training_contains_all_data(loaded_training)


async def test_training_can_maybe_resumed_can_be_resumed():
    # NOTE e.g. when a node-computer is restarted

    state_helper.create_active_training_file()
    trainer = TestingTrainer(can_resume=True)
    trainer.training = active_training.load()  # normally done by node
    await trainer.prepare()
    await trainer.download_model()

    train_task = asyncio.get_running_loop().create_task(trainer.run_training())
    await asyncio.sleep(0)

    assert get_coro_name(trainer.training_task) == 'TestingTrainer.resume'
