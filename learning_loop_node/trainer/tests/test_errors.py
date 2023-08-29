from learning_loop_node.trainer.tests.states import state_helper
from learning_loop_node.trainer.tests.testing_trainer import TestingTrainer
from learning_loop_node.trainer.trainer_node import TrainerNode
import asyncio
from learning_loop_node.trainer.tests.states.state_helper import assert_training_state
from learning_loop_node.trainer import active_training
import re


async def test_training_process_is_stopped_when_trainer_reports_error(test_trainer_node: TrainerNode):
    state_helper.create_active_training_file(training_state='train_model_downloaded')
    test_trainer_node.trainer = TestingTrainer()
    trainer = test_trainer_node.trainer
    trainer.training = active_training.load()  # normally done by node
    train_task = asyncio.get_running_loop().create_task(trainer.train(
        uuid=test_trainer_node.uuid, sio_client=test_trainer_node.sio_client))

    await assert_training_state(trainer.training, 'training_running', timeout=1, interval=0.001)

    trainer.error_msg = 'some_error'
    await assert_training_state(trainer.training, 'train_model_downloaded', timeout=6, interval=0.001)


async def test_log_can_provide_only_data_for_current_run(test_trainer_node: TrainerNode):
    state_helper.create_active_training_file(training_state='train_model_downloaded')
    test_trainer_node.trainer = TestingTrainer()
    trainer = test_trainer_node.trainer
    trainer.training = active_training.load()  # normally done by node
    train_task = asyncio.get_running_loop().create_task(trainer.train(
        uuid=test_trainer_node.uuid, sio_client=test_trainer_node.sio_client))
    await assert_training_state(trainer.training, 'training_running', timeout=1, interval=0.001)

    assert len(re.findall('Starting executor', str(trainer.executor.get_log_by_lines()))) == 1

    trainer.error_msg = 'some_error'
    await assert_training_state(trainer.training, 'train_model_downloaded', timeout=6, interval=0.001)
    trainer.error_msg = None
    await assert_training_state(trainer.training, 'training_running', timeout=1, interval=0.001)
    await asyncio.sleep(1)

    assert len(re.findall('Starting executor', str(trainer.executor.get_log_by_lines()))) > 1
    # Here only the current run is provided
    assert len(re.findall('Starting executor', str(trainer.executor.get_log_by_lines(since_last_start=True)))) == 1
