from learning_loop_node.trainer.trainer import Trainer
import asyncio
from learning_loop_node.trainer.tests.states.state_helper import assert_training_file
from learning_loop_node.trainer.tests.states import state_helper
from learning_loop_node.trainer import active_training


async def test_abort_preparing():
    state_helper.create_active_training_file()
    trainer = Trainer(model_format='mocked')
    trainer.training = active_training.load()  # normally done by node

    training_task = asyncio.get_running_loop().create_task(trainer.prepare())

    await asyncio.sleep(0.1)
    assert trainer.training is not None
    assert trainer.training.training_state == 'data_downloading'
    assert trainer.prepare_task is not None
    assert_training_file(exists=True)

    trainer.stop()
    await asyncio.sleep(0.1)

    assert trainer.prepare_task is None
    assert trainer.training == None
    assert_training_file(exists=False)

    training_task.cancel()
