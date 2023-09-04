from learning_loop_node.trainer.tests.states import state_helper
from learning_loop_node.trainer.tests.testing_trainer import TestingTrainer


async def test_cleanup_successfull(test_initialized_trainer: TestingTrainer):
    trainer = test_initialized_trainer

    state_helper.create_active_training_file(training_state='ready_for_cleanup')

    trainer.load_active_training()
    trainer.active_training_io.det_save([])

    trainer.active_training_io.dup_save(42)
    trainer.active_training_io.dufi_save(1)

    assert trainer.last_training_io.exists() is True
    assert trainer.active_training_io.det_exists() is True
    assert trainer.active_training_io.dup_exists() is True
    assert trainer.active_training_io.dufi_exists() is True

    await trainer.clear_training()

    assert trainer._training is None  # pylint: disable=protected-access
    assert trainer.last_training_io.exists() is False
    assert trainer.active_training_io.det_exists() is False
    assert trainer.active_training_io.dup_exists() is False
    assert trainer.active_training_io.dufi_exists() is False
