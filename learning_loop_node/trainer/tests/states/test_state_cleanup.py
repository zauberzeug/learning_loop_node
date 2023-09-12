from learning_loop_node.trainer.tests.state_helper import \
    create_active_training_file
from learning_loop_node.trainer.tests.testing_trainer_logic import \
    TestingTrainerLogic


async def test_cleanup_successfull(test_initialized_trainer: TestingTrainerLogic):
    trainer = test_initialized_trainer
    create_active_training_file(trainer, training_state='ready_for_cleanup')
    trainer.load_last_training()
    trainer.active_training_io.det_save(detections=[])

    trainer.active_training_io.dup_save(count=42)
    trainer.active_training_io.dufi_save(index=1)

    assert trainer.node.last_training_io.exists() is True
    assert trainer.active_training_io.det_exists() is True
    assert trainer.active_training_io.dup_exists() is True
    assert trainer.active_training_io.dufi_exists() is True

    await trainer.clear_training()

    assert trainer._training is None  # pylint: disable=protected-access
    assert trainer.node.last_training_io.exists() is False
    assert trainer.active_training_io.det_exists() is False
    assert trainer.active_training_io.dup_exists() is False
    assert trainer.active_training_io.dufi_exists() is False
