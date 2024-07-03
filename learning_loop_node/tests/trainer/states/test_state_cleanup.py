from ..state_helper import create_active_training_file
from ..testing_trainer_logic import TestingTrainerLogic

# pylint: disable=protected-access


async def test_cleanup_successfull(test_initialized_trainer: TestingTrainerLogic):
    trainer = test_initialized_trainer
    create_active_training_file(trainer, training_state='ready_for_cleanup')
    trainer._init_from_last_training()
    trainer.active_training_io.save_detections(detections=[])

    trainer.active_training_io.save_detection_upload_progress(count=42)
    trainer.active_training_io.save_detections_upload_file_index(index=1)

    assert trainer.node.last_training_io.exists() is True
    assert trainer.active_training_io.detections_exist() is True
    assert trainer.active_training_io.detection_upload_progress_exist() is True
    assert trainer.active_training_io.detections_upload_file_index_exists() is True

    await trainer._clear_training()

    assert trainer._training is None
    assert trainer.node.last_training_io.exists() is False
    assert trainer.active_training_io.detections_exist() is False
    assert trainer.active_training_io.detection_upload_progress_exist() is False
    assert trainer.active_training_io.detections_upload_file_index_exists() is False
