import asyncio

import pytest
from dacite import from_dict

from ....data_classes import BoxDetection, Context, Detections, TrainerState
from ....loop_communication import LoopCommunicator
from ....trainer.trainer_logic import TrainerLogic
from ...test_helper import get_dummy_detections
from ..state_helper import assert_training_state, create_active_training_file
from ..testing_trainer_logic import TestingTrainerLogic

# pylint: disable=protected-access
error_key = 'upload_detections'


def trainer_has_error(trainer: TrainerLogic):
    return trainer.errors.has_error_for(error_key)


async def create_valid_detection_file(trainer: TrainerLogic, number_of_entries: int = 1, file_index: int = 0):
    loop_communicator = LoopCommunicator()
    response = await loop_communicator.get('/zauberzeug/projects/demo/data')
    await loop_communicator.shutdown()

    assert response.status_code == 200, response
    content = response.json()

    category = content['categories'][0]
    image_id = content['image_ids'][0]
    model_version = '1.2'
    box_detection = BoxDetection(category_name=category['name'], x=1, y=2, width=30, height=40,
                                 model_name=model_version, confidence=.99, category_id=category['id'])
    box_detections = [box_detection]

    detection_entry = from_dict(data_class=Detections, data={'image_id': image_id, 'box_detections': box_detections,
                                                             'point_detections': [], 'segmentation_detections': []})
    detections = [detection_entry] * number_of_entries

    assert trainer.active_training_io is not None
    trainer.active_training_io.save_detections(detections, file_index)


@pytest.mark.asyncio
async def test_upload_successful(test_initialized_trainer: TestingTrainerLogic):
    trainer = test_initialized_trainer
    create_active_training_file(trainer, training_state=TrainerState.Detected)
    trainer._init_from_last_training()

    await create_valid_detection_file(trainer)
    await asyncio.get_running_loop().create_task(
        trainer._perform_state('upload_detections', TrainerState.DetectionUploading, TrainerState.ReadyForCleanup, trainer.active_training_io.upload_detetions))

    assert trainer.training.training_state == TrainerState.ReadyForCleanup
    assert trainer.node.last_training_io.load() == trainer.training


@pytest.mark.asyncio
async def test_detection_upload_progress_is_stored(test_initialized_trainer: TestingTrainerLogic):
    trainer = test_initialized_trainer

    create_active_training_file(trainer, training_state=TrainerState.Detected)
    trainer._init_from_last_training()

    await create_valid_detection_file(trainer)

    assert trainer.active_training_io.load_detections_upload_file_index() == 0
    # await trainer.upload_detections()
    await asyncio.get_running_loop().create_task(
        trainer._perform_state('upload_detections', TrainerState.DetectionUploading, TrainerState.ReadyForCleanup, trainer.active_training_io.upload_detetions))

    assert trainer.active_training_io.load_detection_upload_progress() == 0  # Progress is reset for every file
    assert trainer.active_training_io.load_detections_upload_file_index() == 1


@pytest.mark.asyncio
async def test_ensure_all_detections_are_uploaded(test_initialized_trainer: TestingTrainerLogic):
    trainer = test_initialized_trainer

    create_active_training_file(trainer, training_state=TrainerState.Detected)
    trainer._init_from_last_training()

    await create_valid_detection_file(trainer, 4, 0)
    await create_valid_detection_file(trainer, 4, 1)

    assert trainer.active_training_io.load_detections_upload_file_index() == 0
    detections = trainer.active_training_io.load_detections(0)
    assert len(detections) == 4

    batch_size = 2
    skip_detections = trainer.active_training_io.load_detection_upload_progress()
    for i in range(skip_detections, len(detections), batch_size):
        batch_detections = detections[i:i+batch_size]
        progress = i + batch_size if i + batch_size < len(detections) else 0
        await trainer.active_training_io._upload_detections_and_save_progress(trainer.training.context, batch_detections, progress)

        expected_value = progress  # Progress is reset for every file
        assert trainer.active_training_io.load_detection_upload_progress() == expected_value

    assert trainer.active_training_io.load_detections_upload_file_index() == 0
    trainer.active_training_io.save_detections_upload_file_index(1)
    assert trainer.active_training_io.load_detections_upload_file_index() == 1
    assert trainer.active_training_io.load_detection_upload_progress() == 0  # Progress is reset for every file

    detections = trainer.active_training_io.load_detections(1)

    skip_detections = trainer.active_training_io.load_detection_upload_progress()
    for i in range(skip_detections, len(detections), batch_size):
        batch_detections = detections[i:i+batch_size]

        progress = i + batch_size if i + batch_size < len(detections) else 0
        await trainer.active_training_io._upload_detections_and_save_progress(trainer.training.context, batch_detections, progress)

        expected_value = progress  # Progress is reset for every file
        assert trainer.active_training_io.load_detection_upload_progress() == expected_value
        assert trainer.active_training_io.load_detections_upload_file_index() == 1


@pytest.mark.asyncio
async def test_bad_status_from_LearningLoop(test_initialized_trainer: TestingTrainerLogic):
    trainer = test_initialized_trainer

    create_active_training_file(trainer, training_state=TrainerState.Detected, context=Context(
        organization='zauberzeug', project='some_bad_project'))
    trainer._init_from_last_training()
    trainer.active_training_io.save_detections([get_dummy_detections()])

    _ = asyncio.get_running_loop().create_task(trainer._run())
    await assert_training_state(trainer.training, TrainerState.DetectionUploading, timeout=1, interval=0.001)
    await assert_training_state(trainer.training, TrainerState.Detected, timeout=1, interval=0.001)

    assert trainer_has_error(trainer)
    assert trainer.training.training_state == TrainerState.Detected
    assert trainer.node.last_training_io.load() == trainer.training


async def test_go_to_cleanup_if_no_detections_exist(test_initialized_trainer: TestingTrainerLogic):
    """This test simulates a situation where the detection file is missing.
    In this case, the trainer should report an error and move to the ReadyForCleanup state."""
    trainer = test_initialized_trainer

    # e.g. missing detection file
    create_active_training_file(trainer, training_state=TrainerState.Detected)
    trainer._init_from_last_training()

    _ = asyncio.get_running_loop().create_task(trainer._run())
    await assert_training_state(trainer.training, TrainerState.ReadyForCleanup, timeout=1, interval=0.001)


async def test_abort_uploading(test_initialized_trainer: TestingTrainerLogic):
    trainer = test_initialized_trainer

    create_active_training_file(trainer, training_state=TrainerState.Detected)
    trainer._init_from_last_training()
    await create_valid_detection_file(trainer)

    _ = asyncio.get_running_loop().create_task(trainer._run())

    await assert_training_state(trainer.training, TrainerState.DetectionUploading, timeout=1, interval=0.001)

    await trainer.stop()
    await asyncio.sleep(0.1)

    assert trainer._training is None
    assert trainer.node.last_training_io.exists() is False
