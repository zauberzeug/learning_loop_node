import asyncio

import pytest
from dacite import from_dict

from learning_loop_node.conftest import get_dummy_detections
from learning_loop_node.data_classes import BoxDetection, Context, Detections
from learning_loop_node.loop_communication import LoopCommunicator
from learning_loop_node.trainer.tests.state_helper import (
    assert_training_state, create_active_training_file)
from learning_loop_node.trainer.tests.testing_trainer_logic import \
    TestingTrainerLogic
from learning_loop_node.trainer.trainer_logic import TrainerLogic

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

    assert trainer.active_training_io is not None  # pylint: disable=protected-access
    trainer.active_training_io.save_detections(detections, file_index)


@pytest.mark.asyncio
async def test_upload_successful(test_initialized_trainer: TestingTrainerLogic):
    trainer = test_initialized_trainer
    create_active_training_file(trainer, training_state='detected')
    trainer.load_last_training()

    await create_valid_detection_file(trainer)
    await trainer.upload_detections()

    assert trainer.training.training_state == 'ready_for_cleanup'
    assert trainer.node.last_training_io.load() == trainer.training


@pytest.mark.asyncio
async def test_detection_upload_progress_is_stored(test_initialized_trainer: TestingTrainerLogic):
    trainer = test_initialized_trainer

    create_active_training_file(trainer, training_state='detected')
    trainer.load_last_training()

    await create_valid_detection_file(trainer)

    assert trainer.active_training_io.load_detections_upload_file_index() == 0
    await trainer.upload_detections()
    assert trainer.active_training_io.load_detection_upload_progress() == 0  # Progress is reset for every file
    assert trainer.active_training_io.load_detections_upload_file_index() == 1


@pytest.mark.asyncio
async def test_ensure_all_detections_are_uploaded(test_initialized_trainer: TestingTrainerLogic):
    trainer = test_initialized_trainer

    create_active_training_file(trainer, training_state='detected')
    trainer.load_last_training()

    await create_valid_detection_file(trainer, 2, 0)
    await create_valid_detection_file(trainer, 2, 1)

    assert trainer.active_training_io.load_detections_upload_file_index() == 0
    detections = trainer.active_training_io.load_detections(0)
    assert len(detections) == 2

    batch_size = 1
    skip_detections = trainer.active_training_io.load_detection_upload_progress()
    for i in range(skip_detections, len(detections), batch_size):
        batch_detections = detections[i:i+batch_size]
        # pylint: disable=protected-access
        await trainer._upload_detections(trainer.training.context, batch_detections, i + batch_size)

        expected_value = i + batch_size if i + batch_size < len(detections) else 0  # Progress is reset for every file
        assert trainer.active_training_io.load_detection_upload_progress() == expected_value

    assert trainer.active_training_io.load_detections_upload_file_index() == 0
    trainer.active_training_io.save_detections_upload_file_index(1)
    assert trainer.active_training_io.load_detections_upload_file_index() == 1
    assert trainer.active_training_io.load_detection_upload_progress() == 0  # Progress is reset for every file

    detections = trainer.active_training_io.load_detections(1)

    skip_detections = trainer.active_training_io.load_detection_upload_progress()
    for i in range(skip_detections, len(detections), batch_size):
        batch_detections = detections[i:i+batch_size]
        # pylint: disable=protected-access
        await trainer._upload_detections(trainer.training.context, batch_detections, i + batch_size)

        expected_value = i + batch_size if i + batch_size < len(detections) else 0  # Progress is reset for every file
        assert trainer.active_training_io.load_detection_upload_progress() == expected_value
        assert trainer.active_training_io.load_detections_upload_file_index() == 1


@pytest.mark.asyncio
async def test_bad_status_from_LearningLoop(test_initialized_trainer: TestingTrainerLogic):
    trainer = test_initialized_trainer

    create_active_training_file(trainer, training_state='detected', context=Context(
        organization='zauberzeug', project='some_bad_project'))
    trainer.load_last_training()
    trainer.active_training_io.save_detections([get_dummy_detections()])

    _ = asyncio.get_running_loop().create_task(trainer.run())
    await assert_training_state(trainer.training, 'detection_uploading', timeout=1, interval=0.001)
    await assert_training_state(trainer.training, 'detected', timeout=1, interval=0.001)

    assert trainer_has_error(trainer)
    assert trainer.training.training_state == 'detected'
    assert trainer.node.last_training_io.load() == trainer.training


async def test_other_errors(test_initialized_trainer: TestingTrainerLogic):
    trainer = test_initialized_trainer

    # e.g. missing detection file
    create_active_training_file(trainer, training_state='detected')
    trainer.load_last_training()

    _ = asyncio.get_running_loop().create_task(trainer.run())
    await assert_training_state(trainer.training, 'detection_uploading', timeout=1, interval=0.001)
    await assert_training_state(trainer.training, 'detected', timeout=1, interval=0.001)

    assert trainer_has_error(trainer)
    assert trainer.training.training_state == 'detected'
    assert trainer.node.last_training_io.load() == trainer.training


async def test_abort_uploading(test_initialized_trainer: TestingTrainerLogic):
    trainer = test_initialized_trainer

    create_active_training_file(trainer, training_state='detected')
    trainer.load_last_training()
    await create_valid_detection_file(trainer)

    _ = asyncio.get_running_loop().create_task(trainer.run())

    await assert_training_state(trainer.training, 'detection_uploading', timeout=1, interval=0.001)

    await trainer.stop()
    await asyncio.sleep(0.1)

    assert trainer._training is None  # pylint: disable=protected-access
    assert trainer.node.last_training_io.exists() is False
