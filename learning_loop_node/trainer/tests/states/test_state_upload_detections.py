import asyncio

from fastapi.encoders import jsonable_encoder

from learning_loop_node.data_classes import Context
from learning_loop_node.data_classes.detections import BoxDetection
from learning_loop_node.loop_communication import glc
from learning_loop_node.trainer.tests.states.state_helper import (
    assert_training_state, create_active_training_file)
from learning_loop_node.trainer.tests.testing_trainer import TestingTrainer
from learning_loop_node.trainer.trainer import Trainer

error_key = 'upload_detections'


def trainer_has_error(trainer: Trainer):
    return trainer.errors.has_error_for(error_key)


async def create_valid_detection_file(trainer: Trainer, number_of_entries: int = 1, file_index: int = 0):
    response = await glc.get('/zauberzeug/projects/demo/data')
    assert response.status_code == 200, response
    content = response.json()

    category = content['categories'][0]
    image_id = content['image_ids'][0]
    model_version = '1.2'
    box_detection = BoxDetection(category['name'], x=1, y=2, width=30, height=40,
                                 net=model_version, confidence=.99, category_id=category['id'])
    box_detections = [box_detection]

    image_entry = {'image_id': image_id, 'box_detections': box_detections,
                   'point_detections': [], 'segmentation_detections': []}
    image_entries = [image_entry] * number_of_entries

    assert trainer._active_training_io is not None
    trainer._active_training_io.det_save(jsonable_encoder(image_entries), file_index)


async def test_upload_successful(test_initialized_trainer: TestingTrainer):
    trainer = test_initialized_trainer
    assert trainer._training is not None and trainer._active_training_io is not None

    create_active_training_file(training_state='detected')
    trainer.load_active_training()

    await create_valid_detection_file(trainer)
    await trainer.upload_detections()

    assert trainer._training.training_state == 'ready_for_cleanup'
    assert trainer._active_training_io.load() == trainer._training


async def test_detection_upload_progress_is_stored(test_initialized_trainer: TestingTrainer):
    trainer = test_initialized_trainer
    assert trainer._training is not None and trainer._active_training_io is not None

    create_active_training_file(training_state='detected')
    trainer.load_active_training()

    await create_valid_detection_file(trainer)

    assert trainer._active_training_io.dufi_load() == 0
    await trainer.upload_detections()
    assert trainer._active_training_io.dup_load() == 0  # Progress is reset for every file
    assert trainer._active_training_io.dufi_load() == 1


async def test_ensure_all_detections_are_uploaded(test_initialized_trainer: TestingTrainer):
    trainer = test_initialized_trainer
    assert trainer._training is not None and trainer._active_training_io is not None

    create_active_training_file(training_state='detected')
    trainer.load_active_training()

    await create_valid_detection_file(trainer, 2, 0)
    await create_valid_detection_file(trainer, 2, 1)

    assert trainer._active_training_io.dufi_load() == 0
    detections = trainer._active_training_io.det_load(0)
    assert len(detections) == 2

    batch_size = 1
    skip_detections = trainer._active_training_io.dup_load()
    for i in range(skip_detections, len(detections), batch_size):
        batch_detections = detections[i:i+batch_size]
        # pylint: disable=protected-access
        await trainer._upload_to_learning_loop(trainer._training.context, batch_detections, i + batch_size)

        expected_value = i + batch_size if i + batch_size < len(detections) else 0  # Progress is reset for every file
        assert trainer._active_training_io.dup_load() == expected_value

    assert trainer._active_training_io.dufi_load() == 0
    trainer._active_training_io.dufi_save(1)
    assert trainer._active_training_io.dufi_load() == 1
    assert trainer._active_training_io.dup_load() == 0  # Progress is reset for every file

    detections = trainer._active_training_io.det_load(1)

    skip_detections = trainer._active_training_io.dup_load()
    for i in range(skip_detections, len(detections), batch_size):
        batch_detections = detections[i:i+batch_size]
        # pylint: disable=protected-access
        await trainer._upload_to_learning_loop(trainer._training.context, batch_detections, i + batch_size)

        expected_value = i + batch_size if i + batch_size < len(detections) else 0  # Progress is reset for every file
        assert trainer._active_training_io.dup_load() == expected_value
        assert trainer._active_training_io.dufi_load() == 1


async def test_bad_status_from_LearningLoop(test_initialized_trainer: TestingTrainer):
    trainer = test_initialized_trainer
    assert trainer._training is not None and trainer._active_training_io is not None

    create_active_training_file(training_state='detected', context=Context(
        organization='zauberzeug', project='some_bad_project'))
    trainer.load_active_training()
    trainer._active_training_io.det_save([{'some_bad_data': 'some_bad_data'}])

    _ = asyncio.get_running_loop().create_task(trainer.train(None, None))
    await assert_training_state(trainer._training, 'detection_uploading', timeout=1, interval=0.001)
    await assert_training_state(trainer._training, 'detected', timeout=1, interval=0.001)

    assert trainer_has_error(trainer)
    assert trainer._training.training_state == 'detected'
    assert trainer._active_training_io.load() == trainer._training


async def test_other_errors(test_initialized_trainer: TestingTrainer):
    trainer = test_initialized_trainer
    assert trainer._training is not None and trainer._active_training_io is not None

    # e.g. missing detection file
    create_active_training_file(training_state='detected')
    trainer.load_active_training()

    _ = asyncio.get_running_loop().create_task(trainer.train(None, None))
    await assert_training_state(trainer._training, 'detection_uploading', timeout=1, interval=0.001)
    await assert_training_state(trainer._training, 'detected', timeout=1, interval=0.001)

    assert trainer_has_error(trainer)
    assert trainer._training.training_state == 'detected'
    assert trainer._active_training_io.load() == trainer._training


async def test_abort_uploading(test_initialized_trainer: TestingTrainer):
    trainer = test_initialized_trainer
    assert trainer._training is not None and trainer._active_training_io is not None

    create_active_training_file(training_state='detected')
    trainer.load_active_training()
    await create_valid_detection_file(trainer)

    _ = asyncio.get_running_loop().create_task(trainer.train(None, None))

    await assert_training_state(trainer._training, 'detection_uploading', timeout=1, interval=0.001)

    trainer.stop()
    await asyncio.sleep(0.1)

    assert trainer._training is None
    assert trainer._active_training_io.exists() is False
