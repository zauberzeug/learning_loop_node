from learning_loop_node.detector.box_detection import BoxDetection
from learning_loop_node.trainer.tests.testing_trainer import TestingTrainer
from learning_loop_node.trainer.tests.states import state_helper
from learning_loop_node.trainer.tests.states.state_helper import assert_training_state
from learning_loop_node.trainer import active_training
from learning_loop_node.trainer.trainer import Trainer
from learning_loop_node.trainer.training import Training
from learning_loop_node.context import Context
import asyncio
from learning_loop_node import loop
from fastapi.encoders import jsonable_encoder
error_key = 'upload_detections'


def trainer_has_error(trainer: Trainer):
    return trainer.errors.has_error_for(error_key)


async def create_valid_detection_file(training: Training):
    async with loop.get(f'/api/zauberzeug/projects/demo/data') as response:
        assert response.status == 200, response
        content = await response.json()

    category = content['categories'][0]
    image_id = content['image_ids'][0]
    model_version = '1.2'
    box_detections = [BoxDetection(category['name'], x=1, y=2, width=30, height=40,
                                   net=model_version, confidence=.99, category_id=category['id'])]

    image_entry = {'image_id': image_id, 'box_detections': box_detections,
                   'point_detections': [], 'segmentation_detections': []}
    active_training.detections.save(training, jsonable_encoder([image_entry]))


async def test_upload_successful():
    state_helper.create_active_training_file(training_state='detected')
    trainer = TestingTrainer()
    trainer.training = active_training.load()  # normally done by node

    await create_valid_detection_file(trainer.training)

    await trainer.upload_detections()

    assert trainer.training.training_state == 'ready_for_cleanup'
    assert active_training.load() == trainer.training


async def test_detection_upload_progress_is_stored():
    state_helper.create_active_training_file(training_state='detected')
    trainer = TestingTrainer()
    trainer.training = active_training.load()  # normally done by node

    await create_valid_detection_file(trainer.training)

    await trainer.upload_detections()
    assert active_training.detections_upload_progress.load(trainer.training) == 1


async def test_bad_status_from_LearningLoop():
    state_helper.create_active_training_file(training_state='detected', context=Context(
        organization='zauberzeug', project='some_bad_project'))
    trainer = TestingTrainer()
    trainer.training = active_training.load()  # normally done by node
    active_training.detections.save(trainer.training, [{'some_bad_data': 'some_bad_data'}])

    train_task = asyncio.get_running_loop().create_task(trainer.train(None, None))
    await assert_training_state(trainer.training, 'detection_uploading', timeout=1, interval=0.001)
    await assert_training_state(trainer.training, 'detected', timeout=1, interval=0.001)

    assert trainer_has_error(trainer)
    assert trainer.training.training_state == 'detected'
    assert active_training.load() == trainer.training


async def test_other_errors():
    # e.g. missing detection file
    state_helper.create_active_training_file(training_state='detected')
    trainer = TestingTrainer()
    trainer.training = active_training.load()  # normally done by node

    train_task = asyncio.get_running_loop().create_task(trainer.train(None, None))
    await assert_training_state(trainer.training, 'detection_uploading', timeout=1, interval=0.001)
    await assert_training_state(trainer.training, 'detected', timeout=1, interval=0.001)

    assert trainer_has_error(trainer)
    assert trainer.training.training_state == 'detected'
    assert active_training.load() == trainer.training


async def test_abort_uploading():
    state_helper.create_active_training_file(training_state='detected')
    trainer = TestingTrainer()
    trainer.training = active_training.load()  # normally done by node
    await create_valid_detection_file(trainer.training)

    train_task = asyncio.get_running_loop().create_task(trainer.train(None, None))

    await assert_training_state(trainer.training, 'detection_uploading', timeout=1, interval=0.001)

    trainer.stop()
    await asyncio.sleep(0.1)

    assert trainer.training is None
    assert active_training.exists() is False
