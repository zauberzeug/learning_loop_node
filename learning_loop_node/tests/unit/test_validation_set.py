from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ...data_classes import Context, Training, TrainingStateData
from ...trainer import trainer_logic_generic
from ...trainer.exceptions import CriticalError
from ...trainer.trainer_logic_generic import TrainerLogicGeneric
from ...trainer.validation_set import (COMPUTE_VALIDATION_SET, VALIDATION_FRACTION, assign_validation_set,
                                       compute_validation_set)

TRAIN_IMAGE_COUNT = 200


def image(image_id: str, image_set: str) -> dict:
    return {'id': image_id, 'set': image_set}


def image_data() -> list[dict]:
    return [image(f'train_{i}', 'train') for i in range(TRAIN_IMAGE_COUNT)] + [image('test_1', 'test')]


def ids(images: list[dict], image_set: str) -> set[str]:
    return {i['id'] for i in images if i['set'] == image_set}


def test_without_the_hyperparameter_nothing_changes():
    images = image_data()

    assign_validation_set(images, {})

    assert images == image_data()


def test_the_hyperparameter_splits_a_stable_validation_set_off_the_train_images():
    images = image_data()

    assign_validation_set(images, {COMPUTE_VALIDATION_SET: True})

    val_ids = ids(images, 'val')
    assert 0 < len(val_ids) < TRAIN_IMAGE_COUNT * 2 * VALIDATION_FRACTION
    assert ids(images, 'train') | val_ids == ids(image_data(), 'train')
    assert ids(images, 'test') == {'test_1'}

    more_images = [*image_data(), image('train_new', 'train')]
    assign_validation_set(more_images, {COMPUTE_VALIDATION_SET: True})
    assert ids(more_images, 'val') - {'train_new'} == val_ids  # new images do not move existing ones


def test_a_validation_set_of_the_loop_wins_over_the_hyperparameter():
    images = [*image_data(), image('val_1', 'val')]

    assign_validation_set(images, {COMPUTE_VALIDATION_SET: True})

    assert ids(images, 'val') == {'val_1'}
    assert len(ids(images, 'train')) == TRAIN_IMAGE_COUNT


def test_at_least_one_image_is_split_off():
    two = [image('a', 'train'), image('b', 'train')]
    assign_validation_set(two, {COMPUTE_VALIDATION_SET: True})
    assert len(ids(two, 'val')) == 1 and len(ids(two, 'train')) == 1


def test_a_single_train_image_fails_the_training():
    with pytest.raises(CriticalError, match='too few'):
        assign_validation_set([image('a', 'train')], {COMPUTE_VALIDATION_SET: True})


@pytest.mark.parametrize('value, expected', [(None, False), (False, False), (True, True)])
def test_compute_validation_set_accepts_booleans(value, expected):
    hyperparameters = {} if value is None else {COMPUTE_VALIDATION_SET: value}
    assert compute_validation_set(hyperparameters) == expected


@pytest.mark.parametrize('value', ['true', '', 1, None])
def test_compute_validation_set_rejects_anything_but_a_boolean(value):
    with pytest.raises(CriticalError, match=COMPUTE_VALIDATION_SET):
        compute_validation_set({COMPUTE_VALIDATION_SET: value})


def training_with(images: list[dict]) -> Training:
    return Training(id='t', context=Context(organization='o', project='p'), project_folder='', images_folder='',
                    training_folder='', categories=[], hyperparameters={}, training_number=1,
                    training_state='', model_variant='', image_data=images)


def test_training_hands_out_its_images_by_set():
    training = training_with([image('a', 'train'), image('b', 'val'), image('c', 'test')])

    assert [i['id'] for i in training.train_images()] == ['a']
    assert [i['id'] for i in training.val_images()] == ['b']
    assert [i['id'] for i in training.test_images()] == ['c']


def test_without_val_images_the_training_selects_on_the_test_images():
    training = training_with([image('a', 'train'), image('c', 'test')])

    assert [i['id'] for i in training.val_images()] == ['c']


async def prepare(images: list[dict], hyperparameters: dict | None = None) -> Training:
    training = training_with([])
    training.hyperparameters = hyperparameters or {}
    trainer = SimpleNamespace(node=SimpleNamespace(data_exchanger=MagicMock()), training=training)
    downloader = MagicMock(download_training_data=AsyncMock(return_value=(images, 0)))
    with patch.object(trainer_logic_generic, 'TrainingsDownloader', return_value=downloader):
        await TrainerLogicGeneric._prepare(trainer)  # type: ignore[arg-type]
    return training


@pytest.mark.parametrize('images, missing', [
    ([image('a', 'train')], 'test'),
    ([image('a', 'train'), image('b', 'val')], 'test'),
    ([image('c', 'test')], 'train'),
    ([image('b', 'val'), image('c', 'test')], 'train'),
])
async def test_preparing_a_training_without_train_or_test_images_fails(images, missing):
    with pytest.raises(CriticalError, match=f'no {missing} images'):
        await prepare(images)


async def test_preparing_a_training_without_val_images_succeeds():
    training = await prepare(image_data())

    assert training.image_data == image_data()


async def test_preparing_splits_off_the_validation_set_before_the_check():
    training = await prepare([image('a', 'train'), image('b', 'train'), image('c', 'test')],
                             {COMPUTE_VALIDATION_SET: True})

    assert training.image_data is not None
    assert len(ids(training.image_data, 'train')) == 1 and len(ids(training.image_data, 'val')) == 1


async def sync_training(training: Training) -> dict:
    sio_client = SimpleNamespace(connected=True, call=AsyncMock(return_value={'success': True}))
    trainer = SimpleNamespace(node=SimpleNamespace(sio_client=sio_client, uuid='n', name='node'), training=training,
                              errors=MagicMock(), hyperparameters_for_state_sync={}, _on_metrics_published=MagicMock(),
                              _get_new_best_training_state=lambda: TrainingStateData(confusion_matrix={}))

    await TrainerLogicGeneric._sync_training(trainer)  # type: ignore[arg-type]

    return sio_client.call.call_args.args[1][2]


async def test_the_loop_counts_the_val_images_as_train_images():
    sent = await sync_training(training_with([image('a', 'train'), image('b', 'val'), image('c', 'test')]))

    assert (sent['train_image_count'], sent['test_image_count']) == (2, 1)


async def test_the_loop_never_counts_the_test_images_a_training_selects_on_as_train_images():
    sent = await sync_training(training_with([image('a', 'train'), image('c', 'test')]))

    assert (sent['train_image_count'], sent['test_image_count']) == (1, 1)
