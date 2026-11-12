from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from dacite import from_dict

from ...data_classes import Context, Training, TrainingStateData
from ...trainer import trainer_logic_generic
from ...trainer.downloader import TrainingsDownloader
from ...trainer.exceptions import CriticalError
from ...trainer.subsets import restrict_to_subsets
from ...trainer.trainer_logic_generic import TrainerLogicGeneric


def subset(name: str) -> dict[str, str]:
    return {'uuid': f'uuid-{name}', 'name': name}


SUBSETS = {'train': [subset('winter')], 'test': [subset('hard')]}


def image(image_id: str, image_set: str, *subsets: str) -> dict:
    return {'id': image_id, 'set': image_set,
            'subsets': [{'id': f'uuid-{name}', 'name': name, 'color': '#139c84'} for name in subsets]}


def image_data() -> list[dict]:
    return [image('a', 'train', 'winter'), image('b', 'train', 'summer'), image('c', 'train', 'winter', 'hard'),
            image('v', 'val'), image('t', 'test', 'hard'), image('u', 'test')]


def ids(images: list[dict]) -> list[str]:
    return [i['id'] for i in images]


def test_a_restricted_set_keeps_the_images_in_its_subset():
    assert ids(restrict_to_subsets(image_data(), SUBSETS)) == ['a', 'c', 'v', 't']


def test_a_set_restricted_to_several_subsets_keeps_the_images_in_any_of_them():
    assert ids(restrict_to_subsets(image_data(), {'train': [subset('winter'), subset('summer')]})) == \
        ['a', 'b', 'c', 'v', 't', 'u']


def test_an_image_matches_its_subset_by_uuid_after_a_rename():
    renamed = [{**image('a', 'train', 'winter'), 'subsets': [{'id': 'uuid-winter', 'name': 'winter 2026'}]}]

    assert ids(restrict_to_subsets(renamed, {'train': [subset('winter')]})) == ['a']


def test_without_subsets_all_images_stay():
    assert restrict_to_subsets(image_data(), {}) == image_data()


def test_an_image_without_subsets_is_in_none():
    assert not restrict_to_subsets([{'id': 'a', 'set': 'train'}], {'train': [subset('winter')]})


def generate(tmp_path: Path, **config) -> Training:
    data = {'hyperparameters': {}, 'categories': [], 'training_number': 1, 'model_variant': 'small', **config}
    return Training.generate_training(str(tmp_path), Context(organization='o', project='p'), data)


def test_a_training_takes_the_subsets_of_its_config(tmp_path: Path):
    assert generate(tmp_path, subsets=SUBSETS).subsets == SUBSETS


def test_a_config_without_subsets_restricts_nothing(tmp_path: Path):
    assert generate(tmp_path).subsets == {}


@pytest.mark.parametrize('subsets', [[subset('winter')], {'prepare': [subset('winter')]}, {'train': 'winter'},
                                     {'train': ['winter']}, {'train': [{'uuid': 'uuid-winter'}]}])
def test_malformed_subsets_are_rejected(tmp_path: Path, subsets):
    with pytest.raises(ValueError, match='subsets'):
        generate(tmp_path, subsets=subsets)


def test_a_training_saved_without_subsets_loads_unrestricted(tmp_path: Path):
    saved = asdict(generate(tmp_path, subsets=SUBSETS))
    del saved['subsets']

    assert from_dict(data_class=Training, data=saved).subsets == {}


async def test_the_downloader_skips_the_images_outside_the_subsets(tmp_path: Path):
    downloaded: list[str] = []

    async def download_images(image_ids: list[str], image_folder: str) -> None:
        downloaded.extend(image_ids)
        for image_id in image_ids:
            (Path(image_folder) / f'{image_id}.jpg').touch()

    data_exchanger = MagicMock(fetch_image_uuids=AsyncMock(return_value=ids(image_data())),
                               download_images_data=AsyncMock(return_value=image_data()),
                               download_images=AsyncMock(side_effect=download_images))
    images, skipped = await TrainingsDownloader(data_exchanger, subsets=SUBSETS).download_training_data(str(tmp_path))

    assert ids(images) == downloaded == ['a', 'c', 'v', 't']
    assert skipped == 0


def restricted_training(subsets: dict[str, list[dict[str, str]]]) -> Training:
    return Training(id='t', context=Context(organization='o', project='p'), project_folder='', images_folder='',
                    training_folder='', categories=[], hyperparameters={}, training_number=1,
                    training_state='', model_variant='', subsets=subsets)


async def prepare(images: list[dict], subsets: dict[str, list[dict[str, str]]]) -> MagicMock:
    """Prepare a training restricted to ``subsets`` whose download yields ``images``; return the downloader class."""
    trainer = SimpleNamespace(node=SimpleNamespace(data_exchanger=MagicMock()), training=restricted_training(subsets))
    downloader = MagicMock(download_training_data=AsyncMock(return_value=(images, 0)))
    with patch.object(trainer_logic_generic, 'TrainingsDownloader', return_value=downloader) as downloader_class:
        await TrainerLogicGeneric._prepare(trainer)  # type: ignore[arg-type]
    return downloader_class


async def test_preparing_hands_the_subsets_to_the_downloader():
    downloader_class = await prepare([image('a', 'train', 'winter'), image('t', 'test', 'hard')], SUBSETS)

    assert downloader_class.call_args.kwargs['subsets'] == SUBSETS


async def test_preparing_names_the_subset_without_images():
    with pytest.raises(CriticalError, match="no train images in the subsets 'winter'"):
        await prepare([image('t', 'test', 'hard')], SUBSETS)


@pytest.mark.parametrize('subsets', [SUBSETS, {}])
async def test_syncing_reports_the_applied_subsets(subsets: dict[str, list[dict[str, str]]]):
    training = restricted_training(subsets)
    training.image_data = [image('a', 'train', 'winter'), image('t', 'test', 'hard')]
    sio_client = SimpleNamespace(connected=True, call=AsyncMock(return_value={'success': True}))
    trainer = SimpleNamespace(node=SimpleNamespace(sio_client=sio_client, uuid='n', name='node'), training=training,
                              errors=MagicMock(), hyperparameters_for_state_sync={}, _on_metrics_published=MagicMock(),
                              _get_new_best_training_state=lambda: TrainingStateData(confusion_matrix={}))

    await TrainerLogicGeneric._sync_training(trainer)  # type: ignore[arg-type]

    assert sio_client.call.call_args.args[1][2]['subsets'] == subsets
