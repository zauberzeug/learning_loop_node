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
from ...trainer.states import restrict_to_states, states_to_fetch
from ...trainer.trainer_logic_generic import TrainerLogicGeneric

STATES = {'train': ['complete', 'review'], 'test': ['complete']}


def image(image_id: str, image_set: str, state: str) -> dict:
    return {'id': image_id, 'set': image_set, 'state': state}


def image_data() -> list[dict]:
    return [image('a', 'train', 'complete'), image('b', 'train', 'review'), image('c', 'train', 'annotate'),
            image('v', 'val', 'complete'), image('w', 'val', 'review'),
            image('t', 'test', 'complete'), image('u', 'test', 'review')]


def ids(images: list[dict]) -> list[str]:
    return [i['id'] for i in images]


def test_a_set_keeps_the_images_in_its_states():
    assert ids(restrict_to_states(image_data(), STATES)) == ['a', 'b', 'v', 't']


def test_a_set_that_is_not_named_keeps_its_complete_images():
    assert ids(restrict_to_states(image_data(), {'test': ['review']})) == ['a', 'v', 'u']


def test_without_states_all_images_stay():
    assert restrict_to_states(image_data(), {}) == image_data()


def test_an_image_without_a_state_is_in_none():
    assert not restrict_to_states([{'id': 'a', 'set': 'train'}], STATES)


def test_each_state_of_any_set_is_fetched_once():
    assert states_to_fetch({'train': ['backlog', 'complete'], 'test': ['review', 'complete']}) == \
        ['backlog', 'complete', 'review']


def test_the_complete_images_are_fetched_for_a_set_that_is_not_named():
    assert states_to_fetch({'train': ['review']}) == ['review', 'complete']


def generate(tmp_path: Path, **config) -> Training:
    data = {'hyperparameters': {}, 'categories': [], 'training_number': 1, 'model_variant': 'small', **config}
    return Training.generate_training(str(tmp_path), Context(organization='o', project='p'), data)


def test_a_training_takes_the_states_of_its_config(tmp_path: Path):
    assert generate(tmp_path, states=STATES).states == STATES


def test_a_config_without_states_names_none(tmp_path: Path):
    assert generate(tmp_path).states == {}


@pytest.mark.parametrize('states', [['review'], {'prepare': ['review']}, {'train': 'review'}, {'train': []},
                                    {'train': [1]}])
def test_malformed_states_are_rejected(tmp_path: Path, states):
    with pytest.raises(ValueError, match='states'):
        generate(tmp_path, states=states)


def test_a_training_saved_without_states_loads_with_none(tmp_path: Path):
    saved = asdict(generate(tmp_path, states=STATES))
    del saved['states']

    assert from_dict(data_class=Training, data=saved).states == {}


def data_exchanger(images_by_state: dict[str, list[dict]], downloaded: list[str]) -> MagicMock:
    """A loop that holds the given images; the ids of the downloaded image files are appended to ``downloaded``."""
    images = {i['id']: i for state_images in images_by_state.values() for i in state_images}

    async def fetch_image_uuids(query_params: str) -> list[str]:
        return ids(images_by_state.get(query_params.removeprefix('state='), []))

    async def download_images_data(image_ids: list[str]) -> list[dict]:
        return [images[image_id] for image_id in image_ids]

    async def download_images(image_ids: list[str], image_folder: str) -> None:
        downloaded.extend(image_ids)
        for image_id in image_ids:
            (Path(image_folder) / f'{image_id}.jpg').touch()

    return MagicMock(fetch_image_uuids=AsyncMock(side_effect=fetch_image_uuids),
                     download_images_data=AsyncMock(side_effect=download_images_data),
                     download_images=AsyncMock(side_effect=download_images))


def images_by_state() -> dict[str, list[dict]]:
    return {state: [i for i in image_data() if i['state'] == state] for state in ('complete', 'review', 'annotate')}


async def test_the_downloader_takes_each_set_from_its_states(tmp_path: Path):
    downloaded: list[str] = []
    exchanger = data_exchanger(images_by_state(), downloaded)

    images, skipped = await TrainingsDownloader(exchanger, states=STATES).download_training_data(str(tmp_path))

    assert [call.kwargs['query_params'] for call in exchanger.fetch_image_uuids.call_args_list] == \
        ['state=complete', 'state=review']
    assert ids(images) == downloaded == ['a', 'v', 't', 'b']
    assert skipped == 0


async def test_the_downloader_takes_the_complete_images_without_states(tmp_path: Path):
    downloaded: list[str] = []
    exchanger = data_exchanger(images_by_state(), downloaded)

    images, _ = await TrainingsDownloader(exchanger).download_training_data(str(tmp_path))

    assert [call.kwargs['query_params'] for call in exchanger.fetch_image_uuids.call_args_list] == ['state=complete']
    assert ids(images) == downloaded == ['a', 'v', 't']


async def test_the_downloader_lists_an_image_once_that_changed_its_state_in_between(tmp_path: Path):
    moved = image('a', 'train', 'review')
    downloaded: list[str] = []
    exchanger = data_exchanger({'complete': [moved], 'review': [moved]}, downloaded)

    images, _ = await TrainingsDownloader(exchanger, states=STATES).download_training_data(str(tmp_path))

    assert ids(images) == downloaded == ['a']


def training_with(states: dict[str, list[str]]) -> Training:
    return Training(id='t', context=Context(organization='o', project='p'), project_folder='', images_folder='',
                    training_folder='', categories=[], hyperparameters={}, training_number=1,
                    training_state='', model_variant='', states=states)


async def prepare(images: list[dict], states: dict[str, list[str]]) -> MagicMock:
    """Prepare a training with ``states`` whose download yields ``images``; return the downloader class."""
    trainer = SimpleNamespace(node=SimpleNamespace(data_exchanger=MagicMock()), training=training_with(states))
    downloader = MagicMock(download_training_data=AsyncMock(return_value=(images, 0)))
    with patch.object(trainer_logic_generic, 'TrainingsDownloader', return_value=downloader) as downloader_class:
        await TrainerLogicGeneric._prepare(trainer)  # type: ignore[arg-type]
    return downloader_class


async def test_preparing_hands_the_states_to_the_downloader():
    downloader_class = await prepare([image('a', 'train', 'review'), image('t', 'test', 'complete')], STATES)

    assert downloader_class.call_args.kwargs['states'] == STATES


async def test_preparing_names_the_states_of_a_set_without_images():
    with pytest.raises(CriticalError, match="no train images in the states 'complete', 'review'"):
        await prepare([image('t', 'test', 'complete')], STATES)


@pytest.mark.parametrize('states', [STATES, {}])
async def test_syncing_reports_the_applied_states(states: dict[str, list[str]]):
    training = training_with(states)
    training.image_data = [image('a', 'train', 'review'), image('t', 'test', 'complete')]
    sio_client = SimpleNamespace(connected=True, call=AsyncMock(return_value={'success': True}))
    trainer = SimpleNamespace(node=SimpleNamespace(sio_client=sio_client, uuid='n', name='node'), training=training,
                              errors=MagicMock(), hyperparameters_for_state_sync={}, _on_metrics_published=MagicMock(),
                              _get_new_best_training_state=lambda: TrainingStateData(confusion_matrix={}))

    await TrainerLogicGeneric._sync_training(trainer)  # type: ignore[arg-type]

    assert sio_client.call.call_args.args[1][2]['states'] == states
