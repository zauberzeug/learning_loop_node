import asyncio
import json
import zipfile
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from ...data_classes import Context
from ...data_exchanger import DataExchanger
from ...enums import TrainerState
from ...globals import GLOBALS
from ...trainer.exceptions import CriticalError, ModelUploadFailedError
from ...trainer.trainer_node import TrainerNode
from ...trainer.training_results import META_FILE, parts_to_upload, write_zip
from ..trainer.testing_trainer_logic import TestingTrainerLogic

# pylint: disable=protected-access

CONTEXT = Context(organization='zauberzeug', project='demo')
MODEL_ID = '6f1c7a52-4b1e-4e0a-9d3f-2b8c5e7a9d10'
ACCEPTED_MODEL = httpx.Response(200, json={'id': MODEL_ID})
ACCEPTED_RESULTS = httpx.Response(200, json={})
ALL_PARTS = {'core': [], 'examples': [], 'logs': []}


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    async def sleep(_: float) -> None:
        pass
    monkeypatch.setattr(asyncio, 'sleep', sleep)


@pytest.mark.parametrize(('hyperparameters', 'parts'), [
    ({}, ['core']),
    ({'upload_examples': True}, ['core', 'examples']),
    ({'upload_logs': True}, ['core', 'logs']),
    ({'upload_examples': True, 'upload_logs': True}, ['core', 'examples', 'logs']),
    ({'upload_examples': False, 'upload_logs': False}, ['core']),
])
def test_core_is_always_uploaded_and_first_the_other_parts_only_if_asked_for(hyperparameters: dict, parts: list[str]):
    assert parts_to_upload({'examples': [], 'logs': [], 'core': []}, hyperparameters) == parts


def test_a_malformed_upload_flag_is_a_critical_error():
    with pytest.raises(CriticalError):
        parts_to_upload(ALL_PARTS, {'upload_examples': 'true'})


def test_the_zip_keeps_the_paths_inside_the_training_folder(tmp_path: Path):
    training_folder = tmp_path / 'training'
    (training_folder / 'result' / 'examples').mkdir(parents=True)
    (training_folder / 'result' / 'results.csv').write_text('epoch\n1\n')
    (training_folder / 'result' / 'examples' / 'a.png').write_text('png')
    (training_folder / 'hyp.yaml').write_text('lr: 1')
    zip_path = training_folder / 'training_results' / 'core.zip'

    write_zip([training_folder / 'result' / 'results.csv', training_folder / 'result' / 'examples',
               training_folder / 'hyp.yaml', training_folder / 'missing.json'],
              training_folder, zip_path, meta={'node_name': 'test'})

    with zipfile.ZipFile(zip_path) as zipf:
        assert sorted(zipf.namelist()) == ['hyp.yaml', META_FILE, 'result/examples/a.png', 'result/results.csv']
        assert zipf.read('result/results.csv') == b'epoch\n1\n'
        assert json.loads(zipf.read(META_FILE)) == {'node_name': 'test'}
    assert not zip_path.with_name('core.zip.partial').exists()


async def test_a_malformed_upload_flag_ends_the_training_before_it_has_run(trainer: TestingTrainerLogic):
    trainer.training.hyperparameters['upload_logs'] = 'sometimes'

    with pytest.raises(CriticalError):
        await trainer._prepare()


async def test_accepted_results_are_stored_beside_the_latest_model():
    communicator = _communicator(ACCEPTED_RESULTS)

    await DataExchanger(None, communicator).upload_training_results(CONTEXT, 3, 'core', '/tmp/core.zip')

    assert communicator.put.await_args.args[0] == \
        '/zauberzeug/projects/demo/trainings/3/models/latest/training_results/core'
    assert communicator.put.await_args.kwargs['files'] == ['/tmp/core.zip']
    assert communicator.put.await_args.kwargs['timeout'] == 600


@pytest.mark.parametrize(('status', 'error'), [
    (404, CriticalError),
    (413, CriticalError),
    (408, ModelUploadFailedError),
    (500, ModelUploadFailedError),
    (507, ModelUploadFailedError),
])
async def test_results_the_loop_does_not_store_are_answered_like_a_model(status: int, error: type[Exception]):
    communicator = _communicator(httpx.Response(status, text='no'))

    with pytest.raises(error):
        await DataExchanger(None, communicator).upload_training_results(CONTEXT, 3, 'core', '/tmp/core.zip')


async def test_the_results_follow_the_model_in_the_upload_state(trainer: TestingTrainerLogic):
    trainer.training.hyperparameters['upload_logs'] = True
    communicator = _communicator(ACCEPTED_MODEL, ACCEPTED_MODEL, ACCEPTED_RESULTS, ACCEPTED_RESULTS)
    trainer.node.data_exchanger.loop_communicator = communicator

    await _perform_upload_state(trainer)

    assert _uploads(communicator) == ['first/file', 'second/file', 'training_results/core', 'training_results/logs']
    assert trainer.training.training_state == TrainerState.TrainModelUploaded
    assert trainer.training.model_uuid_for_detecting == MODEL_ID
    with zipfile.ZipFile(Path(trainer.training.training_folder) / 'training_results' / 'core.zip') as zipf:
        assert sorted(zipf.namelist()) == [META_FILE, 'result/results.csv']
        meta = json.loads(zipf.read(META_FILE))
    assert meta['model_uuid'] == MODEL_ID
    assert meta['training_number'] == 3
    assert meta['node_name'] == 'test'
    assert meta['stop_requested'] is False
    assert meta['hyperparameters'] == {'upload_logs': True}


async def test_a_trainer_without_results_uploads_only_its_model(trainer: TestingTrainerLogic):
    trainer._get_training_results = lambda _: {}  # type: ignore[method-assign]
    communicator = _communicator(ACCEPTED_MODEL, ACCEPTED_MODEL)
    trainer.node.data_exchanger.loop_communicator = communicator

    await _perform_upload_state(trainer)

    assert _uploads(communicator) == ['first/file', 'second/file']
    assert trainer.training.training_state == TrainerState.TrainModelUploaded


async def test_results_the_loop_cannot_store_keep_the_training_and_are_sent_again_later(trainer: TestingTrainerLogic):
    trainer.training.hyperparameters['upload_logs'] = True
    communicator = _communicator(ACCEPTED_MODEL, ACCEPTED_MODEL, ACCEPTED_RESULTS, httpx.Response(507),
                                 ACCEPTED_RESULTS)
    trainer.node.data_exchanger.loop_communicator = communicator

    await _perform_upload_state(trainer)

    assert trainer.training.training_state == TrainerState.ConfusionMatrixSynced
    assert trainer.errors.has_error_for('upload_model')
    assert trainer.training.model_uuid_for_detecting is None

    await _perform_upload_state(trainer)

    assert _uploads(communicator) == ['first/file', 'second/file', 'training_results/core', 'training_results/logs',
                                      'training_results/logs']
    assert trainer.training.training_state == TrainerState.TrainModelUploaded
    assert trainer.training.model_uuid_for_detecting == MODEL_ID


async def test_rejected_results_end_the_training_like_a_rejected_model(trainer: TestingTrainerLogic):
    trainer.node.data_exchanger.loop_communicator = _communicator(ACCEPTED_MODEL, ACCEPTED_MODEL, httpx.Response(404))

    await _perform_upload_state(trainer)

    assert trainer.training.training_state == TrainerState.ReadyForCleanup
    assert trainer.training.model_uuid_for_detecting is None


@pytest.fixture(name='trainer')
async def trainer_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[TestingTrainerLogic]:
    """A trainer with a fresh training whose model comes in two formats and whose results are on disk."""
    monkeypatch.setattr(GLOBALS, 'data_folder', str(tmp_path))
    trainer = TestingTrainerLogic()
    node = TrainerNode(name='test', trainer_logic=trainer, uuid='NODE-000-0000-0000-0000-000000000000')
    trainer._init_new_training(CONTEXT, {'categories': [], 'training_number': 3, 'model_variant': '',
                                         'hyperparameters': {}})
    trainer.training.training_state = TrainerState.ConfusionMatrixSynced
    node.last_training_io.save(trainer.training)
    trainer._get_latest_model_files = AsyncMock(return_value={'first': ['/tmp/model.pt'],  # type: ignore[method-assign]
                                                              'second': ['/tmp/model.wts']})
    training_folder = Path(trainer.training.training_folder)
    (training_folder / 'result' / 'examples').mkdir(parents=True)
    (training_folder / 'result' / 'results.csv').write_text('epoch\n1\n')
    (training_folder / 'result' / 'examples' / 'a.png').write_text('png')
    (training_folder / 'last_training.log').write_text('log')
    trainer._get_training_results = lambda folder: {  # type: ignore[method-assign]
        'core': [folder / 'result' / 'results.csv'],
        'examples': [folder / 'result' / 'examples'],
        'logs': [folder / 'last_training.log']}
    yield trainer
    await node.loop_communicator.shutdown()


def _communicator(*responses: httpx.Response) -> Mock:
    return Mock(put=AsyncMock(side_effect=responses))


def _uploads(communicator: Mock) -> list[str]:
    return ['/'.join(put.args[0].split('/')[-2:]) for put in communicator.put.await_args_list]


async def _perform_upload_state(trainer: TestingTrainerLogic) -> None:
    """The upload state as `_training_loop` performs it."""
    await trainer._perform_state('upload_model', TrainerState.TrainModelUploading, TrainerState.TrainModelUploaded,
                                 trainer._upload_model)
