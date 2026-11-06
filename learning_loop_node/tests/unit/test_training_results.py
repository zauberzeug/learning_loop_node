import asyncio
import json
import zipfile
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from ...data_classes import Context
from ...data_exchanger import DataExchanger
from ...enums import TrainerState
from ...globals import GLOBALS
from ...trainer.exceptions import CriticalError, TrainingResultsNotAnsweredError, TrainingResultsUploadFailedError
from ...trainer.trainer_node import TrainerNode
from ...trainer import training_results
from ...trainer.training_results import MAX_ATTEMPTS, META_FILE, check_hyperparameters, upload_results
from ..trainer.testing_trainer_logic import TestingTrainerLogic

# pylint: disable=protected-access

CONTEXT = Context(organization='zauberzeug', project='demo')
MODEL_ID = '6f1c7a52-4b1e-4e0a-9d3f-2b8c5e7a9d10'
ACCEPTED_MODEL = httpx.Response(200, json={'id': MODEL_ID})
ACCEPTED_RESULTS = httpx.Response(200, json={})


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
async def test_core_is_always_uploaded_and_first_the_other_parts_only_if_asked_for(
        trainer: TestingTrainerLogic, hyperparameters: dict, parts: list[str]):
    trainer.training.hyperparameters.update(hyperparameters)
    communicator = _communicator(*[ACCEPTED_RESULTS] * 3)

    await _upload_results(trainer, communicator, lambda _: {'examples': [], 'logs': [], 'core': []})

    assert _uploads(communicator) == [f'training_results/{part}' for part in parts]


def test_a_malformed_upload_flag_is_a_critical_error():
    with pytest.raises(CriticalError):
        check_hyperparameters({'upload_examples': 'true'})


async def test_the_zip_keeps_the_paths_inside_the_training_folder(trainer: TestingTrainerLogic):
    training_folder = Path(trainer.training.training_folder)
    (training_folder / 'hyp.yaml').write_text('lr: 1')

    await _upload_results(trainer, _communicator(ACCEPTED_RESULTS), lambda folder: {'core': [
        folder / 'result' / 'results.csv', folder / 'result' / 'examples', folder / 'hyp.yaml',
        folder / 'missing.json']})

    zip_path = training_folder / 'training_results' / 'core.zip'
    with zipfile.ZipFile(zip_path) as zipf:
        assert sorted(zipf.namelist()) == ['hyp.yaml', META_FILE, 'result/examples/a.png', 'result/results.csv']
        assert zipf.read('result/results.csv') == b'epoch\n1\n'
        assert json.loads(zipf.read(META_FILE))['model_uuid'] == MODEL_ID
    assert not zip_path.with_name('core.zip.partial').exists()


async def test_the_zip_leaves_out_the_results_folder_itself(trainer: TestingTrainerLogic):
    training_folder = Path(trainer.training.training_folder)
    (training_folder / 'training_results').mkdir()
    (training_folder / 'training_results' / 'examples.zip').write_text('an earlier part')

    await _upload_results(trainer, _communicator(ACCEPTED_RESULTS), lambda folder: {'core': [folder]})

    with zipfile.ZipFile(training_folder / 'training_results' / 'core.zip') as zipf:
        assert not [name for name in zipf.namelist() if name.startswith('training_results/')]
        assert 'result/results.csv' in zipf.namelist()


async def test_paths_outside_the_training_folder_keep_their_named_directory(trainer: TestingTrainerLogic,
                                                                           tmp_path: Path):
    for run in ('a', 'b'):
        (tmp_path / 'runs' / run).mkdir(parents=True)
        (tmp_path / 'runs' / run / 'curve.png').write_text(run)

    await _upload_results(trainer, _communicator(ACCEPTED_RESULTS),
                          lambda _: {'core': [tmp_path / 'runs' / 'a', tmp_path / 'runs' / 'b']})

    with zipfile.ZipFile(Path(trainer.training.training_folder) / 'training_results' / 'core.zip') as zipf:
        assert sorted(n for n in zipf.namelist() if n != META_FILE) == ['a/curve.png', 'b/curve.png']


async def test_compressed_files_are_stored_and_the_others_compressed(trainer: TestingTrainerLogic):
    await _upload_results(trainer, _communicator(ACCEPTED_RESULTS), lambda folder: {'core': [folder / 'result']})

    with zipfile.ZipFile(Path(trainer.training.training_folder) / 'training_results' / 'core.zip') as zipf:
        assert zipf.getinfo('result/examples/a.png').compress_type == zipfile.ZIP_STORED
        assert zipf.getinfo('result/results.csv').compress_type == zipfile.ZIP_DEFLATED


async def test_a_malformed_upload_flag_ends_the_training_before_it_has_run(trainer: TestingTrainerLogic):
    trainer.training.hyperparameters['upload_logs'] = 'sometimes'

    with pytest.raises(CriticalError):
        await trainer._prepare()


async def test_accepted_results_are_stored_beside_the_latest_model():
    communicator = _communicator(ACCEPTED_RESULTS)

    await DataExchanger(None, communicator).upload_training_results(CONTEXT, 3, 'core', '/tmp/core.zip')

    assert communicator.put.await_args.args[0] == \
        '/zauberzeug/projects/demo/trainings/3/training_results/core'
    assert communicator.put.await_args.kwargs['files'] == ['/tmp/core.zip']
    assert communicator.put.await_args.kwargs['timeout'] == 600


@pytest.mark.parametrize(('status', 'error'), [
    (404, TrainingResultsUploadFailedError),
    (413, TrainingResultsUploadFailedError),
    (422, TrainingResultsUploadFailedError),
    (408, TrainingResultsNotAnsweredError),
    (500, TrainingResultsNotAnsweredError),
    (502, TrainingResultsNotAnsweredError),
    (503, TrainingResultsNotAnsweredError),
])
async def test_a_client_error_is_final_and_a_408_or_server_error_counts_as_no_answer(status: int, error: type[Exception]):
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


@pytest.mark.parametrize('no_answer', [httpx.ReadTimeout('no answer'), httpx.Response(408), httpx.Response(503)])
async def test_a_part_that_got_no_answer_is_sent_again_in_the_same_state(
        trainer: TestingTrainerLogic, no_answer: Exception | httpx.Response):
    trainer.training.hyperparameters['upload_logs'] = True
    communicator = _communicator(ACCEPTED_MODEL, ACCEPTED_MODEL, ACCEPTED_RESULTS, no_answer, ACCEPTED_RESULTS)
    trainer.node.data_exchanger.loop_communicator = communicator

    await _perform_upload_state(trainer)

    assert _uploads(communicator) == ['first/file', 'second/file', 'training_results/core', 'training_results/logs',
                                      'training_results/logs']
    assert trainer.training.training_state == TrainerState.TrainModelUploaded
    assert trainer.training.model_uuid_for_detecting == MODEL_ID
    assert not trainer.errors.has_error_for('upload_model')


async def test_a_part_that_never_gets_an_answer_is_skipped_and_the_training_goes_on(trainer: TestingTrainerLogic):
    communicator = _communicator(ACCEPTED_MODEL, ACCEPTED_MODEL, *[httpx.ReadTimeout('no answer')] * MAX_ATTEMPTS)
    trainer.node.data_exchanger.loop_communicator = communicator

    await _perform_upload_state(trainer)

    assert _uploads(communicator) == ['first/file', 'second/file'] + ['training_results/core'] * MAX_ATTEMPTS
    assert trainer.training.training_state == TrainerState.TrainModelUploaded
    assert trainer.training.model_uuid_for_detecting == MODEL_ID


async def test_a_failing_results_hook_does_not_stop_the_training(trainer: TestingTrainerLogic):
    def broken_hook(_: Path) -> dict[str, list[Path]]:
        raise RuntimeError('bug in the trainer')
    trainer._get_training_results = broken_hook  # type: ignore[method-assign]
    communicator = _communicator(ACCEPTED_MODEL, ACCEPTED_MODEL)
    trainer.node.data_exchanger.loop_communicator = communicator

    await _perform_upload_state(trainer)

    assert _uploads(communicator) == ['first/file', 'second/file']
    assert trainer.training.training_state == TrainerState.TrainModelUploaded
    assert trainer.training.model_uuid_for_detecting == MODEL_ID


async def test_a_failing_meta_does_not_stop_the_training(trainer: TestingTrainerLogic):
    trainer.training.start_time = None  # type: ignore[assignment]
    communicator = _communicator(ACCEPTED_MODEL, ACCEPTED_MODEL)
    trainer.node.data_exchanger.loop_communicator = communicator

    await _perform_upload_state(trainer)

    assert _uploads(communicator) == ['first/file', 'second/file']
    assert trainer.training.training_state == TrainerState.TrainModelUploaded


async def test_a_malformed_upload_flag_at_upload_time_skips_only_its_part(trainer: TestingTrainerLogic):
    trainer.training.hyperparameters['upload_logs'] = 'true'  # a training prepared before the flag was checked
    communicator = _communicator(ACCEPTED_MODEL, ACCEPTED_MODEL, ACCEPTED_RESULTS)
    trainer.node.data_exchanger.loop_communicator = communicator

    await _perform_upload_state(trainer)

    assert _uploads(communicator) == ['first/file', 'second/file', 'training_results/core']
    assert trainer.training.training_state == TrainerState.TrainModelUploaded


async def test_a_part_of_another_name_is_left_out(trainer: TestingTrainerLogic):
    communicator = _communicator(ACCEPTED_RESULTS)

    await _upload_results(trainer, communicator, lambda folder: {'core': [], 'log': [], '../escape': []})

    assert _uploads(communicator) == ['training_results/core']
    assert not (Path(trainer.training.training_folder) / 'escape.zip').exists()


async def test_a_failing_record_of_done_parts_does_not_stop_the_upload(trainer: TestingTrainerLogic,
                                                                       monkeypatch: pytest.MonkeyPatch):
    def full_disk(*_) -> None:
        raise OSError(28, 'No space left on device')
    monkeypatch.setattr(training_results, '_save_done', full_disk)
    trainer.training.hyperparameters['upload_logs'] = True
    communicator = _communicator(ACCEPTED_RESULTS, ACCEPTED_RESULTS)

    await _upload_results(trainer, communicator)

    assert _uploads(communicator) == ['training_results/core', 'training_results/logs']


async def test_parts_that_are_done_are_not_sent_again(trainer: TestingTrainerLogic):
    trainer.training.hyperparameters['upload_logs'] = True
    communicator = _communicator(ACCEPTED_RESULTS, httpx.Response(413, text='too large'))

    for _ in range(2):
        await _upload_results(trainer, communicator)

    assert _uploads(communicator) == ['training_results/core', 'training_results/logs']


async def test_results_the_loop_does_not_store_are_skipped_and_the_training_goes_on(trainer: TestingTrainerLogic):
    trainer.training.hyperparameters['upload_logs'] = True
    communicator = _communicator(ACCEPTED_MODEL, ACCEPTED_MODEL, httpx.Response(413, text='too large'),
                                 ACCEPTED_RESULTS)
    trainer.node.data_exchanger.loop_communicator = communicator

    await _perform_upload_state(trainer)

    assert _uploads(communicator) == ['first/file', 'second/file', 'training_results/core', 'training_results/logs']
    assert trainer.training.training_state == TrainerState.TrainModelUploaded
    assert trainer.training.model_uuid_for_detecting == MODEL_ID
    assert not trainer.errors.has_error_for('upload_model')


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


async def _upload_results(trainer: TestingTrainerLogic, communicator: Mock,
                          get_results: Callable[[Path], dict[str, list[Path]]] | None = None) -> None:
    await upload_results(trainer.training, model_uuid=MODEL_ID, node_name='test', node_uuid=trainer.node.uuid,
                         get_results=get_results or trainer._get_training_results, stop_requested=False,
                         data_exchanger=DataExchanger(None, communicator))


def _uploads(communicator: Mock) -> list[str]:
    return ['/'.join(put.args[0].split('/')[-2:]) for put in communicator.put.await_args_list]


async def _perform_upload_state(trainer: TestingTrainerLogic) -> None:
    """The upload state as `_training_loop` performs it."""
    await trainer._perform_state('upload_model', TrainerState.TrainModelUploading, TrainerState.TrainModelUploaded,
                                 trainer._upload_model)
