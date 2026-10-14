from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from ... import data_exchanger
from ...data_classes import Context
from ...data_exchanger import DataExchanger
from ...enums import TrainerState
from ...globals import GLOBALS
from ...trainer.exceptions import CriticalError, ModelAlreadyUploadedError
from ...trainer.trainer_node import TrainerNode
from ..trainer.testing_trainer_logic import TestingTrainerLogic

# pylint: disable=protected-access

CONTEXT = Context(organization='zauberzeug', project='demo')
ACCEPTED = httpx.Response(200, json={'id': 'model-id'})


@pytest.fixture(autouse=True)
def no_retry_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data_exchanger, 'MODEL_UPLOAD_RETRY_DELAY', 0)


@pytest.mark.parametrize(('status', 'error'), [
    (409, ModelAlreadyUploadedError),
    (404, CriticalError),
    (500, CriticalError),
])
async def test_a_refused_upload_tells_the_trainer_to_skip_or_give_up(status: int, error: type[Exception]):
    communicator = _communicator(httpx.Response(status, text='no'))
    with pytest.raises(error):
        await DataExchanger(None, communicator).upload_model_get_uuid(CONTEXT, [], 1, 'some_format')
    assert communicator.put.await_count == 1


async def test_an_accepted_upload_returns_the_model_id():
    communicator = _communicator(ACCEPTED)
    assert await DataExchanger(None, communicator).upload_model_get_uuid(CONTEXT, [], 1, 'some_format') == 'model-id'


@pytest.mark.parametrize('status', [408, 429, 502, 503, 504])
async def test_an_error_that_may_pass_is_retried(status: int):
    communicator = _communicator(httpx.Response(status), ACCEPTED)
    assert await DataExchanger(None, communicator).upload_model_get_uuid(CONTEXT, [], 1, 'some_format') == 'model-id'


async def test_an_error_that_does_not_pass_ends_the_upload_after_the_last_retry():
    communicator = _communicator(*[httpx.Response(502)] * data_exchanger.MAX_MODEL_UPLOAD_ATTEMPTS)
    with pytest.raises(CriticalError):
        await DataExchanger(None, communicator).upload_model_get_uuid(CONTEXT, [], 1, 'some_format')
    assert communicator.put.await_count == data_exchanger.MAX_MODEL_UPLOAD_ATTEMPTS


async def test_a_format_the_loop_already_stores_counts_as_uploaded(trainer: TestingTrainerLogic):
    _answer_uploads(trainer, ModelAlreadyUploadedError(), 'model-id')

    assert await trainer._upload_model_return_new_model_uuid(CONTEXT) == 'model-id'
    assert trainer.active_training_io.load_model_upload_progress() == ['first', 'second']


async def test_a_retry_after_a_restart_still_knows_the_model_id(trainer: TestingTrainerLogic):
    _answer_uploads(trainer, 'model-id', httpx.ReadTimeout('timed out'))
    with pytest.raises(httpx.ReadTimeout):
        await trainer._upload_model_return_new_model_uuid(CONTEXT)
    assert trainer.active_training_io.load_model_upload_progress() == ['first']

    trainer._init_from_last_training()
    _answer_uploads(trainer, ModelAlreadyUploadedError())
    assert await trainer._upload_model_return_new_model_uuid(CONTEXT) == 'model-id'


async def test_an_uploaded_model_without_a_known_id_ends_the_training(trainer: TestingTrainerLogic):
    _answer_uploads(trainer, ModelAlreadyUploadedError(), ModelAlreadyUploadedError())

    with pytest.raises(CriticalError):
        await trainer._upload_model_return_new_model_uuid(CONTEXT)


async def test_the_upload_state_outlasts_an_error_that_passes(trainer: TestingTrainerLogic):
    trainer.node.data_exchanger.loop_communicator = _communicator(httpx.Response(502), ACCEPTED, ACCEPTED)

    await _perform_upload_state(trainer)

    assert trainer.training.training_state == TrainerState.TrainModelUploaded
    assert trainer.training.model_uuid_for_detecting == 'model-id'


async def test_a_rejected_format_leaves_the_training_without_a_model_id(trainer: TestingTrainerLogic):
    trainer.node.data_exchanger.loop_communicator = _communicator(ACCEPTED, httpx.Response(413))

    await _perform_upload_state(trainer)

    assert trainer.training.training_state == TrainerState.ReadyForCleanup
    assert trainer.training.model_uuid_for_detecting is None


@pytest.fixture(name='trainer')
async def trainer_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[TestingTrainerLogic]:
    """A trainer with a fresh training whose model comes in two formats."""
    monkeypatch.setattr(GLOBALS, 'data_folder', str(tmp_path))
    trainer = TestingTrainerLogic()
    node = TrainerNode(name='test', trainer_logic=trainer, uuid='NODE-000-0000-0000-0000-000000000000')
    trainer._init_new_training(CONTEXT, {'categories': [], 'training_number': 0, 'model_variant': '',
                                         'hyperparameters': {}})
    trainer.training.training_state = TrainerState.ConfusionMatrixSynced
    node.last_training_io.save(trainer.training)
    trainer._get_latest_model_files = AsyncMock(return_value={'first': ['/tmp/model.pt'],
                                                              'second': ['/tmp/model.wts']})
    yield trainer
    await node.loop_communicator.shutdown()


def _communicator(*responses: httpx.Response) -> Mock:
    return Mock(put=AsyncMock(side_effect=responses))


def _answer_uploads(trainer: TestingTrainerLogic, *answers: str | Exception) -> None:
    trainer.node.data_exchanger.upload_model_get_uuid = AsyncMock(side_effect=answers)


async def _perform_upload_state(trainer: TestingTrainerLogic) -> None:
    await trainer._perform_state('upload_model', TrainerState.TrainModelUploading, TrainerState.TrainModelUploaded,
                                 trainer._upload_model)
