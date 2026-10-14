from pathlib import Path
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from ...data_classes import Context
from ...data_exchanger import DataExchanger, ModelAlreadyUploadedError
from ...globals import GLOBALS
from ...loop_communication import LoopCommunicationException
from ...trainer.exceptions import CriticalError
from ...trainer.trainer_node import TrainerNode
from ..trainer.testing_trainer_logic import TestingTrainerLogic

# pylint: disable=protected-access

CONTEXT = Context(organization='zauberzeug', project='demo')


@pytest.mark.parametrize(('status', 'error'), [
    (409, ModelAlreadyUploadedError),
    (502, LoopCommunicationException),
    (404, CriticalError),
])
async def test_a_refused_upload_tells_the_trainer_to_skip_retry_or_give_up(status: int, error: type[Exception]):
    communicator = Mock(put=AsyncMock(return_value=httpx.Response(status, text='no')))
    with pytest.raises(error):
        await DataExchanger(None, communicator).upload_model_get_uuid(CONTEXT, [], 1, 'some_format')


async def test_an_accepted_upload_returns_the_model_id():
    communicator = Mock(put=AsyncMock(return_value=httpx.Response(200, json={'id': 'model-id'})))
    assert await DataExchanger(None, communicator).upload_model_get_uuid(CONTEXT, [], 1, 'some_format') == 'model-id'


async def test_a_format_the_loop_already_stores_counts_as_uploaded(trainer: TestingTrainerLogic):
    _answer_uploads(trainer, ModelAlreadyUploadedError(), 'model-id')

    assert await trainer._upload_model_return_new_model_uuid(CONTEXT) == 'model-id'
    assert trainer.active_training_io.load_model_upload_progress() == ['first', 'second']


async def test_a_retry_after_a_server_error_still_knows_the_model_id(trainer: TestingTrainerLogic):
    _answer_uploads(trainer, 'model-id', LoopCommunicationException('502'))
    with pytest.raises(LoopCommunicationException):
        await trainer._upload_model_return_new_model_uuid(CONTEXT)
    assert trainer.active_training_io.load_model_upload_progress() == ['first']

    _answer_uploads(trainer, ModelAlreadyUploadedError())
    assert await trainer._upload_model_return_new_model_uuid(CONTEXT) == 'model-id'


async def test_an_uploaded_model_without_a_known_id_ends_the_training(trainer: TestingTrainerLogic):
    _answer_uploads(trainer, ModelAlreadyUploadedError(), ModelAlreadyUploadedError())

    with pytest.raises(CriticalError):
        await trainer._upload_model_return_new_model_uuid(CONTEXT)


@pytest.fixture(name='trainer')
def trainer_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestingTrainerLogic:
    """A trainer with a fresh training whose model comes in two formats."""
    monkeypatch.setattr(GLOBALS, 'data_folder', str(tmp_path))
    trainer = TestingTrainerLogic()
    TrainerNode(name='test', trainer_logic=trainer, uuid='NODE-000-0000-0000-0000-000000000000')
    trainer._init_new_training(CONTEXT, {'categories': [], 'training_number': 0, 'model_variant': '',
                                         'hyperparameters': {}})
    trainer._get_latest_model_files = AsyncMock(return_value={'first': ['/tmp/model.pt'],
                                                              'second': ['/tmp/model.wts']})
    return trainer


def _answer_uploads(trainer: TestingTrainerLogic, *answers: str | Exception) -> None:
    trainer.node.data_exchanger.upload_model_get_uuid = AsyncMock(side_effect=answers)
