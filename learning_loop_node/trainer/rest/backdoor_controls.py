"""These restful endpoints are only to be used for testing purposes and are not part of the 'offical' trainer behavior."""
from __future__ import annotations

import logging
from dataclasses import asdict
from typing import TYPE_CHECKING, Dict

from fastapi import APIRouter, HTTPException, Request

from ...data_classes import ErrorConfiguration, NodeState
from ..trainer_logic import TrainerLogic

if TYPE_CHECKING:
    from ..trainer_node import TrainerNode

router = APIRouter()


@router.put("/socketio")
async def switch_socketio(request: Request):
    '''
    Example Usage

        curl -X PUT -d "on" http://localhost:8001/socketio
    '''
    state = str(await request.body(), 'utf-8')
    await _switch_socketio(state, request.app)


async def _switch_socketio(state: str, trainer_node: TrainerNode):
    if state == 'off':
        if trainer_node.status.state != NodeState.Offline:
            logging.debug('turning socketio off')
            await trainer_node.sio_client.disconnect()  # pylint: disable=protected-access
    if state == 'on':
        if trainer_node.status.state == NodeState.Offline:
            logging.debug('turning socketio on')
            await trainer_node.connect_sio()


@router.put("/provide_new_model")
async def provide_new_model(request: Request):
    value = str(await request.body(), 'utf-8')
    trainer_node = trainer_node_from_request(request)
    # trainer_logic is MockTrainerLogic which has a property provide_new_model
    assert hasattr(trainer_node.trainer_logic,
                   'provide_new_model'), 'trainer_logic does not have property provide_new_model'
    if value == 'off':
        trainer_node.trainer_logic.provide_new_model = False  # type: ignore
        trainer_node.status.reset_all_errors()
    if value == 'on':
        trainer_node.trainer_logic.provide_new_model = True  # type: ignore

    logging.debug(f'turning automatically provide_new_model {value}')


@router.post("/reset")
async def reset(request: Request):
    trainer_node = trainer_node_from_request(request)
    await _switch_socketio('on', trainer_node)

    await trainer_node.trainer_logic.stop()  # NOTE first stop may only kill running training process
    await trainer_node.trainer_logic.stop()

    trainer_node.last_training_io.delete()

    trainer_node.status.reset_all_errors()
    logging.info('training should be killed, sending new state to LearningLoop')
    await trainer_node.send_status()


@router.put("/error_configuration")
def set_error_configuration(msg: Dict, request: Request):
    '''
    Example Usage
        curl -X PUT http://localhost:8001/error_configuration -d '{"get_new_model": "True"}' -H  "Content-Type: application/json"
    '''
    logging.info(msg)
    error_configuration = ErrorConfiguration(begin_training=msg.get('begin_training', None),
                                             crash_training=msg.get('crash_training', None),
                                             get_new_model=msg.get('get_new_model', None),
                                             save_model=msg.get('save_model', None), )

    logging.info(f'setting error configuration to: {asdict(error_configuration)}')
    trainer_logic = trainer_node_from_request(request).trainer_logic

    # NOTE: trainer_logic is MockTrainerLogic which has a property error_configuration
    assert hasattr(trainer_logic, 'error_configuration'), 'trainer_logic does not have property error_configuration'
    trainer_logic.error_configuration = error_configuration  # type: ignore


@router.post("/steps")
async def add_steps(request: Request):
    logging.warning('Steps was called')
    trainer_node = trainer_node_from_request(request)
    trainer_logic = trainer_node.trainer_logic  # NOTE: is MockTrainerLogic which has 'provide_new_model' and 'current_iteration'

    assert isinstance(trainer_logic, TrainerLogic), 'trainer_logic is not TrainerLogic'

    if not trainer_logic._executor or not trainer_logic._executor.is_running():  # pylint: disable=protected-access
        training = trainer_logic._training  # pylint: disable=protected-access
        logging.error(f'cannot add steps when not running, state: {training.training_state if training else "None"}')
        raise HTTPException(status_code=409, detail="trainer is not running")

    steps = int(str(await request.body(), 'utf-8'))

    previous_state = trainer_logic.provide_new_model  # type: ignore
    trainer_logic.provide_new_model = True  # type: ignore
    logging.warning(f'simulating newly completed models by moving {steps} forward')

    for _ in range(steps):
        try:
            logging.warning('calling sync_confusion_matrix')
            await trainer_logic._sync_confusion_matrix()  # pylint: disable=protected-access
        except Exception:
            pass  # Tests can force synchroniation to fail, error state is reported to backend
    trainer_logic.provide_new_model = previous_state  # type: ignore
    logging.warning(f'progress increased to {trainer_logic.current_iteration}')  # type: ignore
    await trainer_node.send_status()


@router.post("/kill_training_process")
async def kill_process(request: Request):

    # pylint: disable=protected-access
    trainer_node = trainer_node_from_request(request)
    trainer_logic = trainer_node.trainer_logic
    assert isinstance(trainer_logic, TrainerLogic), 'trainer_logic is not TrainerLogic'
    if not trainer_logic._executor or not trainer_logic._executor.is_running():
        raise HTTPException(status_code=409, detail="trainer is not running")
    await trainer_logic._executor.stop_and_wait()


@router.post("/force_status_update")
async def force_status_update(request: Request):
    trainer_node = trainer_node_from_request(request)
    await trainer_node.send_status()


def trainer_node_from_request(request: Request) -> TrainerNode:
    return request.app
