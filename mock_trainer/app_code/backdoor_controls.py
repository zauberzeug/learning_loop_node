"""These restful endpoints are only to be used for testing purposes and are not part of the 'offical' trainer behavior."""

import logging
from dataclasses import asdict

from fastapi import APIRouter, HTTPException, Request

from learning_loop_node.data_classes import ErrorConfiguration, NodeState
from learning_loop_node.trainer.trainer_node import TrainerNode

from .mock_trainer_logic import MockTrainerLogic

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
    assert isinstance(trainer_node.trainer_logic, MockTrainerLogic)
    if value == 'off':
        trainer_node.trainer_logic.provide_new_model = False
        trainer_node.status.reset_all_errors()
    if value == 'on':
        trainer_node.trainer_logic.provide_new_model = True

    logging.debug(f'turning automatically provide_new_model {value}')


@router.post("/reset")
async def reset(request: Request):
    trainer_node = trainer_node_from_request(request)
    await _switch_socketio('on', trainer_node)

    await trainer_node.trainer_logic.stop()
    await trainer_node.trainer_logic.stop()
    # NOTE first stop may only kill running training process

    trainer_node.last_training_io.delete()

    trainer_node.status.reset_all_errors()
    logging.error('training should be killed, sending new state to LearningLoop')
    await trainer_node.send_status()


@router.put("/error_configuration")
def set_error_configuration(error_configuration: ErrorConfiguration, request: Request):
    '''
    Example Usage
        curl -X PUT http://localhost:8001/error_configuration -d '{"get_new_model": "True"}' -H  "Content-Type: application/json"
    '''
    print(f'setting error configuration to: {asdict(error_configuration)}')
    trainer_logic = trainer_node_from_request(request).trainer_logic
    assert isinstance(trainer_logic, MockTrainerLogic)
    trainer_logic.error_configuration = error_configuration


@router.post("/steps")
async def add_steps(request: Request):
    trainer_node = trainer_node_from_request(request)

    assert isinstance(trainer_node.trainer_logic, MockTrainerLogic)

    if not trainer_node.trainer_logic._executor or not trainer_node.trainer_logic._executor.is_process_running():  # pylint: disable=protected-access
        logging.error(
            f'cannot add steps when training is not running, state:  { trainer_node.trainer_logic._training.training_state}')
        raise HTTPException(status_code=409, detail="trainer is not running")

    steps = int(str(await request.body(), 'utf-8'))
    previous_state = trainer_node.trainer_logic.provide_new_model
    trainer_node.trainer_logic.provide_new_model = True
    print(f'simulating newly completed models by moving {steps} forward', flush=True)
    for i in range(0, steps):
        try:
            await trainer_node.trainer_logic.sync_confusion_matrix(trainer_node.uuid, trainer_node._sio_client)
        except Exception:
            # Tests can force synchroniation to fail, error state is reported to backend
            pass
    trainer_node.trainer_logic.provide_new_model = previous_state
    await trainer_node.send_status()


@router.post("/kill_training_process")
async def kill_process(request: Request):
    # pylint: disable=protected-access
    trainer_node = trainer_node_from_request(request)
    if not trainer_node.trainer_logic._executor or not trainer_node.trainer_logic._executor.is_process_running():
        raise HTTPException(status_code=409, detail="trainer is not running")
    trainer_node.trainer_logic._executor.stop()


@router.post("/force_status_update")
async def force_status_update(request: Request):
    trainer_node = trainer_node_from_request(request)
    await trainer_node.send_status()


def trainer_node_from_request(request: Request) -> TrainerNode:
    return request.app
