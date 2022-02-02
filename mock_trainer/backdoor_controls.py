"""These restful endpoints are only to be used for testing purposes and are not part of the 'offical' trainer behavior."""
from learning_loop_node.trainer.trainer_node import TrainerNode
from learning_loop_node.trainer.error_configuration import ErrorConfiguration
from fastapi import APIRouter,  Request,  HTTPException
from learning_loop_node.status import Status, State
import logging

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
        if trainer_node.status.state != State.Offline:
            logging.debug('turning socketio off')
            await trainer_node.sio_client.disconnect()
    if state == 'on':
        if trainer_node.status.state == State.Offline:
            logging.debug('turning socketio on')
            await trainer_node.connect()


@router.put("/check_state")
async def check_state(request: Request):
    value = str(await request.body(), 'utf-8')
    trainer_node = trainer_node_from_request(request)
    if value == 'off':
        trainer_node.skip_check_state = True
        trainer_node.status.reset_all_errors()
    if value == 'on':
        trainer_node.skip_check_state = False
    logging.debug(f'turning automatically check_state {value}')


@router.post("/reset")
async def reset(request: Request):
    trainer_node = trainer_node_from_request(request)
    await _switch_socketio('on', trainer_node)
    if trainer_node.status.state == State.Running:
        await trainer_node.stop_training()

    trainer_node.status.reset_all_errors()


@router.put("/error_configuration")
def set_error_configuration(error_configuration: ErrorConfiguration, request: Request):
    '''
    Example Usage
        curl -X PUT -d '{"save_model": "True"}' http://localhost:8001/error_configuration
    '''
    print(f'setting error configuration to: {error_configuration.json()}')
    trainer_node_from_request(request).trainer.error_configuration = error_configuration


@router.post("/steps")
async def add_steps(request: Request):
    trainer_node = trainer_node_from_request(request)
    if trainer_node.status.state != State.Running:
        raise HTTPException(status_code=409, detail="trainer is not running")

    steps = int(str(await request.body(), 'utf-8'))
    print(f'simulating newly completed models by moving {steps} forward', flush=True)
    for i in range(0, steps):
        await trainer_node.check_state()


@router.post("/kill_training_process")
async def kill_process(request: Request):
    trainer_node = trainer_node_from_request(request)
    if trainer_node.status.state != State.Running:
        raise HTTPException(status_code=409, detail="trainer is not running")
    trainer_node.trainer.executor.stop()


def trainer_node_from_request(request: Request) -> TrainerNode:
    return request.app
