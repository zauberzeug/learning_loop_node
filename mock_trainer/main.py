import sys
import asyncio
import threading
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import status
from threading import Thread
import backdoor_controls
from fastapi_utils.tasks import repeat_every
import simplejson as json
import requests
import io
from learning_loop_node.node import Node

node = Node()
hostname = 'backend'


@node.on_event("startup")
@repeat_every(seconds=5, raise_exceptions=True, wait_first=True)
async def step() -> None:
    if status.status.model and status.status.model['context']['project'] == 'demo':
        await status.increment_time(node.sio)


@node.sio.on('run')
async def run(source_model):
    print('---- running training with source model', source_model, flush=True)

    status.status.model = json.loads(source_model)
    context = status.status.model['context']

    data = requests.get(
        f'http://{hostname}/api/{context["organization"]}/projects/{context["project"]}/data?state=complete&mode=boxes').json()
    status.status.box_categories = data['box_categories']
    status.status.train_images = [
        i for i in data['images'] if i['set'] == 'train']
    status.status.test_images = [
        i for i in data['images'] if i['set'] == 'test']
    await status.update_state(node.sio, status.State.Running)
    return True


@node.sio.on('stop')
async def stop():
    print('---- stopping', flush=True)
    await status.update_state(node.sio, status.State.Idle)
    return True


@node.sio.on('save')
async def save(model):
    print('---- saving model', model['id'], flush=True)
    fake_weight_file = open('/tmp/fake_weight_file', 'wb+')
    fake_weight_file.write(
        b"\x00\x00\x00\x00\x00\x00\x00\x00\x01\x01\x01\x01\x01\x01")
    context = model['context']
    response = requests.put(
        f'http://{hostname}/api/{context["organization"]}/projects/{context["project"]}/models/{model["id"]}/file', files={'data': fake_weight_file}
    )
    if response.status_code == 200:
        return True
    else:
        return response.json()['detail']
        return False


@node.on_event("startup")
async def startup():
    print('startup', flush=True)
    await connect()


@node.on_event("shutdown")
async def shutdown():
    print('shutting down', flush=True)
    await node.sio.disconnect()


@node.sio.on('connect')
async def on_connect():
    if status.status.id:
        await status.update_state(node.sio, status.State.Idle)


@node.sio.on('disconnect')
async def on_disconnect():
    await status.update_state(node.sio, status.State.Offline)


async def connect():
    await node.sio.disconnect()
    print('connecting to Learning Loop', flush=True)

    try:
        await node.sio.connect(f"ws://{hostname}", socketio_path="/ws/socket.io")
        print('my sid is', node.sio.sid, flush=True)
    except:
        await asyncio.sleep(0.2)
        await connect()
    print('connected to Learning Loop', flush=True)

# setting up backdoor_controls
node.connect = connect
node.include_router(backdoor_controls.router, prefix="")
