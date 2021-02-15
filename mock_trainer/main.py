import sys
import asyncio
import threading
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from threading import Thread
import backdoor_controls
from fastapi_utils.tasks import repeat_every
import simplejson as json
import requests
import io
from learning_loop_node.node import Node, State
import results

hostname = 'backend'
node = Node(hostname, uuid='85ef1a58-308d-4c80-8931-43d1f752f4f2', name='mocked trainer')


@node.on_event("startup")
@repeat_every(seconds=5, raise_exceptions=True, wait_first=True)
async def step() -> None:
    """creating new model every 5 seconds for the demo project"""
    if node.status.model and node.status.model['context']['project'] == 'demo':
        await results.increment_time(node)


@node.sio.on('run')
async def run(source_model):
    print('---- running training with source model', source_model, flush=True)

    node.status.model = json.loads(source_model)
    context = node.status.model['context']

    data = requests.get(
        f'http://{hostname}/api/{context["organization"]}/projects/{context["project"]}/data?state=complete&mode=boxes').json()
    node.status.box_categories = data['box_categories']
    node.status.train_images = [
        i for i in data['images'] if i['set'] == 'train']
    node.status.test_images = [
        i for i in data['images'] if i['set'] == 'test']
    await node.update_state(State.Running)
    return True


@node.sio.on('stop')
async def stop():
    print('---- stopping', flush=True)
    await node.update_state(State.Idle)
    return True


@node.get_weightfile
def get_weightfile(model) -> io.BufferedRandom:
    fake_weight_file = open('/tmp/fake_weight_file', 'wb+')
    fake_weight_file.write(b"\x00\x00\x00\x00\x00\x00\x00\x00\x01\x01\x01\x01\x01\x01")
    return fake_weight_file


# setting up backdoor_controls
node.include_router(backdoor_controls.router, prefix="")
