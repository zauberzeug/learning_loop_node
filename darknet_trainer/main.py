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
node = Node(hostname, uuid='c34dc41f-9b76-4aa9-8b8d-9d27e33a19e4', name='darknet trainer')


@node.begin_training
def begin_training(data):
    print('################ begin_training START', flush=True)
    resources = [i['resource'] for i in data['images']]
    print(resources, flush=True)
    for resource in resources:
        url = f'http://{node.hostname}/api{resource}'
        print('URL: ', url, flush=True)
        response = requests.get(url)
        print('RESPONSE: ', response, flush=True)
        with open('img.jpg', 'wb') as f:
            f.write(response.content)

    print('################ begin_training END', flush=True)
    return True


@node.stop_training
def stop():
    # nothing to do for the darknet trainer
    pass


@node.get_weightfile
def get_weightfile(organization: str, project: str, model_id: str) -> io.BufferedRandom:
    fake_weight_file = open('/tmp/fake_weight_file', 'wb+')
    fake_weight_file.write(b"\x00\x00\x00\x00\x00\x00\x00\x00\x01\x01\x01\x01\x01\x01")
    return fake_weight_file


@node.on_event("startup")
@repeat_every(seconds=5, raise_exceptions=True, wait_first=True)
async def step() -> None:
    """creating new model every 5 seconds for the demo project"""
    if node.status.model and node.status.project == 'pytest':
        await results.increment_time(node)

# setting up backdoor_controls
node.include_router(backdoor_controls.router, prefix="")
