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


@node.begin_training
def begin_training(data):

    node.status.box_categories = data['box_categories']
    node.status.train_images = [i for i in data['images'] if i['set'] == 'train']
    node.status.test_images = [i for i in data['images'] if i['set'] == 'test']


@node.stop_training
def stop():
    # nothing to do for the mock trainer
    pass


@node.get_weightfile
def get_weightfile(ogranization: str, project: str, model_id: str) -> io.BufferedRandom:
    fake_weight_file = open('/tmp/fake_weight_file', 'wb+')
    fake_weight_file.write(b"\x00\x00\x00\x00\x00\x00\x00\x00\x01\x01\x01\x01\x01\x01")
    return fake_weight_file


@node.on_event("startup")
@repeat_every(seconds=5, raise_exceptions=True, wait_first=True)
async def step() -> None:
    """creating new model every 5 seconds for the demo project"""
    if node.status.model and node.status.project == 'demo':
        await results.increment_time(node)

# setting up backdoor_controls
node.include_router(backdoor_controls.router, prefix="")
