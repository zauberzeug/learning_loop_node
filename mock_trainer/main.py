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
from typing import List

node = Node(uuid='85ef1a58-308d-4c80-8931-43d1f752f4f2', name='mocked trainer')


@node.begin_training
async def begin_training(data):
    node.status.box_categories = data['box_categories']
    node.status.train_images = [i for i in data['images'] if i['set'] == 'train']
    node.status.test_images = [i for i in data['images'] if i['set'] == 'test']


@node.stop_training
def stop():
    # nothing to do for the mock trainer
    pass


@node.get_model_files
def get_model_files(ogranization: str, project: str, model_id: str) -> List[str]:

    fake_weight_file = '/tmp/weightfile.weights'
    with open(fake_weight_file, 'wb') as f:
        f.write(b'\x42')

    more_data_file = '/tmp/some_more_data.txt'
    with open(more_data_file, 'w') as f:
        f.write('zweiundvierzig')

    return [fake_weight_file, more_data_file]


@node.on_event("startup")
@repeat_every(seconds=5, raise_exceptions=True, wait_first=True)
async def step() -> None:
    """creating new model every 5 seconds for the demo project"""
    if node.status.model and node.status.project == 'demo':
        await results.increment_time(node)

# setting up backdoor_controls
node.include_router(backdoor_controls.router, prefix="")
