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
from typing import List
import cv2
from glob import glob

hostname = 'backend'
node = Node(hostname, uuid='12d7750b-4f0c-4d8d-86c6-c5ad04e19d57', name='detection node')


@node.get_model_files
def get_model_files(ogranization: str, project: str, model_id: str) -> List[str]:
    return _get_model_files


def _get_model_files() -> List[str]:
    files = glob('/data/*')
    return files


# setting up backdoor_controls
node.include_router(backdoor_controls.router, prefix="")
