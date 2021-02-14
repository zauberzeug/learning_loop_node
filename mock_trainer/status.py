from typing import List, Optional
from pydantic import BaseModel
from enum import Enum
import uuid
import random
import asyncio


class State(str, Enum):
    Idle = "idle"
    Running = "running"
    Offline = "offline"


class Status(BaseModel):
    id: str
    name: str
    state: Optional[State]
    uptime: Optional[int] = 0
    model: Optional[dict]
    hyperparameters: Optional[str]
    box_categories: Optional[dict]
    train_images: Optional[List[dict]]
    test_images: Optional[List[dict]]


async def update_state(sio, state: State):
    global status
    status.state = state
    if status.state != State.Offline:
        await send_status(sio)


async def update_status(sio, new_status: Status):
    global status
    status.id = new_status.id
    status.name = new_status.name
    status.uptime = new_status.uptime
    status.model = new_status.model
    status.hyperparameters = new_status.hyperparameters
    status.box_categories = new_status.box_categories
    status.train_images = new_status.train_images
    status.test_images = new_status.test_images

    if status.state != State.Offline:
        status.state = State.Idle
        await send_status(sio)


async def send_status(sio):
    content = status.dict()
    if status.model:
        content['latest_produced_model_id'] = status.model['id']
    del content['model']
    await sio.call('update_trainer', content)


async def increment_time(sio):
    global status
    if status.state != State.Running or getattr(status, 'box_categories') is None:
        return

    status.uptime = status.uptime + 5
    print('---- time', status.uptime, flush=True)
    confusion_matrix = {}
    for category in status.box_categories:
        try:
            minimum = status.model['confusion_matrix'][category['id']]['tp']
        except:
            minimum = 0
        maximum = minimum + 1
        confusion_matrix[category['id']] = {
            'tp': random.randint(minimum, maximum),
            'tn': random.randint(minimum, maximum),
            'fp': max(random.randint(10-maximum, 10-minimum), 2),
            'fn': max(random.randint(10-maximum, 10-minimum), 2),
        }
    new_model = {
        'id': str(uuid.uuid4()),
        'hyperparameters': status.hyperparameters,
        'confusion_matrix': confusion_matrix,
        'parent_id': status.model['id'],
        'context': status.model['context'],
        'train_image_count': len(status.train_images),
        'test_image_count': len(status.test_images),
        'trainer_id': status.id,
    }

    await sio.call('update_model', new_model)
    status.model = new_model
    await update_state(sio, State.Running)


status = Status(id='85ef1a58-308d-4c80-8931-43d1f752f4f2', name='mocked trainer',  state=State.Offline)
