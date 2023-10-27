import asyncio
import logging
import os
from typing import Dict

from tqdm import tqdm

from learning_loop_node.loop_communication import LoopCommunicator

os.environ['LOOP_HOST'] = 'preview.learning-loop.ai'
os.environ['LOOP_ORGANIZATION'] = 'zauberzeug'
os.environ['LOOP_PROJECT'] = 'demo'

logging.info('Starting novelty_score_updater')
glc = LoopCommunicator()

project_path = f'/{glc.organization}/projects/{glc.project}'


async def get_json_async(path) -> Dict:
    url = f'{project_path}{path}'
    response = await glc.get(url)
    if response.status_code != 200:
        raise Exception('bad response: ' + str(response))
    return response.json()


def get_json(path):
    return asyncio.get_event_loop().run_until_complete(get_json_async(path))


async def put_json_async(path, json) -> Dict:
    response = await glc.async_client.put(path, json=json)
    if response.status_code != 200:
        raise Exception(f'bad response: {str(response)} \n {response.json()}')
    return response.json()


def put_json(path, json):
    return asyncio.get_event_loop().run_until_complete(put_json_async(path, json))


data = get_json(f'{project_path}/data')
categories = data['categories']
image_ids = data['image_ids'][0:1]  # use only one image for testing
chunk_size = 50

images_data = []
for i in tqdm(range(0, len(image_ids), chunk_size), position=0, leave=True):
    chunk_ids = image_ids[i:i+chunk_size]
    chunk_data = get_json(f'{project_path}/images?ids={",".join(chunk_ids)}&model_version=1.1')
    images_data += chunk_data['images']

novelty_scores = {}
for image_data in images_data:
    image_id = image_data['id']
    detections = image_data['box_detections']
    novelty_score = 0
    for d in detections:
        if d['confidence'] > 20 and d['confidence'] < 80:
            novelty_score += 1
    novelty_scores[image_id] = novelty_score


put_json('/images/novelty_scores', json=novelty_scores)

loop = asyncio.get_event_loop()
loop.run_until_complete(glc.shutdown())

logging.info('novelty_score_updater stopped')
