import asyncio

from tqdm import tqdm

from learning_loop_node.loop_communication import LoopCommunicator

glc = LoopCommunicator()

glc.organization = 'zauberzeug'
glc.project = 'demo'

data = glc.get_json('/data')
categories = data['categories']
image_ids = data['image_ids'][0:1]  # use only one image for testing
chunk_size = 50

images_data = []
for i in tqdm(range(0, len(image_ids), chunk_size), position=0, leave=True):
    chunk_ids = image_ids[i:i+chunk_size]
    chunk_data = glc.get_json(f'/images?ids={",".join(chunk_ids)}&model_version=1.1')
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


glc.put_json('/images/novelty_scores', json=novelty_scores)

loop = asyncio.get_event_loop()
loop.run_until_complete(glc.shutdown())
