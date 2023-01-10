from learning_loop_node import loop
from icecream import ic
from tqdm import tqdm

loop.organization = 'zauberzeug'
loop.project = 'demo'
data = loop.get_json('/data')
categories = data['categories']
image_ids = data['image_ids'][0:1]  # use only one image for testing
chunk_size = 50

images_data = []
for i in tqdm(range(0, len(image_ids), chunk_size), position=0, leave=True):
    chunk_ids = image_ids[i:i+chunk_size]
    chunk_data = loop.get_json(f'/images?ids={",".join(chunk_ids)}&model_version=1.1')
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


loop.put_json('/images/novelty_scores', json=novelty_scores)
