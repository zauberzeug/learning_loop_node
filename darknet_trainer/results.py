import uuid
import random
from learning_loop_node.node import Node, State


async def increment_time(node: Node):
    if node.status.state != State.Running or getattr(node.status, 'box_categories') is None:
        return

    node.status.uptime = node.status.uptime + 5
    print('---- time', node.status.uptime, flush=True)
    confusion_matrix = {}
    for category in node.status.box_categories:
        try:
            minimum = node.status.model['confusion_matrix'][category['id']]['tp']
        except:
            minimum = 0
        maximum = minimum + 1
        confusion_matrix[category['id']] = {
            'tp': random.randint(minimum, maximum),
            'fp': max(random.randint(10-maximum, 10-minimum), 2),
            'fn': max(random.randint(10-maximum, 10-minimum), 2),
        }
    new_model = {
        'id': str(uuid.uuid4()),
        'hyperparameters': node.status.hyperparameters,
        'confusion_matrix': confusion_matrix,
        'parent_id': node.status.model['id'],
        'train_image_count': len(node.status.train_images),
        'test_image_count': len(node.status.test_images),
        'trainer_id': node.status.id,
    }

    result = await node.sio.call('update_model', (node.status.organization, node.status.project, new_model))
    if result != True:
        raise Exception('could not update model: ' + str(result))
    node.status.model = new_model
    await node.update_state(State.Running)
