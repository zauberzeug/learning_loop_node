from learning_loop_node import node_helper
from learning_loop_node.trainer.training_data import TrainingData
import uvicorn
import backdoor_controls
from fastapi_utils.tasks import repeat_every
import results
from typing import List
from learning_loop_node.trainer.trainer import Trainer
from typing import Optional

trainer = Trainer(uuid='85ef1a58-308d-4c80-8931-43d1f752f4f2', name='mocked trainer')


@trainer.begin_training
async def begin_training(data):
    image_data = await node_helper.download_images_data(trainer.url, trainer.headers, trainer.training.organization, trainer.training.project, data['image_ids'])
    trainer.training.data = TrainingData(image_data=image_data, box_categories=data['box_categories'])


@trainer.stop_training
async def stop():
    return True


@trainer.get_model_files
def get_model_files(ogranization: str, project: str, model_id: str) -> List[str]:

    fake_weight_file = '/tmp/weightfile.weights'
    with open(fake_weight_file, 'wb') as f:
        f.write(b'\x42')

    more_data_file = '/tmp/some_more_data.txt'
    with open(more_data_file, 'w') as f:
        f.write('zweiundvierzig')

    return [fake_weight_file, more_data_file]


@trainer.on_event("startup")
@repeat_every(seconds=5, raise_exceptions=True, wait_first=True)
async def step() -> None:
    await _step()


async def _step() -> Optional[dict]:
    """creating new model every 5 seconds for the demo project"""
    if trainer.training and trainer.training.project in ['demo', 'pytest']:
        return await results.increment_time(trainer)


# setting up backdoor_controls
trainer.include_router(backdoor_controls.router, prefix="")

if __name__ == "__main__":
    uvicorn.run("main:trainer", host="0.0.0.0", port=80, lifespan='on', reload=True)
