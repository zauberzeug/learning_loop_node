import backdoor_controls
from custom_formatter import CustomFormatter
import uvicorn
from learning_loop_node.trainer.trainer_node import TrainerNode
from mock_trainer import MockTrainer
import os
import logging
import icecream
icecream.install()

logging.basicConfig(level=logging.DEBUG)
logging.getLogger().handlers[0].setFormatter(CustomFormatter())

mock_trainer = MockTrainer(model_format='mocked')
trainer_node = TrainerNode(uuid='85ef1a58-308d-4c80-8931-43d1f752f4f2', name='mocked trainer', trainer=mock_trainer)


# setting up backdoor_controls
trainer_node.include_router(backdoor_controls.router, prefix="")

if __name__ == "__main__":
    reload_dirs = ['./restart'] if os.environ.get('MANUAL_RESTART', None) \
        else ['./', './learning-loop-node', '/usr/local/lib/python3.7/site-packages/learning_loop_node']
    uvicorn.run("main:trainer_node", host="0.0.0.0", port=80, lifespan='on',
                reload=True, use_colors=True, reload_dirs=reload_dirs)
