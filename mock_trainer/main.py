import backdoor_controls
import uvicorn
from learning_loop_node.trainer.capability import Capability
from learning_loop_node.trainer.trainer_node import TrainerNode
from mock_trainer import MockTrainer
import os
import logging

logging.basicConfig(level=logging.DEBUG)


mock_trainer = MockTrainer(capability=Capability.Box, model_format='mocked')
trainer_node = TrainerNode(name='mocked trainer', trainer=mock_trainer)


# setting up backdoor_controls
trainer_node.include_router(backdoor_controls.router, prefix="")

if __name__ == "__main__":
    reload_dirs = ['./restart'] if os.environ.get('MANUAL_RESTART', None) else ['./', './learning-loop-node', '/usr/local/lib/python3.7/site-packages/learning_loop_node']
    uvicorn.run("main:trainer_node", host="0.0.0.0", port=80, lifespan='on', reload=True, reload_dirs=reload_dirs)
