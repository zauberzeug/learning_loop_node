import backdoor_controls
import uvicorn
from learning_loop_node.trainer.capability import Capability
from learning_loop_node.trainer.trainer_node import TrainerNode
from mock_trainer import MockTrainer


mock_trainer = MockTrainer(capability=Capability.Box, model_format='mocked')
trainer_node = TrainerNode(uuid='85ef1a58-308d-4c80-8931-43d1f752f4f2', name='mocked trainer', trainer=mock_trainer)


# setting up backdoor_controls
trainer_node.include_router(backdoor_controls.router, prefix="")

if __name__ == "__main__":
    uvicorn.run("main:trainer_node", host="0.0.0.0", port=80, lifespan='on', reload=True, reload_dirs=['./restart'])
