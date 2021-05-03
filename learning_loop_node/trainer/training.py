from learning_loop_node.trainer.training_data import TrainingData
from pydantic import BaseModel
from typing import Optional
from learning_loop_node.trainer.model import Model
from learning_loop_node.context import Context


class Training(BaseModel):
    id: str
    base_model: Model
    context: Context

    project_folder: str
    images_folder: str

    training_folder: Optional[str]

    data: Optional[TrainingData]
    last_produced_model: Optional[Model]
    last_published_iteration: Optional[int]

    @property
    def last_known_model(self):
        if self.last_produced_model:
            return self.last_produced_model
        else:
            return self.base_model
