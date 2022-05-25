from learning_loop_node.trainer.training_data import TrainingData
from pydantic import BaseModel
from typing import Optional
from learning_loop_node.context import Context


class Training(BaseModel):
    base_model_id: Optional[str]
    id: str
    context: Context

    project_folder: str
    images_folder: str

    training_folder: Optional[str]

    data: Optional[TrainingData]

    training_number: Optional[int]


class TrainingOut(BaseModel):
    confusion_matrix: Optional[dict]
    train_image_count: Optional[int]
    test_image_count: Optional[int]
    trainer_id: Optional[str]
    hyperparameters: Optional[str]
