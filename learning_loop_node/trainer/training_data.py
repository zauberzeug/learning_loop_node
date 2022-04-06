from typing import Dict, Optional
from learning_loop_node.data_classes.category import Category
from learning_loop_node.trainer.hyperparameter import Hyperparameter
from pydantic import BaseModel
from typing import List


class TrainingData(BaseModel):
    image_data: List[dict] = []
    skipped_image_count: Optional[int] = 0
    categories: List[Category] = []
    hyperparameter: Optional[Hyperparameter]

    def image_ids(self):
        return [image['id'] for image in self.image_data]

    def train_image_count(self):
        return len([image for image in self.image_data if image['set'] == 'train'])

    def test_image_count(self):
        return len([image for image in self.image_data if image['set'] == 'test'])
