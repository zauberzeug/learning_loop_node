from typing import Dict, Optional
from learning_loop_node.model_information import ModelInformation
from pydantic import BaseModel
from typing import List


class TrainingData(BaseModel):
    image_data: List[dict] = []
    skipped_image_count: Optional[int] = 0
    base_model: Optional[ModelInformation] = None

    def image_ids(self):
        return [image['id'] for image in self.image_data]

    def train_image_count(self):
        return len([image for image in self.image_data if image['set'] == 'train'])

    def test_image_count(self):
        return len([image for image in self.image_data if image['set'] == 'test'])

    @property
    def categories(self) -> Dict[str, str]:
        return {c.name: c.id for c in self.base_model.categories}
