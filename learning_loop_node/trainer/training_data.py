from typing import Optional
from pydantic import BaseModel
from typing import List


class TrainingData(BaseModel):
    image_data: List[dict]
    categories: dict
    skipped_image_count: Optional[int] = 0

    def image_ids(self):
        return [image['id'] for image in self.image_data]

    def train_image_count(self):
        return len([image for image in self.image_data if image['set'] == 'train'])

    def test_image_count(self):
        return len([image for image in self.image_data if image['set'] == 'test'])
