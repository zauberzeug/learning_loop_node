from pydantic.main import BaseModel
from typing import List, Optional
from learning_loop_node.context import Context
from learning_loop_node.data_classes.category import Category


class ModelInformation(BaseModel):
    id: str
    host: str
    organization: str
    project: str
    version: str
    categories: List[Category]
    resolution: Optional[int]

    @property
    def context(self):
        return Context(organization=self.organization, project=self.project)
