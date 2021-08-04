from pydantic.main import BaseModel
from ..context import Context
from typing import List

class Category(BaseModel):
    id : str
    name : str

class ModelInformation(BaseModel):
    organization: str
    project: str
    version: str
    model_id: str
    project_categories: List[Category]

    @property
    def context(self):
        return Context(organization=self.organization, project=self.project)