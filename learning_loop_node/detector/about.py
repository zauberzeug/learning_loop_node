from pydantic.main import BaseModel
from ..context import Context
from typing import List

class Category(BaseModel):
    id : str
    name : str

class About(BaseModel):
    host: str
    organization: str
    project: str
    version: str
    categories: List[Category]
    resolution: int
    
    @property
    def context(self):
        return Context(organization=self.organization, project=self.project)