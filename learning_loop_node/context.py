from pydantic.main import BaseModel


class Context(BaseModel):
    organization: str
    project: str
