from pydantic.main import BaseModel


class Context(BaseModel):
    project: str
    organization: str
