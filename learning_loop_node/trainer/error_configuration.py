from typing import Optional
from pydantic.main import BaseModel


class ErrorConfiguration(BaseModel):
    save_model: Optional[bool] = False
