from typing import Optional
from pydantic.main import BaseModel


class ErrorConfiguration(BaseModel):
    begin_training: Optional[bool] = False
    save_model: Optional[bool] = False
    get_new_model: Optional[bool] = False
    crash_training: Optional[bool] = False
