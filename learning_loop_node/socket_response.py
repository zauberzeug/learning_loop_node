from typing import Optional
from pydantic import BaseModel

class SocketResponse(BaseModel):
    success : bool
    error_msg : Optional[str]
    payload: Optional[str]
   
    @staticmethod
    def failure(error_msg: str):
        return SocketResponse(success = False, error_msg = error_msg)
    
    