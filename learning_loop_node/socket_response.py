from typing import Optional
from pydantic import BaseModel

class SocketResponse(BaseModel):
    success : bool
    error_msg : Optional[str]
    payload: Optional[str]
   
    @staticmethod
    def for_failure(error_msg: str):
        return SocketResponse(success = False, error_msg = error_msg)

    @staticmethod
    def for_success(payload : Optional[str] = ''):
        return SocketResponse(success = True, payload = payload)
    
    