import sys
from dataclasses import dataclass
from typing import Any, Optional

KWONLY_SLOTS = {'kw_only': True, 'slots': True} if sys.version_info >= (3, 10) else {}


@dataclass(**KWONLY_SLOTS)
class SocketResponse():
    success: bool
    error_msg: Optional[str] = None
    payload: Optional[Any] = None

    @staticmethod
    def for_failure(error_msg: str):
        return SocketResponse(success=False, error_msg=error_msg)

    @staticmethod
    def for_success(payload: Optional[str] = ''):
        return SocketResponse(success=True, payload=payload)

    @staticmethod
    def from_bool(value: bool):
        return SocketResponse(success=value)
