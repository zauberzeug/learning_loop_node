import asyncio
import functools
import logging
import sys
import traceback
from dataclasses import dataclass
from typing import Any, Optional

from data_classes import asdict

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


def ensure_socket_response(func):
    @functools.wraps(func)
    async def wrapper_ensure_socket_response(*args, **kwargs):
        try:
            if asyncio.iscoroutinefunction(func):
                value = await func(*args, **kwargs)
            else:
                value = func(*args, **kwargs)

            if isinstance(value, str):
                return asdict(SocketResponse.for_success(value))
            elif isinstance(value, bool):
                return asdict(SocketResponse.from_bool(value))
            elif isinstance(value, SocketResponse):
                return value
            elif (args[0] in ['connect', 'disconnect', 'connect_error']):
                return value
            elif value is None:
                return None
            else:
                raise Exception(
                    f"Return type for sio must be str, bool, SocketResponse or None', but was {type(value)}'"
                )
        except Exception as e:
            trace = ''.join(traceback.format_stack())
            logging.error(
                f'\nAn error occured for {args[0]}:  \
                \nStacktrace: \
                \n{trace} \
                \nError: \
                \n {str(e)} \n'
            )

            return asdict(SocketResponse.for_failure(str(e)))

    return wrapper_ensure_socket_response
