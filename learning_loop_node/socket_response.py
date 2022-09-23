from typing import Any, Optional
from pydantic import BaseModel
import asyncio
import functools
import logging
import traceback


class SocketResponse(BaseModel):
    success: bool
    error_msg: Optional[str]
    payload: Optional[Any]

    @staticmethod
    def for_failure(error_msg: str):
        return SocketResponse(success=False, error_msg=error_msg)

    @staticmethod
    def for_success(payload: Optional[str] = ''):
        return SocketResponse(success=True, payload=payload)

    @staticmethod
    def from_bool(value: bool):
        return SocketResponse(success=value)

    @staticmethod
    def from_dict(value: dict):
        try:
            return SocketResponse.parse_obj(value)
        except:
            logging.exception(f'Error parsing SocketResponse: value : {value}')
            raise


def ensure_socket_response(func):
    @functools.wraps(func)
    async def wrapper_ensure_socket_response(*args, **kwargs):
        try:
            if asyncio.iscoroutinefunction(func):
                value = await func(*args, **kwargs)
            else:
                value = func(*args, **kwargs)

            if isinstance(value, str):
                return SocketResponse.for_success(value).__dict__
            elif isinstance(value, bool):
                return SocketResponse.from_bool(value).__dict__
            elif isinstance(value, SocketResponse):
                return value
            elif (args[0] in ['connect', 'disconnect', 'connect_error']):
                return value
            else:
                raise Exception(f"Returntype for sio must be str or bool or SocketResponse', but was {type(value)}'")
        except Exception:
            error = traceback.print_exc()
            trace = ''.join(traceback.format_stack())
            logging.error(
                f'\nAn error occured for {args[0]}:  \
                \nStacktrace: \
                \n{trace} \
                \nError: \
                \n {str(error)} \n'
            )

            return SocketResponse.for_failure(str(error)).__dict__

    return wrapper_ensure_socket_response
