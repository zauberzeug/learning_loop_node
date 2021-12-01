
import asyncio
import functools
import logging
from learning_loop_node.socket_response import SocketResponse
from icecream import ic


def try_except(func):
    @functools.wraps(func)
    async def wrapper_handle_error(*args, **kwargs):
        try:
            if asyncio.iscoroutinefunction(func):
                value = await func(*args, **kwargs)
            else:
                value = func(*args, **kwargs)
            if isinstance(value, str):
                return SocketResponse.for_success(value).__dict__
                
            return value
        except Exception as e:
            logging.error(f'An error occured for {func.__name__}: {str(e)}')
            return SocketResponse.for_failure(str(e)).__dict__
            
    return wrapper_handle_error

