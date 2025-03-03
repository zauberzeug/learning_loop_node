import asyncio
from typing import Callable, ParamSpec, TypeVar

T = TypeVar('T')
P = ParamSpec('P')


async def io_bound(func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    """Run a blocking function in a thread pool executor.
    This is useful for disk I/O operations that would block the event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))
