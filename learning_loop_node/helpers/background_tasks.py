# Copy of Nicegui background_tasks.py
# MIT License

# Copyright (c) 2021 Zauberzeug GmbH

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import asyncio
import logging
from typing import Coroutine, Dict, Set

running_tasks: Set[asyncio.Task] = set()
lazy_tasks_running: Dict[str, asyncio.Task] = {}
lazy_tasks_waiting: Dict[str, Coroutine] = {}


def create(coroutine: Coroutine, *, name: str = 'unnamed task') -> asyncio.Task:
    """Wraps a loop.create_task call and ensures there is an exception handler added to the task.

    If the task raises an exception, it is logged and handled by the global exception handlers.
    Also a reference to the task is kept until it is done, so that the task is not garbage collected mid-execution.
    See https://docs.python.org/3/library/asyncio-task.html#asyncio.create_task.
    """
    loop = asyncio.get_event_loop()
    coroutine = coroutine if asyncio.iscoroutine(coroutine) else asyncio.wait_for(coroutine, None)
    task: asyncio.Task = loop.create_task(coroutine, name=name)
    task.add_done_callback(_handle_task_result)
    running_tasks.add(task)
    task.add_done_callback(running_tasks.discard)
    return task


def create_lazy(coroutine: Coroutine, *, name: str) -> None:
    """Wraps a create call and ensures a second task with the same name is delayed until the first one is done.

    If a third task with the same name is created while the first one is still running, the second one is discarded.
    """
    if name in lazy_tasks_running:
        if name in lazy_tasks_waiting:
            asyncio.Task(lazy_tasks_waiting[name]).cancel()
        lazy_tasks_waiting[name] = coroutine
        return

    def finalize(name: str) -> None:
        lazy_tasks_running.pop(name)
        if name in lazy_tasks_waiting:
            create_lazy(lazy_tasks_waiting.pop(name), name=name)
    task = create(coroutine, name=name)
    lazy_tasks_running[name] = task
    task.add_done_callback(lambda _: finalize(name))


def _handle_task_result(task: asyncio.Task) -> None:
    try:
        task.result()
    except asyncio.CancelledError:
        pass
    except Exception:
        logging.exception('Background task %s raised an exception', task.get_name())
