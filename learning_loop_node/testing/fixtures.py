"""Pytest fixtures shared by the library's own suite and by node repositories.

Opt in from a `conftest.py`:

    pytest_plugins = ['learning_loop_node.testing.fixtures']

The fixtures are deliberately *not* registered as a `pytest11` entry point: `data_folder` is
autouse and wipes a directory, which no project should get merely by installing the library.
"""

import logging
import os
import shutil

import pytest

from ..globals import GLOBALS

TEST_DATA_FOLDER = '/tmp/learning_loop_lib_data'


@pytest.fixture(autouse=True, scope='session')
def clear_loggers():
    """Remove handlers from all loggers"""
    # see https://github.com/pytest-dev/pytest/issues/5502
    yield

    loggers = [logging.getLogger()] + list(logging.Logger.manager.loggerDict.values())
    for logger in loggers:
        if not isinstance(logger, logging.Logger):
            continue
        handlers = getattr(logger, 'handlers', [])
        for handler in handlers:
            logger.removeHandler(handler)


@pytest.fixture(autouse=True, scope='function')
def data_folder():
    GLOBALS.data_folder = TEST_DATA_FOLDER
    shutil.rmtree(GLOBALS.data_folder, ignore_errors=True)
    os.makedirs(GLOBALS.data_folder, exist_ok=True)
    yield
    shutil.rmtree(GLOBALS.data_folder, ignore_errors=True)
