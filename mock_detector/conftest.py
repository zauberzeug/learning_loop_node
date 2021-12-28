from learning_loop_node.globals import GLOBALS
import shutil
import pytest
import logging
logging.basicConfig(level=logging.INFO)


@pytest.fixture(autouse=True, scope='function')
def data_folder():
    GLOBALS.data_folder = '/tmp/learning_loop_lib_data'
    shutil.rmtree(GLOBALS.data_folder, ignore_errors=True)
    yield
    shutil.rmtree(GLOBALS.data_folder, ignore_errors=True)
