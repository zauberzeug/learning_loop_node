import pytest
import os
import shutil
import requests
from main import data_dir


@pytest.fixture(autouse=True)
def delete_data_dir():
    os.makedirs(data_dir, exist_ok=True)
    yield
    shutil.rmtree(data_dir, ignore_errors=True)


@pytest.fixture(autouse=True, scope='session')
def check_for_model_data():
    assert os.path.exists(
        '/model/some_weightfile.weights'), "Error: Could not find weightfile. Need to execute 'detection_node % ./download_model_for_testing.sh'"
