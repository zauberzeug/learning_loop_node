import pytest
import os
import shutil
import requests


@pytest.fixture(autouse=True)
def delete_data_dir():
    os.makedirs('/data', exist_ok=True)
    yield
    shutil.rmtree('/data', ignore_errors=True)

@pytest.fixture(autouse=True, scope='session')
def check_for_model_data():
    assert os.path.exists('/model/some_weightfile.weights'), "Error: Could not find weightfile. Need to execute 'detection_node % ./download_model_for_testing.sh'"