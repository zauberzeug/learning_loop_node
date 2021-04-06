import pytest
import os
import shutil
import requests


@pytest.fixture(autouse=True)
def delete_data_dir():
    os.makedirs('/data', exist_ok=True)
    yield
    shutil.rmtree('/data', ignore_errors=True)
