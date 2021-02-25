import pytest
from typing import Generator
from requests import Session
from urllib.parse import urljoin
import darknet_tests.test_helper as test_helper


@pytest.fixture()
def web() -> Generator:
    with test_helper.LiveServerSession() as c:
        yield c
