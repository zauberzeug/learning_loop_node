import pytest
from learning_loop_node import __version__
from learning_loop_node import node_helper
from typing import List


def test_version():
    assert __version__ == '0.1.0'


@pytest.mark.parametrize("image_ids,expected_urls,expected_ids", [
    (['some_id'], ['http://localhost/api/zauberzeug/projects/pytest/images/some_id/main'], ['some_id']),
    (['some_id_1', 'some_id_2'], ['http://localhost/api/zauberzeug/projects/pytest/images/some_id_1/main',
     'http://localhost/api/zauberzeug/projects/pytest/images/some_id_2/main'], ['some_id_1', 'some_id_2']),
    ([], [], [])
])
def test_resource_url_creation(image_ids: List[str], expected_urls: List[str], expected_ids: List['str']):
    urls, ids = node_helper.create_resource_urls('http://localhost', 'zauberzeug', 'pytest', image_ids)

    assert urls == expected_urls
    assert ids == expected_ids
