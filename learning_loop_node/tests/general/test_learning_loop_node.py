from typing import List

import pytest

from ...helpers.misc import create_resource_paths

# Used by all Nodes


@pytest.mark.parametrize("image_ids,expected_urls,expected_ids", [
    (['some_id'], ['/zauberzeug/projects/pytest_nodelib_general/images/some_id/main'], ['some_id']),
    (['some_id_1', 'some_id_2'], ['/zauberzeug/projects/pytest_nodelib_general/images/some_id_1/main',
     '/zauberzeug/projects/pytest_nodelib_general/images/some_id_2/main'], ['some_id_1', 'some_id_2']),
    ([], [], [])
])
def test_resource_path_creation(image_ids: List[str], expected_urls: List[str], expected_ids: List['str']):
    urls, ids = create_resource_paths('zauberzeug', 'pytest_nodelib_general', image_ids)

    assert urls == expected_urls
    assert ids == expected_ids
