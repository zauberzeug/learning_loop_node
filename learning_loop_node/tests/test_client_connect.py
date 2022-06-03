import pytest
from learning_loop_node.loop import loop


@pytest.mark.asyncio
async def test_connect_headers():
    headers = loop.get_headers()
    assert headers['node_type'] == 'trainer'
    assert 'organization' in headers
    assert 'project' in headers
