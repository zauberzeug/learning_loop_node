import pytest
from learning_loop_node.loop import loop


@pytest.mark.asyncio
async def test_connect_headers():
    headers = loop.get_headers()
    assert 'node_type' in headers
    assert 'organization' in headers
    assert 'project' in headers
