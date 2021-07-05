from learning_loop_node import DetectorNode
import pytest

@pytest.mark.asyncio
async def test_connect_without_credentials():
    node = DetectorNode(uuid='4ad7750b-4f0c-4d8d-86c6-c5ad04e19d2c', name='test detector')
    await node.connect() # should be able to connect without username/password
