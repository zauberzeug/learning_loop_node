from ...data_classes import SocketResponse
from ...helpers.misc import ensure_socket_response


async def test_a_returned_socket_response_is_sent_as_a_dict():
    @ensure_socket_response
    async def handler(event: str):
        return SocketResponse.for_failure('refused')

    assert await handler('some_event') == {'success': False, 'error_msg': 'refused', 'payload': None}
