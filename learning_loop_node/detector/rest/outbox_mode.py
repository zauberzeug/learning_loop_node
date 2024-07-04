from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import PlainTextResponse

from ..outbox import Outbox

router = APIRouter()


@router.get("/outbox_mode")
async def get_outbox_mode(request: Request):
    '''
    Example Usage
        curl http://localhost/outbox_mode
    '''
    outbox: Outbox = request.app.outbox
    return PlainTextResponse(outbox.get_mode().value)


@router.put("/outbox_mode")
async def put_outbox_mode(request: Request):
    '''
    Example Usage
        curl -X PUT -d "continuous_upload" http://localhost/outbox_mode
        curl -X PUT -d "stopped" http://localhost/outbox_mode
    '''
    outbox: Outbox = request.app.outbox
    content = str(await request.body(), 'utf-8')
    try:
        await outbox.set_mode(content)
    except TimeoutError as e:
        raise HTTPException(202, 'Setting has not completed, yet: ' + str(e)) from e
    except ValueError as e:
        raise HTTPException(422, 'Could not set outbox mode: ' + str(e)) from e

    return "OK"
