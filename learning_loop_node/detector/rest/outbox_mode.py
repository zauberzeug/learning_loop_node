
import logging
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import PlainTextResponse


router = APIRouter()


@router.get("/outbox_mode")
async def get_outbox_mode(request: Request):
    '''
    Example Usage
        curl http://localhost/outbox_mode
    '''
    logging.error("\nget_outbox_mode\n")
    outbox = request.app.outbox
    return PlainTextResponse(str(outbox.get_mode()))


@router.put("/outbox_mode")
async def put_outbox_mode(request: Request):
    '''
    Example Usage
        curl -X PUT -d "continuous_upload" http://localhost/outbox_mode
        curl -X PUT -d "stopped" http://localhost/outbox_mode
    '''
    print("put_outbox_mode")
    outbox = request.app.outbox
    content = str(await request.body(), 'utf-8')
    try:
        outbox.set_mode(content)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    return "OK"
