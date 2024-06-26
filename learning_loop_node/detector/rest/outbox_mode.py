
from fastapi import APIRouter, HTTPException, Response

router = APIRouter()


@router.get("/outbox_mode")
async def get_outbox_mode(request):
    outbox = request.app.outbox
    return Response(content=str(outbox.get_mode()))


@router.put("/outbox_mode")
async def put_outbox_mode(request):
    outbox = request.app.outbox
    content = str(await request.body(), 'utf-8')
    try:
        outbox.set_mode(content)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    return Response(status_code=200)
