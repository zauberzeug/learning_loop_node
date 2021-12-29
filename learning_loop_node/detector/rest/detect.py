from fastapi import APIRouter, Request, File, UploadFile, Header
from typing import Optional
import numpy as np
from fastapi.responses import JSONResponse
from icecream import ic


router = APIRouter()


@router.post("/detect")
async def http_detect(request: Request, file: UploadFile = File(...), mac: str = Header(...), tags: Optional[str] = Header(None)):
    """
    Example Usage

        curl --request POST -H 'mac: FF:FF' -F 'file=@test.jpg' localhost:8004/detect

        for i in `seq 1 10`; do time curl --request POST -H 'mac: FF:FF' -F 'file=@test.jpg' localhost:8004/detect; done
    """
    try:
        np_image = np.fromfile(file.file, np.uint8)
    except:
        raise Exception(f'Uploaded file {file.filename} is no image file.')
    detections = await request.app.get_detections(np_image, mac, tags.split(',') if tags else [])
    return JSONResponse(detections)
