from fastapi import APIRouter, Request, File, UploadFile, Header
from typing import Optional
import numpy as np
from fastapi.responses import JSONResponse
from icecream import ic
from ...inbox_filter import relevance_filter

router = APIRouter()


@router.post("/detect")
async def http_detect(
    request: Request,
    file: UploadFile = File(...),
    camera_id: Optional[str] = Header(None),
    mac: Optional[str] = Header(None),
    tags: Optional[str] = Header(None),
    autoupload: Optional[str] = Header(None),
):
    """
    Example Usage

        curl --request POST -F 'file=@test.jpg' localhost:8004/detect

        for i in `seq 1 10`; do time curl --request POST -F 'file=@test.jpg' localhost:8004/detect; done

        You can additionally provide the following camera parameters:
          - `autoupload`: configures auto-submission to the learning loop; `filtered` (default), `all`, `disabled` (example curl parameter `-H 'autoupload: all'`)
          - `camera-id`: a string which groups images for submission together (example curl parameter `-H 'camera-id: front_cam'`)
    """
    try:
        np_image = np.fromfile(file.file, np.uint8)
    except:
        raise Exception(f'Uploaded file {file.filename} is no image file.')
    detections = await request.app.get_detections(
        raw_image=np_image,
        camera_id=camera_id or mac or None,
        tags=tags.split(',') if tags else [],
        autoupload=autoupload,
    )
    return JSONResponse(detections)
