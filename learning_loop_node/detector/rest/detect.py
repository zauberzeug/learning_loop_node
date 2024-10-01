import logging
from typing import TYPE_CHECKING, Optional

import numpy as np
from fastapi import APIRouter, File, Header, Request, UploadFile
from fastapi.responses import JSONResponse

if TYPE_CHECKING:
    from ..detector_node import DetectorNode


router = APIRouter()


@router.post("/detect")
async def http_detect(
    request: Request,
    file: UploadFile = File(...),
    camera_id: Optional[str] = Header(None),
    mac: Optional[str] = Header(None),
    tags: Optional[str] = Header(None),
    source: Optional[str] = Header(None),
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
    except Exception as exc:
        logging.exception(f'Error during reading of image {file.filename}.')
        raise Exception(f'Uploaded file {file.filename} is no image file.') from exc

    try:
        app: 'DetectorNode' = request.app
        detections = await app.get_detections(raw_image=np_image,
                                              camera_id=camera_id or mac or None,
                                              tags=tags.split(',') if tags else [],
                                              source=source,
                                              autoupload=autoupload)
    except Exception as exc:
        logging.exception(f'Error during detection of image {file.filename}.')
        raise Exception(f'Error during detection of image {file.filename}.') from exc
    return JSONResponse(detections)
