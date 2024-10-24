import logging
from typing import TYPE_CHECKING, Optional

import numpy as np
from fastapi import APIRouter, File, Header, Request, UploadFile
from fastapi.responses import JSONResponse

from ...data_classes.detections import Detections

if TYPE_CHECKING:
    from ..detector_node import DetectorNode

router = APIRouter()


@router.post("/detect", response_model=Detections)
async def http_detect(
    request: Request,
    file: UploadFile = File(..., description='The image file to run detection on'),
    camera_id: Optional[str] = Header(None, description='The camera id (used by learning loop)'),
    mac: Optional[str] = Header(None, description='The camera mac address (used by learning loop)'),
    tags: Optional[str] = Header(None, description='Tags to add to the image (used by learning loop)'),
    source: Optional[str] = Header(None, description='The source of the image (used by learning loop)'),
    autoupload: Optional[str] = Header(None, description='Mode to decide whether to upload the image to the learning loop',
                                       examples=['filtered', 'all', 'disabled']),
):
    """
    Single image example:

        curl --request POST -F 'file=@test.jpg' localhost:8004/detect -H 'autoupload: all' -H 'camera-id: front_cam' -H 'source: test' -H 'tags: test,test2'

    Multiple images example:

        for i in `seq 1 10`; do time curl --request POST -F 'file=@test.jpg' localhost:8004/detect; done

    """
    try:
        np_image = np.fromfile(file.file, np.uint8)
    except Exception as exc:
        logging.exception('Error during reading of image %s.', file.filename)
        raise Exception(f'Uploaded file {file.filename} is no image file.') from exc

    try:
        app: 'DetectorNode' = request.app
        detections = await app.get_detections(raw_image=np_image,
                                              camera_id=camera_id or mac or None,
                                              tags=tags.split(',') if tags else [],
                                              source=source,
                                              autoupload=autoupload)
    except Exception as exc:
        logging.exception('Error during detection of image %s.', file.filename)
        raise Exception(f'Error during detection of image {file.filename}.') from exc
    return detections
