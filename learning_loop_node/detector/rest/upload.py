from io import BytesIO
from typing import TYPE_CHECKING, List, Optional

import numpy as np
from fastapi import APIRouter, File, Query, Request, UploadFile
from PIL import Image

from ...data_classes.image_metadata import ImageMetadata, ImagesMetadata

if TYPE_CHECKING:
    from ..detector_node import DetectorNode

router = APIRouter()


@router.post("/upload")
async def upload_image(request: Request,
                       files: List[UploadFile] = File(...),
                       source: Optional[str] = Query(None, description='Source of the image'),
                       creation_date: Optional[str] = Query(None, description='Creation date of the image'),
                       upload_priority: bool = Query(False, description='Upload the image with priority')):
    """
    Upload an image or multiple images to the learning loop.

    The image source and the image creation date are optional query parameters.
    Images are automatically tagged with 'picked_by_system'.

    Example Usage

        curl -X POST -F 'files=@test.jpg' "http://localhost:/upload?source=test&creation_date=2024-01-01T00:00:00&upload_priority=true"
    """
    node: 'DetectorNode' = request.app

    files_bytes = [await file.read() for file in files]
    image_metadatas = []
    images = []
    for file_bytes in files_bytes:
        pil_image = Image.open(BytesIO(file_bytes))
        images.append(np.array(pil_image))
        image_metadatas.append(ImageMetadata(source=source, created=creation_date))

    images_metadata = ImagesMetadata(items=image_metadatas)

    await node.upload_images(images=images,
                             images_metadata=images_metadata,
                             upload_priority=upload_priority)
    return 200, "OK"
