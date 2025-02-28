from typing import TYPE_CHECKING, List, Optional

from fastapi import APIRouter, File, Query, Request, UploadFile

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
    raw_files = [await file.read() for file in files]
    node: DetectorNode = request.app
    await node.upload_images(images=raw_files, source=source, creation_date=creation_date, upload_priority=upload_priority)
    return 200, "OK"
