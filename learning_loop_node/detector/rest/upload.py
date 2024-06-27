from typing import TYPE_CHECKING, List

from fastapi import APIRouter, File, Request, UploadFile

if TYPE_CHECKING:
    from ..detector_node import DetectorNode

router = APIRouter()


@router.post("/upload")
async def upload_image(request: Request, files: List[UploadFile] = File(...)):
    """
    Example Usage

        curl -X POST -F 'files=@test.jpg' "http://localhost:/upload"
    """
    raw_files = [await file.read() for file in files]
    node: DetectorNode = request.app
    await node.upload_images(raw_files)
    return 200, "OK"
