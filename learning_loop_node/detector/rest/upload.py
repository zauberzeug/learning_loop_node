from fastapi import APIRouter, Request, File, UploadFile
from typing import List


router = APIRouter()


@router.post("/upload")
async def upload_image(request: Request, files: List[UploadFile] = File(...)):
    """
    Example Usage

        curl -X POST -F 'files=@test.jpg' "http://localhost:/upload"
    """
    raw_files = [await file.read() for file in files]
    await request.app.upload_images(raw_files)
    return 200, "OK"
