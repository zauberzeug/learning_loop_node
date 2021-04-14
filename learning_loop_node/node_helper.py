from typing import List
import requests
import shutil
from io import BytesIO
import zipfile
import os
from glob import glob
from icecream import ic
import aiohttp
import aiofiles
from icecream import ic


async def download_images_data(base_url: str, headers: dict, image_ids: List[str]) -> List[dict]:
    images_data = []
    chunk_size = 10
    for i in range(0, len(image_ids), chunk_size):
        chunk_ids = image_ids[i:i+chunk_size]
        async with aiohttp.ClientSession() as session:
            async with session.get(f'{base_url}/api/zauberzeug/projects/pytest/images?ids={",".join(chunk_ids)}', headers=headers) as response:
                assert response.status == 200, f'Error during downloading list of images. Statuscode is {response.status}'
                images_data += (await response.json())['images']
                ic(f'[+] Downloaded image data: {images_data} / {len(image_ids)}')
    return images_data


async def download_images(base_url: str, headers: dict, images_data: List[dict], image_folder: str) -> None:
    chunk_size = 10
    for i in range(0, len(images_data), chunk_size):
        chunk_data = images_data[i:i+chunk_size]
        for image_information in chunk_data:
            resource = image_information['resource']
            id = image_information['id']
            await _download_resource(base_url, headers, resource, id, image_folder)
        ic(f'[+] Downloading images: {(i+1)*chunk_size} / {len(images_data)}')


async def _download_resource(base_url: str, headers: dict, image_resource: str, image_id: str, image_folder: str) -> None:
    async with aiohttp.ClientSession() as session:
        resource_url = f"{base_url}/api{image_resource}"
        async with session.get(resource_url, headers=headers) as response:
            assert response.status == 200
            async with aiofiles.open(f'/{image_folder}/{image_id}.jpg', 'wb') as out_file:
                await out_file.write(await response.read())


def download_model(base_url: str, headers: dict, training_folder: str, organization: str, project: str, model_id: str):
    # download model
    download_response = requests.get(
        f'{base_url}/api/{organization}/projects/{project}/models/{model_id}/file', headers=headers)
    assert download_response.status_code == 200,  download_response.status_code
    provided_filename = download_response.headers.get(
        "Content-Disposition").split("filename=")[1].strip('"')

    # unzip and place downloaded model
    target_path = f'/tmp/{os.path.splitext(provided_filename)[0]}'
    shutil.rmtree(target_path, ignore_errors=True)
    filebytes = BytesIO(download_response.content)
    with zipfile.ZipFile(filebytes, 'r') as zip:
        zip.extractall(target_path)

    files = glob(f'{target_path}/**/*', recursive=True)
    for file in files:
        ic(f'moving model file {os.path.basename(file)} to training folder.')
        shutil.move(file, training_folder)
