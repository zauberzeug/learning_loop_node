from typing import List
import requests
import shutil
from io import BytesIO
import zipfile
import os
from glob import glob
from node import Node
from icecream import ic
import asyncio
import aiohttp
import aiofiles


async def download_images_data(node: Node, image_ids: List[str]) -> List[dict]:
    session = requests.Session()
    session.headers = node.headers
    images_data = []
    chunk_size = 10
    for i in range(0, len(image_ids), chunk_size):
        chunk_ids = image_ids[i:i+chunk_size]
        async with aiohttp.ClientSession() as session:
            async with session.get(f'{node.url}/api/zauberzeug/projects/pytest/images?ids={",".join(chunk_ids)}', headers=node.headers) as response:
                assert response.status == 200, f'Error during downloading list of images. Statuscode is {response.status}'
                images_data += (await response.json())['images']
    return images_data


async def download_images(node: Node, images_data: List[dict], image_folder: str) -> None:
    session = requests.Session()
    session.headers = node.headers
    chunk_size = 10
    for i in range(0, len(images_data), chunk_size):
        chunk_data = images_data[i:i+chunk_size]
        for image_information in chunk_data:
            resource = image_information['resource']
            id = image_information['id']
            await _download_resource(node, resource, id, image_folder)


async def _download_resource(node: Node, image_resource: str, image_id: str, image_folder: str) -> None:
    async with aiohttp.ClientSession() as session:
        resource_url = f"{node.url}/api{image_resource}"
        async with session.get(resource_url, headers=node.headers) as response:
            assert response.status == 200
            async with aiofiles.open(f'/{image_folder}/{image_id}.jpg', 'wb') as out_file:
                await out_file.write(await response.read())


def download_model(node: Node, training_folder: str, organization: str, project: str, model_id: str):
    # download model
    download_response = requests.get(
        f'{node.url}/api/{organization}/projects/{project}/models/{model_id}/file', headers=node.headers)
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
