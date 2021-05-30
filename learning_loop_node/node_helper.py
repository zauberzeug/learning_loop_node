from typing import List, Tuple
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
from .trainer.training_data import TrainingData
import asyncio
import time


async def download_images_data(base_url: str, headers: dict, organization: str, project: str, image_ids: List[str], chunk_size: int = 100) -> List[dict]:
    images_data = []
    starttime = time.time()
    url = f'{base_url}/api/{organization}/projects/{project}/images'
    for i in range(0, len(image_ids), chunk_size):
        chunk_ids = image_ids[i:i+chunk_size]
        async with aiohttp.ClientSession() as session:
            async with session.get(f'{url}?ids={",".join(chunk_ids)}', headers=headers) as response:
                assert response.status == 200, f'Error during downloading list of images. Statuscode is {response.status}'
                images_data += (await response.json())['images']
                ic(f'[+] Downloaded image data: {len(images_data)} / {len(image_ids)}')
                total_time = round(time.time() - starttime, 1)
                if(images_data):
                    ic(f'[+] Performance: {total_time} sec total. Per 100 : {(total_time / len(images_data ) * 100):.1f} sec')
                else:
                    ic(f'[+] Performance: {total_time} sec total.')
    return images_data


def create_resource_urls(base_url: str, organization_name: str, project_name: str, image_ids: List[str]) -> Tuple[List[str], List[str]]:
    if not image_ids:
        return [], []
    url_ids = [(f'{base_url}/api/{organization_name}/projects/{project_name}/images/{id}/main', id)
               for id in image_ids]
    urls, ids = list(map(list, zip(*url_ids)))

    return urls, ids


async def download_images(loop: asyncio.BaseEventLoop, urls: List[str], image_ids: List[str],  headers: dict, image_folder: str, chunk_size: int = 10) -> None:
    starttime = time.time()
    for i in range(0, len(image_ids), chunk_size):
        chunk_urls = urls[i:i+chunk_size]
        chunk_ids = image_ids[i:i+chunk_size]
        tasks = []
        for j in range(len(chunk_urls)):
            tasks.append(loop.create_task(download_one_image(chunk_urls[j], chunk_ids[j], headers, image_folder)))
        await asyncio.gather(*tasks)
        total_time = round(time.time() - starttime, 1)
        ic(f'Downnloaed {i + len(tasks)} image files.')
        ic(f'[+] Performance (image files): {total_time} sec total. Per 100 : {(total_time / (i + len(tasks)) * 100):.1f}')


async def download_one_image(url: str, image_id: str, headers: dict, image_folder: str):
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            assert response.status == 200, f'{response.status} for {url}'
            async with aiofiles.open(f'{image_folder}/{image_id}.jpg', 'wb') as out_file:
                await out_file.write(await response.read())


def download_model(base_url: str, headers: dict, target_folder: str, organization: str, project: str, model_id: str, format: str) -> List[str]:
    # download model
    download_response = requests.get(
        f'{base_url}/api/{organization}/projects/{project}/models/{model_id}/{format}/file', headers=headers)
    assert download_response.status_code == 200,  download_response.status_code
    provided_filename = download_response.headers.get(
        "Content-Disposition").split("filename=")[1].strip('"')

    # unzip and place downloaded model
    tmp_paht = f'/tmp/{os.path.splitext(provided_filename)[0]}'
    shutil.rmtree(tmp_paht, ignore_errors=True)
    filebytes = BytesIO(download_response.content)
    with zipfile.ZipFile(filebytes, 'r') as zip:
        zip.extractall(tmp_paht)

    created_files = []
    files = glob(f'{tmp_paht}/**/*', recursive=True)
    for file in files:
        ic(f'moving model file {os.path.basename(file)} to training folder.')
        new_file = shutil.move(file, target_folder)
        created_files.append(new_file)
    return created_files


async def upload_model(base_url: str, headers: dict, organization: str, project: str, files: List[str], model_id: str, format: str) -> None:
    uri_base = f'{base_url}/api/{organization}/projects/{project}'
    data = aiohttp.FormData()

    for file_name in files:
        data.add_field('files',  open(file_name, 'rb'))

    async with aiohttp.ClientSession() as session:
        async with session.put(f'{uri_base}/models/{model_id}/{format}/file', data=data, headers=headers) as response:
            if response.status != 200:
                msg = f'---- could not save model with id {model_id}'
                raise Exception(msg)
            else:
                ic(f'---- uploaded model with id {model_id}')
