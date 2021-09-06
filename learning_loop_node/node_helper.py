from http import HTTPStatus
from .context import Context
from .task_logger import create_task
import aiohttp
from typing import List, Tuple
import shutil
from io import BytesIO
import zipfile
import os
from glob import glob
from icecream import ic
import aiofiles
import asyncio
import time
from .loop import loop
import logging
from fastapi.encoders import jsonable_encoder


async def download_images_data(organization: str, project: str, image_ids: List[str], chunk_size: int = 100) -> List[dict]:
    images_data = []
    starttime = time.time()
    for i in range(0, len(image_ids), chunk_size):
        chunk_ids = image_ids[i:i+chunk_size]
        async with loop.get(f'api/{organization}/projects/{project}/images?ids={",".join(chunk_ids)}') as response:
            if response.status != 200:
                logging.error(f'Error during downloading list of images. Statuscode is {response.status}')
                continue
            images_data += (await response.json())['images']
            logging.info(f'[+] Downloaded image data: {len(images_data)} / {len(image_ids)}')
            total_time = round(time.time() - starttime, 1)
            if(images_data):
                per100 = total_time / len(images_data) * 100
                logging.debug(f'[+] Performance: {total_time} sec total. Per 100 : {per100:.1f} sec')
            else:
                logging.debug(f'[+] Performance: {total_time} sec total.')
    return images_data


def create_resource_paths(organization_name: str, project_name: str, image_ids: List[str]) -> Tuple[List[str], List[str]]:
    if not image_ids:
        return [], []
    url_ids = [(f'api/{organization_name}/projects/{project_name}/images/{id}/main', id)
               for id in image_ids]
    urls, ids = list(map(list, zip(*url_ids)))

    return urls, ids


async def download_images(paths: List[str], image_ids: List[str], image_folder: str, chunk_size: int = 10) -> None:
    starttime = time.time()
    for i in range(0, len(image_ids), chunk_size):
        chunk_paths = paths[i:i+chunk_size]
        chunk_ids = image_ids[i:i+chunk_size]
        tasks = []
        for j in range(len(chunk_paths)):
            tasks.append(create_task(download_one_image(chunk_paths[j], chunk_ids[j], image_folder)))
        await asyncio.gather(*tasks)
        total_time = round(time.time() - starttime, 1)
        logging.info(f'Downnloaed {i + len(tasks)} image files.')
        per100 = total_time / (i + len(tasks)) * 100
        logging.debug(f'[+] Performance (image files): {total_time} sec total. Per 100 : {per100:.1f}')


async def download_one_image(path: str, image_id: str, image_folder: str):
    async with loop.get(path) as response:
        if response.status != HTTPStatus.OK:
            content = await response.read()
            logging.error(f'bad status code {response.status} for {path}: {content}')
            return
        async with aiofiles.open(f'{image_folder}/{image_id}.jpg', 'wb') as out_file:
            await out_file.write(await response.read())


async def download_model(target_folder: str, context: Context, model_id: str, format: str) -> List[str]:
    # download model
    path = f'api/{context.organization}/projects/{context.project}/models/{model_id}/{format}/file'
    async with loop.get(path) as response:
        if response.status != 200:
            content = await response.read()
            raise Exception(f'could not download model from {loop.base_url}/{path}: {content}')
        try:
            provided_filename = response.headers.get("Content-Disposition").split("filename=")[1].strip('"')
            content = await response.read()
        except:
            logging.exception(await response.read())
            raise

    # unzip and place downloaded model
    tmp_path = f'/tmp/{os.path.splitext(provided_filename)[0]}'
    shutil.rmtree(tmp_path, ignore_errors=True)
    with zipfile.ZipFile(BytesIO(content), 'r') as zip:
        zip.extractall(tmp_path)

    created_files = []
    files = glob(f'{tmp_path}/**/*', recursive=True)
    for file in files:
        logging.debug(f'moving model file {os.path.basename(file)} to training folder.')
        new_file = shutil.move(file, target_folder)
        created_files.append(new_file)
    return created_files


async def upload_model(context: Context, files: List[str], model_id: str, format: str) -> None:
    data = aiohttp.FormData()

    for file_name in files:
        data.add_field('files',  open(file_name, 'rb'))
    async with loop.put(f'api/{context.organization}/projects/{context.project}/models/{model_id}/{format}/file', data=data) as response:
        if response.status != 200:
            msg = f'---- could not save model with id {model_id}. Details: {response.content}'
            raise Exception(msg)
        else:
            logging.info(f'---- uploaded model with id {model_id}')
