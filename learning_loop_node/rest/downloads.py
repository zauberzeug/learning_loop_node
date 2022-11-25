from http import HTTPStatus
from typing import List
import shutil
from io import BytesIO
import zipfile
import os
from glob import glob
import aiofiles
import asyncio
import time
from learning_loop_node.loop import loop
from learning_loop_node.context import Context
from learning_loop_node.task_logger import create_task
import logging
from icecream import ic
from tqdm.asyncio import tqdm

check_jpeg = shutil.which('jpeginfo') is not None


def jepeg_check_info():
    if check_jpeg:
        logging.info('Detected command line tool "jpeginfo". Images will be checked for validity')
    else:
        logging.error('Missing command line tool "jpeginfo". We can not check for validity of images.')


async def download_images_data(organization: str, project: str, image_ids: List[str], chunk_size: int = 100) -> List[dict]:
    logging.info('fetching annotations and other image data')
    jepeg_check_info()
    images_data = []
    starttime = time.time()
    for i in tqdm(range(0, len(image_ids), chunk_size), position=0, leave=True):
        chunk_ids = image_ids[i:i+chunk_size]
        async with loop.get(f'api/{organization}/projects/{project}/images?ids={",".join(chunk_ids)}') as response:
            if response.status != 200:
                logging.error(
                    f'Error during downloading list of images. Statuscode is {response.status}')
                continue
            images_data += (await response.json())['images']
            total_time = round(time.time() - starttime, 1)
            if (images_data):
                per100 = total_time / len(images_data) * 100
                logging.debug(f'[+] Performance: {total_time} sec total. Per 100 : {per100:.1f} sec')
            else:
                logging.debug(f'[+] Performance: {total_time} sec total.')
    return images_data


async def download_images(paths: List[str], image_ids: List[str], image_folder: str, chunk_size: int = 10) -> None:
    if len(image_ids) == 0:
        logging.debug('got empty list. No images were downloaded')
        return
    logging.info('fetching image files')
    starttime = time.time()
    os.makedirs(image_folder, exist_ok=True)
    for i in tqdm(range(0, len(image_ids), chunk_size), position=0, leave=True):
        chunk_paths = paths[i:i+chunk_size]
        chunk_ids = image_ids[i:i+chunk_size]
        tasks = []
        for j in range(len(chunk_paths)):
            tasks.append(create_task(download_one_image(chunk_paths[j], chunk_ids[j], image_folder)))
        await asyncio.gather(*tasks)
        total_time = round(time.time() - starttime, 1)
        per100 = total_time / (i + len(tasks)) * 100
        logging.debug(f'[+] Performance (image files): {total_time} sec total. Per 100 : {per100:.1f}')


async def download_one_image(path: str, image_id: str, image_folder: str):
    async with loop.get(path) as response:
        if response.status != HTTPStatus.OK:
            content = await response.read()
            logging.error(
                f'bad status code {response.status} for {path}: {content}')
            return
        filename = f'{image_folder}/{image_id}.jpg'
        async with aiofiles.open(filename, 'wb') as f:
            await f.write(await response.read())
        if not await is_valid_image(filename):
            os.remove(filename)


async def is_valid_image(file):
    if not os.path.isfile(file):
        return False
    if not check_jpeg:
        return True

    info = await asyncio.create_subprocess_shell(
        f'jpeginfo -c {file}',
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE)
    out, err = await info.communicate()
    return "[OK]" in out.decode()


class DownloadError(Exception):

    def __init__(self, cause: str, *args: object) -> None:
        super().__init__(*args)
        self.cause = cause


async def download_model(target_folder: str, context: Context, model_id: str, format: str) -> List[str]:
    path = f'api/{context.organization}/projects/{context.project}/models/{model_id}/{format}/file'
    async with loop.get(path) as response:
        if response.status != 200:
            content = await response.json()
            logging.error(f'could not download {loop.base_url}/{path}: {response.status}, content: {content}')
            raise DownloadError(content['detail'])
        try:
            provided_filename = response.headers.get(
                "Content-Disposition").split("filename=")[1].strip('"')
            content = await response.read()
        except:
            logging.error(f'Error during downloading model {path}:')
            try:
                logging.exception(await response.json())
            except:
                pass
            raise

    # unzip and place downloaded model
    tmp_path = f'/tmp/{os.path.splitext(provided_filename)[0]}'
    shutil.rmtree(tmp_path, ignore_errors=True)
    with zipfile.ZipFile(BytesIO(content), 'r') as zip:
        zip.extractall(tmp_path)

    created_files = []
    files = glob(f'{tmp_path}/**/*', recursive=True)
    for file in files:
        new_file = shutil.move(file, target_folder)
        logging.info(f'moved model file {os.path.basename(file)} to {new_file}.')
        created_files.append(new_file)
    return created_files
