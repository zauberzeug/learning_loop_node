from glob import glob

from fastapi.datastructures import UploadFile
from detection import Detection
from typing import List, Any
import json
import os
from datetime import datetime
import cv2
from fastapi.encoders import jsonable_encoder
import aiofiles
from filelock import FileLock
from icecream import ic
import os


def find_weight_file(path: str) -> str:
    return glob(f'{path}/*.weights', recursive=True)[0]


def find_cfg_file(path: str) -> str:
    return glob(f'{path}/*.cfg', recursive=True)[0]


def extract_macs_and_filenames(file_paths: List[str]) -> List[str]:
    mac_dict = {}
    files = [path.split('/')[-1] for path in file_paths]
    for file in files:
        key = file.split('_')[0]
        value = file.rsplit('.', 1)[0]
        mac_dict.setdefault(key, [])
        if not any(item == value for item in mac_dict.get(key)):
            mac_dict[key].append(value)
    return mac_dict


async def save_detections_and_image(dir: str, detections: List[Detection], image_data: UploadFile, file_name: str, tags: List[str]) -> None:
    os.makedirs(dir, exist_ok=True)
    file_name_without_type = file_name.rsplit('.', 1)[0]
    json_file_name = f'{file_name_without_type}.json'
    await _write_json(json_file_name, detections, tags)
    await write_file(image_data, file_name)


async def _write_json(json_file_name: str, detections: List[Detection], tags: List[str]) -> None:
    date = datetime.utcnow().isoformat(sep='_', timespec='milliseconds')
    json_data = json.dumps({'box_detections': jsonable_encoder(detections),
                           'tags': tags,
                            'date': date})
    with FileLock(lock_file=f'/data/{json_file_name}'):
        async with aiofiles.open(f'/data/{json_file_name}', 'w') as out_file:
            await out_file.write(json_data)


async def write_file(file_data: Any, file_name: str):
    # Be sure to start from beginning.
    await file_data.seek(0)
    with FileLock(lock_file=f'/data/{file_name}.lock'):
        async with aiofiles.open(f'/data/{file_name}', 'wb') as out_file:
            while True:
                content = await file_data.read(1024)  # async read chunk
                if not content:
                    break
                await out_file.write(content)  # async write chunk

    os.remove(f'/data/{file_name}.lock')


def get_data_files():
    return glob('/data/*[!.lock]', recursive=True)
