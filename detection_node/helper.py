from glob import glob
from detection import Detection
from typing import List, Any
import json
import os
from datetime import datetime
import cv2
from fastapi.encoders import jsonable_encoder


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


def save_detections_and_image(dir: str, detections: List[Detection], image: Any, filename: str, tags: List[str]) -> None:
    os.makedirs(dir, exist_ok=True)
    image_path = f'{dir}/{filename}'
    file_name_without_type = filename.rsplit('.', 1)[0]
    json_path = f'{dir}/{file_name_without_type}.json'
    _write_json(json_path, detections, tags)
    cv2.imwrite(image_path, image)


def _write_json(json_path: str, detections: List[Detection], tags: List[str]) -> None:
    date = datetime.utcnow().isoformat(sep='_', timespec='milliseconds')
    with open(json_path, 'w') as f:
        json.dump({'box_detections': jsonable_encoder(detections),
                   'tags': tags,
                   'date': date}, f)
