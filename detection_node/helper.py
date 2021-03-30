from glob import glob
from detection import Detection
from typing import List, Any
import json
import os
from datetime import datetime
import cv2


def find_weight_file(path: str) -> str:
    return glob(f'{path}/*.weights', recursive=True)[0]


def find_cfg_file(path: str) -> str:
    return glob(f'{path}/*.cfg', recursive=True)[0]


def save_detections_and_image(detections: List[Detection], image: Any, mac: str) -> None:
    os.makedirs('/data', exist_ok=True)
    mac = mac.replace(':', '')
    date = datetime.utcnow().isoformat(sep='_', timespec='milliseconds')
    filepath = f'/data/{mac}_{date}'
    _write_json(filepath, detections)
    cv2.imwrite(f'{filepath}.jpg', image)


def _write_json(filepath: str, detections: List[Detection]) -> None:
    with open(f'{filepath}.json', 'w') as f:
        json.dump({'box_detections': detections}, f)
