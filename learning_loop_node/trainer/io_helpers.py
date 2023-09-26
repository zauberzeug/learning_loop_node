
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import List

from dacite import from_dict
from fastapi.encoders import jsonable_encoder

from ..data_classes import Detections, Training
from ..globals import GLOBALS


class LastTrainingIO:

    def __init__(self, node_uuid: str) -> None:
        self.node_uuid = node_uuid

    def save(self, training: Training) -> None:
        with open(f'{GLOBALS.data_folder}/last_training__{self.node_uuid}.json', 'w') as f:
            json.dump(asdict(training), f)

    def load(self) -> Training:
        with open(f'{GLOBALS.data_folder}/last_training__{self.node_uuid}.json', 'r') as f:
            return from_dict(data_class=Training, data=json.load(f))

    def delete(self) -> None:
        if self.exists():
            os.remove(f'{GLOBALS.data_folder}/last_training__{self.node_uuid}.json')

    def exists(self) -> bool:
        return os.path.exists(f'{GLOBALS.data_folder}/last_training__{self.node_uuid}.json')


class ActiveTrainingIO:

    @staticmethod
    def create_mocked_training_io() -> 'ActiveTrainingIO':
        training_folder = ''
        return ActiveTrainingIO(training_folder)

    def __init__(self, training_folder: str):
        self.training_folder = training_folder
        self.mup_path = f'{training_folder}/model_uploading_progress.txt'
        # string with placeholder gor index
        self.det_path = f'{training_folder}' + '/detections_{0}.json'
        self.dufi_path = f'{training_folder}/detection_uploading_json_index.txt'
        self.dup_path = f'{training_folder}/detection_uploading_progress.txt'

    # model upload progress
    # NOTE: progress file is deleted implicitly after training

    def save_model_upload_progress(self, formats: List[str]) -> None:
        with open(self.mup_path, 'w') as f:
            f.write(','.join(formats))

    def load_model_upload_progress(self) -> List[str]:
        if not os.path.exists(self.mup_path):
            return []
        with open(self.mup_path, 'r') as f:
            return f.read().split(',')

    # detections

    def get_detection_file_names(self) -> List[Path]:
        files = [f for f in Path(self.training_folder).iterdir()
                 if f.is_file() and f.name.startswith('detections_')]
        if not files:
            return []
        return files

    # TODO: saving and uploading multiple files is not tested!
    def save_detections(self, detections: List[Detections], index: int = 0) -> None:
        with open(self.det_path.format(index), 'w') as f:
            json.dump(jsonable_encoder([asdict(d) for d in detections]), f)

    def load_detections(self, index: int = 0) -> List[Detections]:
        with open(self.det_path.format(index), 'r') as f:
            dict_list = json.load(f)
            return [from_dict(data_class=Detections, data=d) for d in dict_list]

    def delete_detections(self) -> None:
        for file in self.get_detection_file_names():
            os.remove(Path(self.training_folder) / file)

    def detections_exist(self) -> bool:
        return bool(self.get_detection_file_names())

    # detections upload file index

    def save_detections_upload_file_index(self, index: int) -> None:
        with open(self.dufi_path, 'w') as f:
            f.write(str(index))

    def load_detections_upload_file_index(self) -> int:
        if not self.detections_upload_file_index_exists():
            return 0
        with open(self.dufi_path, 'r') as f:
            return int(f.read())

    def delete_detections_upload_file_index(self) -> None:
        if self.detections_upload_file_index_exists():
            os.remove(self.dufi_path)

    def detections_upload_file_index_exists(self) -> bool:
        return os.path.exists(self.dufi_path)

    # detections upload progress

    def save_detection_upload_progress(self, count: int) -> None:
        with open(self.dup_path, 'w') as f:
            f.write(str(count))

    def load_detection_upload_progress(self) -> int:
        if not self.detection_upload_progress_exist():
            return 0
        with open(self.dup_path, 'r') as f:
            return int(f.read())

    def delete_detection_upload_progress(self) -> None:
        if self.detection_upload_progress_exist():
            os.remove(self.dup_path)

    def detection_upload_progress_exist(self) -> bool:
        return os.path.exists(self.dup_path)
