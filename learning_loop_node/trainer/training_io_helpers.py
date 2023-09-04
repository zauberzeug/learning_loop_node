
import json
import os
from pathlib import Path
from typing import List

from fastapi.encoders import jsonable_encoder

from learning_loop_node.data_classes import Training
from learning_loop_node.globals import GLOBALS


class LastTrainingIO:

    @staticmethod
    def create_mocked_last_training_io() -> 'LastTrainingIO':
        node_uuid = '00000000-0000-0000-0000-000000000000'
        return LastTrainingIO(node_uuid)

    def __init__(self, node_uuid: str) -> None:
        self.node_uuid = node_uuid

    def save(self, training: Training) -> None:
        with open(f'{GLOBALS.data_folder}/last_training__{self.node_uuid}.json', 'w') as f:
            json.dump(jsonable_encoder(training), f)

    def load(self) -> Training:
        with open(f'{GLOBALS.data_folder}/last_training__{self.node_uuid}.json', 'r') as f:
            return Training(**json.load(f))

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
        self.det_path = f'{training_folder}/detections_{0}.json'
        self.dufi_path = f'{training_folder}/detection_uploading_json_index.txt'
        self.dup_path = f'{training_folder}/detection_uploading_progress.txt'

    # model upload progress

    def mup_save(self, formats: List[str]) -> None:
        with open(self.mup_path, 'w') as f:
            f.write(','.join(formats))

    def mup_load(self) -> List[str]:
        if not os.path.exists(self.mup_path):
            return []
        with open(self.mup_path, 'r') as f:
            return f.read().split(',')

    def mup_delete(self) -> None:
        if os.path.exists(self.mup_path):
            os.remove(self.mup_path)

    # detections

    def det_get_file_names(self) -> List:
        files = [f for f in Path(self.training_folder).iterdir()
                 if f.is_file() and f.name.startswith('detections_')]
        if not files:
            return []
        return files

    def det_save(self, detections: List, index: int = 0) -> None:
        with open(self.det_path.format(index), 'w') as f:
            json.dump(jsonable_encoder(detections), f)

    def det_load(self, index: int = 0) -> List:
        with open(self.det_path.format(index), 'r') as f:
            return json.load(f)

    def det_delete(self) -> None:
        for file in self.det_get_file_names():
            os.remove(Path(self.training_folder) / file)

    def det_exists(self) -> bool:
        return bool(self.det_get_file_names())

    # detections upload file index

    def dufi_save(self, index: int) -> None:
        with open(self.dufi_path, 'w') as f:
            f.write(str(index))

    def dufi_load(self) -> int:
        if not self.dufi_exists():
            return 0
        with open(self.dufi_path, 'r') as f:
            return int(f.read())

    def dufi_delete(self) -> None:
        if self.dufi_exists():
            os.remove(self.dufi_path)

    def dufi_exists(self) -> bool:
        return os.path.exists(self.dufi_path)

    # detections upload progress

    def dup_save(self, count: int) -> None:
        with open(self.dup_path, 'w') as f:
            f.write(str(count))

    def dup_load(self) -> int:
        if not self.dup_exists():
            return 0
        with open(self.dup_path, 'r') as f:
            return int(f.read())

    def dup_delete(self) -> None:
        if self.dup_exists():
            os.remove(self.dup_path)

    def dup_exists(self) -> bool:
        return os.path.exists(self.dup_path)
