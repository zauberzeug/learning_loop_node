
import json
import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import List

from dacite import from_dict
from fastapi.encoders import jsonable_encoder

from ..data_classes import Context, Detections, Training
from ..globals import GLOBALS
from ..loop_communication import LoopCommunicator


class EnvironmentVars:
    def __init__(self) -> None:
        self.restart_after_training = os.environ.get(
            'RESTART_AFTER_TRAINING', 'FALSE').lower() in ['true', '1']
        self.keep_old_trainings = os.environ.get(
            'KEEP_OLD_TRAININGS', 'FALSE').lower() in ['true', '1']
        self.inference_batch_size = int(
            os.environ.get('INFERENCE_BATCH_SIZE', '10'))


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

    # @staticmethod
    # def create_mocked_training_io() -> 'ActiveTrainingIO':
    #     training_folder = ''
    #     return ActiveTrainingIO(training_folder)

    def __init__(self, training_folder: str, loop_communicator: LoopCommunicator, context: Context) -> None:
        self.training_folder = training_folder
        self.loop_communicator = loop_communicator
        self.context = context

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

    def _get_detection_file_names(self) -> List[Path]:
        files = [f for f in Path(self.training_folder).iterdir()
                 if f.is_file() and f.name.startswith('detections_')]
        if not files:
            return []
        return files

    def get_number_of_detection_files(self) -> int:
        return len(self._get_detection_file_names())

    # TODO: saving and uploading multiple files is not tested!
    def save_detections(self, detections: List[Detections], index: int = 0) -> None:
        with open(self.det_path.format(index), 'w') as f:
            json.dump(jsonable_encoder([asdict(d) for d in detections]), f)

    def load_detections(self, index: int = 0) -> List[Detections]:
        with open(self.det_path.format(index), 'r') as f:
            dict_list = json.load(f)
            return [from_dict(data_class=Detections, data=d) for d in dict_list]

    def delete_detections(self) -> None:
        for file in self._get_detection_file_names():
            os.remove(Path(self.training_folder) / file)

    def detections_exist(self) -> bool:
        return bool(self._get_detection_file_names())

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

    async def upload_detetions(self):
        num_files = self.get_number_of_detection_files()
        print(f'num_files: {num_files}', flush=True)
        if not num_files:
            logging.error('no detection files found')
            return
        current_json_file_index = self.load_detections_upload_file_index()
        for i in range(current_json_file_index, num_files):
            detections = self.load_detections(i)
            logging.info(f'uploading detections in file {i}/{num_files}')
            await self._upload_detections_batched(self.context, detections)
            self.save_detections_upload_file_index(i+1)

    async def _upload_detections_batched(self, context: Context, detections: List[Detections]):
        batch_size = 100
        skip_detections = self.load_detection_upload_progress()
        for i in range(skip_detections, len(detections), batch_size):
            up_progress = i + batch_size if i + batch_size < len(detections) else 0
            batch_detections = detections[i:i + batch_size]
            await self._upload_detections_and_save_progress(context, batch_detections, up_progress)
            skip_detections = up_progress

        logging.info('uploaded %d detections', len(detections))

    async def _upload_detections_and_save_progress(self, context: Context, batch_detections: List[Detections], up_progress: int):
        if len(batch_detections) == 0:
            print('skipping empty batch', flush=True)
            return
        detections_json = [jsonable_encoder(asdict(detections)) for detections in batch_detections]
        print(f'uploading {len(detections_json)} detections', flush=True)
        response = await self.loop_communicator.post(
            f'/{context.organization}/projects/{context.project}/detections', json=detections_json)
        if response.status_code != 200:
            msg = f'could not upload detections. {str(response)}'
            logging.error(msg)
            raise Exception(msg)

        logging.info('successfully uploaded detections')
        self.save_detection_upload_progress(up_progress)
