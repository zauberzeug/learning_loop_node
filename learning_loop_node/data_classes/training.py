
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

# pylint: disable=no-name-in-module
from .general import Category, Context

KWONLY_SLOTS = {'kw_only': True, 'slots': True} if sys.version_info >= (3, 10) else {}


@dataclass(**KWONLY_SLOTS)
class Hyperparameter():
    resolution: int
    flip_rl: bool
    flip_ud: bool

    @staticmethod
    def from_data(data: Dict):
        return Hyperparameter(
            resolution=data['resolution'],
            flip_rl=data.get('flip_rl', False),
            flip_ud=data.get('flip_ud', False)
        )


@dataclass(**KWONLY_SLOTS)
class TrainingData():
    image_data: List[Dict] = field(default_factory=list)
    skipped_image_count: Optional[int] = 0
    categories: List[Category] = field(default_factory=list)
    hyperparameter: Optional[Hyperparameter] = None

    def image_ids(self):
        return [image['id'] for image in self.image_data]

    def train_image_count(self):
        return len([image for image in self.image_data if image['set'] == 'train'])

    def test_image_count(self):
        return len([image for image in self.image_data if image['set'] == 'test'])


@dataclass(**KWONLY_SLOTS)
class PretrainedModel():
    name: str
    label: str
    description: str


class TrainerState(str, Enum):
    Idle = 'idle'
    Initialized = 'initialized'
    Preparing = 'preparing'
    DataDownloading = 'data_downloading'
    DataDownloaded = 'data_downloaded'
    TrainModelDownloading = 'train_model_downloading'
    TrainModelDownloaded = 'train_model_downloaded'
    TrainingRunning = 'running'
    TrainingFinished = 'training_finished'
    ConfusionMatrixSyncing = 'confusion_matrix_syncing'
    ConfusionMatrixSynced = 'confusion_matrix_synced'
    TrainModelUploading = 'train_model_uploading'
    TrainModelUploaded = 'train_model_uploaded'
    Detecting = 'detecting'
    Detected = 'detected'
    DetectionUploading = 'detection_uploading'
    ReadyForCleanup = 'ready_for_cleanup'


@dataclass(**KWONLY_SLOTS)
class TrainingStatus():
    id: str  # NOTE this must not be changed, but tests wont detect a change -> update tests!
    name: str
    state: Optional[str]
    errors: Optional[Dict]
    uptime: Optional[float]
    progress: Optional[float]

    train_image_count: Optional[int] = None
    test_image_count: Optional[int] = None
    skipped_image_count: Optional[int] = None
    pretrained_models: List[PretrainedModel] = field(default_factory=list)
    hyperparameters: Optional[Dict] = None
    architecture: Optional[str] = None
    context: Optional[Context] = None

    def short_str(self) -> str:
        prgr = f'{self.progress * 100:.0f}%' if self.progress else ''
        trtesk = f'{self.train_image_count}/{self.test_image_count}/{self.skipped_image_count}' if self.train_image_count else 'n.a.'
        cntxt = f'{self.context.organization}/{self.context.project}' if self.context else ''
        hyps = f'({self.hyperparameters})' if self.hyperparameters else ''
        arch = f'.{self.architecture} - ' if self.architecture else ''
        return f'[{str(self.state).rsplit(".", maxsplit=1)[-1]} {prgr}. {self.name}({self.id}). Tr/Ts/Tsk: {trtesk} {cntxt}{arch}{hyps}]'


@dataclass(**KWONLY_SLOTS)
class Training():
    id: str
    context: Context

    project_folder: str  # f'{GLOBALS.data_folder}/{context.organization}/{context.project}'
    images_folder: str  # f'{project_folder}/images'
    training_folder: str  # f'{project_folder}/trainings/{trainings_id}'
    start_time: float = field(default_factory=time.time)

    # model uuid to download (to continue training) | is not a uuid when training from scratch (blank or pt-name from provided_pretrained_models->name)
    base_model_uuid_or_name: Optional[str] = None

    data: Optional[TrainingData] = None
    training_number: Optional[int] = None
    training_state: Optional[str] = None
    model_uuid_for_detecting: Optional[str] = None
    hyperparameters: Optional[Dict] = None

    @property
    def training_folder_path(self) -> Path:
        return Path(self.training_folder)

    def set_values_from_data(self, data: Dict) -> None:
        self.data = TrainingData(categories=Category.from_list(data['categories']))
        self.data.hyperparameter = Hyperparameter.from_data(data=data)
        self.training_number = data['training_number']
        self.base_model_uuid_or_name = data['id']
        self.training_state = TrainerState.Initialized


@dataclass(**KWONLY_SLOTS)
class TrainingOut():
    confusion_matrix: Optional[Dict] = None  # This is actually just class-wise metrics
    train_image_count: Optional[int] = None
    test_image_count: Optional[int] = None
    trainer_id: Optional[str] = None
    hyperparameters: Optional[Dict] = None


@dataclass(**KWONLY_SLOTS)
class TrainingStateData():
    confusion_matrix: Dict = field(default_factory=dict)
    meta_information: Dict = field(default_factory=dict)


@dataclass(**KWONLY_SLOTS)
class Model():
    uuid: str
    confusion_matrix: Optional[Dict] = None
    parent_id: Optional[str] = None
    train_image_count: Optional[int] = None
    test_image_count: Optional[int] = None
    trainer_id: Optional[str] = None
    hyperparameters: Optional[str] = None


class Errors():
    def __init__(self) -> None:
        self._errors: Dict[str, str] = {}

    def set(self, key: str, value: str):
        self._errors[key] = value

    @property
    def errors(self) -> Dict:
        return self._errors

    def reset(self, key: str) -> None:
        try:
            del self._errors[key]
        except AttributeError:
            pass
        except KeyError:
            pass

    def reset_all(self) -> None:
        self._errors = {}

    def has_error_for(self, key: str) -> bool:
        return key in self._errors

    def has_error(self) -> bool:
        return not self._errors


class TrainingError(Exception):
    def __init__(self, cause: str, *args: object) -> None:
        super().__init__(*args)
        self.cause = cause

    def __str__(self) -> str:
        return f'TrainingError: {self.cause}'
