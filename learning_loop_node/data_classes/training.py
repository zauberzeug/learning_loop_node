
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Union

# pylint: disable=no-name-in-module
from .general import Category, Context

KWONLY_SLOTS = {'kw_only': True, 'slots': True} if sys.version_info >= (3, 10) else {}


@dataclass(**KWONLY_SLOTS)
class Hyperparameter():
    resolution: int
    flip_rl: bool
    flip_ud: bool


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


class TrainingState(str, Enum):
    Initialized = 'initialized'
    Preparing = 'preparing'
    DataDownloading = 'data_downloading'
    DataDownloaded = 'data_downloaded'
    TrainModelDownloading = 'train_model_downloading'
    TrainModelDownloaded = 'train_model_downloaded'
    TrainingRunning = 'training_running'
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
    id: str  # TODO this must not be changed, but tests wont detect it -> update tests!
    name: str
    state: Union[Optional[TrainingState], str]
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

    def short_str(self):
        prgr = f'{self.progress * 100:.0f}%' if self.progress else ''
        trtesk = f'{self.train_image_count}/{self.test_image_count}/{self.skipped_image_count}' if self.train_image_count else 'n.a.'
        cntxt = f'{self.context.organization}/{self.context.project}' if self.context else ''
        hyps = f'({self.hyperparameters})' if self.hyperparameters else ''
        arch = f'.{self.architecture} - ' if self.architecture else ''
        return f'[{str(self.state)} {prgr}. {self.name}({self.id}). Tr/Ts/Tsk: {trtesk} {cntxt}{arch}{hyps}]'


@dataclass(**KWONLY_SLOTS)
class Training():
    id: str
    context: Context

    project_folder: str
    images_folder: str
    training_folder: str

    base_model_id: Optional[str] = None
    data: Optional[TrainingData] = None
    training_number: Optional[int] = None
    training_state: Optional[Union[TrainingState, str]] = None
    model_id_for_detecting: Optional[str] = None
    hyperparameters: Optional[Dict] = None


@dataclass(**KWONLY_SLOTS)
class TrainingOut():
    confusion_matrix: Optional[Dict] = None
    train_image_count: Optional[int] = None
    test_image_count: Optional[int] = None
    trainer_id: Optional[str] = None
    hyperparameters: Optional[Dict] = None


@dataclass(**KWONLY_SLOTS)
class BasicModel():
    confusion_matrix: Optional[Dict] = None
    meta_information: Optional[Dict] = None


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
    def __init__(self):
        self._errors: Dict = {}

    def set(self, key: str, value: str):
        self._errors[key] = value

    @property
    def errors(self) -> Dict:
        return self._errors

    def reset(self, key: str):
        try:
            del self._errors[key]
        except AttributeError:
            pass
        except KeyError:
            pass

    def reset_all(self):
        self._errors = {}

    def has_error_for(self, key: str) -> bool:
        return key in self._errors

    def has_error(self) -> bool:
        return not self._errors


class TrainingError(Exception):
    def __init__(self, cause: str, *args: object) -> None:
        super().__init__(*args)
        self.cause = cause
