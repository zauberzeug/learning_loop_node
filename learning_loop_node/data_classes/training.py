
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Union

# pylint: disable=no-name-in-module
from pydantic import BaseModel

from learning_loop_node.data_classes import Category, Context


@dataclass
class Hyperparameter():
    resolution: int
    flip_rl: bool
    flip_ud: bool

    @staticmethod
    def from_dict(value: dict) -> 'Hyperparameter':
        return Hyperparameter(
            resolution=value['resolution'],
            flip_rl=value['flip_rl'],
            flip_ud=value['flip_ud']
        )


class TrainingData(BaseModel):
    image_data: List[Dict] = []
    skipped_image_count: Optional[int] = 0
    categories: List[Category] = []
    hyperparameter: Optional[Hyperparameter] = None

    def image_ids(self):
        return [image['id'] for image in self.image_data]

    def train_image_count(self):
        return len([image for image in self.image_data if image['set'] == 'train'])

    def test_image_count(self):
        return len([image for image in self.image_data if image['set'] == 'test'])


@dataclass
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


@dataclass
class TrainingStatus():
    uuid: str
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


class Training(BaseModel):
    base_model_id: Optional[str] = None
    uuid: str
    context: Context

    project_folder: str
    images_folder: str
    training_folder: str

    data: Optional[TrainingData] = None
    training_number: Optional[int] = None
    training_state: Optional[TrainingState] = None
    model_id_for_detecting: Optional[str] = None
    hyperparameters: Optional[Dict] = None


class TrainingOut(BaseModel):
    confusion_matrix: Optional[Dict] = None
    train_image_count: Optional[int] = None
    test_image_count: Optional[int] = None
    trainer_id: Optional[str] = None
    hyperparameters: Optional[Dict] = None


class BasicModel(BaseModel):
    confusion_matrix: Optional[Dict] = None
    meta_information: Optional[Dict] = None


class Model(BaseModel):
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
