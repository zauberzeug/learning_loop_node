
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from ..enums import TrainerState
from ..helpers.misc import create_image_folder, create_training_folder

# pylint: disable=no-name-in-module
from .general import Category, Context

KWONLY_SLOTS = {'kw_only': True, 'slots': True} if sys.version_info >= (3, 10) else {}


@dataclass(**KWONLY_SLOTS)
class PretrainedModel():
    name: str
    label: str
    description: str


@dataclass(**KWONLY_SLOTS)
class TrainingStatus():
    id: str  # NOTE this must not be changed, but tests wont detect a change -> update tests!
    name: str

    state: Optional[str]
    uptime: Optional[float]
    errors: Optional[Dict[str, Any]]
    progress: Optional[float]

    pretrained_models: List[PretrainedModel] = field(default_factory=list)
    architecture: Optional[str] = None
    context: Optional[Context] = None

    def short_str(self) -> str:
        prgr = f'{self.progress * 100:.0f}%' if self.progress else ''
        cntxt = f'{self.context.organization}/{self.context.project}' if self.context else ''
        arch = f'.{self.architecture} - ' if self.architecture else ''
        return f'[{str(self.state).rsplit(".", maxsplit=1)[-1]} {prgr}. {self.name}({self.id}). {cntxt}{arch}]'


@dataclass(**KWONLY_SLOTS)
class Training():
    id: str
    context: Context

    project_folder: str  # f'{GLOBALS.data_folder}/{context.organization}/{context.project}'
    images_folder: str  # f'{project_folder}/images'
    training_folder: str  # f'{project_folder}/trainings/{trainings_id}'

    categories: List[Category]
    hyperparameters: Dict[str, Any]

    training_number: int
    training_state: str
    model_variant: str  # from `provided_pretrained_models->name`

    start_time: float = field(default_factory=time.time)

    base_model_uuid: Optional[str] = None  # model uuid to continue training (is loaded from loop)

    # NOTE: these are set later after the model has been uploaded
    image_data: Optional[List[Dict]] = None
    skipped_image_count: Optional[int] = None
    model_uuid_for_detecting: Optional[str] = None  # Model uuid to load from the loop after training and upload

    @property
    def training_folder_path(self) -> Path:
        return Path(self.training_folder)

    @classmethod
    def generate_training(cls, project_folder: str, context: Context, data: Dict[str, Any]) -> 'Training':
        if 'hyperparameters' not in data or not isinstance(data['hyperparameters'], dict):
            raise ValueError('hyperparameters missing or not a dict')
        if 'categories' not in data or not isinstance(data['categories'], list):
            raise ValueError('categories missing or not a list')
        if 'training_number' not in data or not isinstance(data['training_number'], int):
            raise ValueError('training_number missing or not an int')
        if 'model_variant' not in data or not isinstance(data['model_variant'], str):
            raise ValueError('model_variant missing or not a str')

        training_uuid = str(uuid4())

        return Training(
            id=training_uuid,
            context=context,
            project_folder=project_folder,
            images_folder=create_image_folder(project_folder),
            training_folder=create_training_folder(project_folder, training_uuid),
            categories=Category.from_list(data['categories']),
            hyperparameters=data['hyperparameters'],
            training_number=data['training_number'],
            base_model_uuid=data.get('base_model_uuid', None),
            model_variant=data['model_variant'],
            training_state=TrainerState.Initialized.value
        )

    def image_ids(self) -> List[str]:
        assert self.image_data is not None, 'Image data not set'
        return [image['id'] for image in self.image_data]

    def train_image_count(self) -> int:
        assert self.image_data is not None, 'Image data not set'
        return len([image for image in self.image_data if image['set'] == 'train'])

    def test_image_count(self) -> int:
        assert self.image_data is not None, 'Image data not set'
        return len([image for image in self.image_data if image['set'] == 'test'])


@dataclass(**KWONLY_SLOTS)
class TrainingOut():
    trainer_id: str
    trainer_name: Optional[str] = None
    confusion_matrix: Optional[Dict] = None  # This is actually just class-wise metrics
    train_image_count: Optional[int] = None
    test_image_count: Optional[int] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    best_epoch: Optional[int] = None


@dataclass(**KWONLY_SLOTS)
class TrainingStateData():
    confusion_matrix: Dict = field(default_factory=dict)
    meta_information: Dict = field(default_factory=dict)
    epoch: Optional[int] = None


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
