from enum import Enum
from learning_loop_node.trainer.training_data import TrainingData
from pydantic import BaseModel
from typing import Optional
from learning_loop_node.context import Context
import json
from fastapi.encoders import jsonable_encoder


class Training(BaseModel):
    base_model_id: Optional[str]
    id: str
    context: Context

    project_folder: str
    images_folder: str

    training_folder: Optional[str]

    data: Optional[TrainingData]

    training_number: Optional[int]

    training_state: Optional[str]

    model_id_for_detecting: Optional[str]


class TrainingOut(BaseModel):
    confusion_matrix: Optional[dict]
    train_image_count: Optional[int]
    test_image_count: Optional[int]
    trainer_id: Optional[str]
    hyperparameters: Optional[dict]


class State(str, Enum):
    Initialized = 'initialized'
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
