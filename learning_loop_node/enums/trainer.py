
from enum import Enum


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
