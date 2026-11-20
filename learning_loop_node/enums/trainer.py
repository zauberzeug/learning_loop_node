from enum import Enum


class TrainerState(str, Enum):
    Idle = 'idle'
    Blocked = 'blocked'  # idle, but unable to take a training
    WaitingForSlot = 'waiting_for_slot'
    # a training is in the state of the step it performs next, also while that step waits for a retry
    DataDownloading = 'data_downloading'
    TrainModelDownloading = 'train_model_downloading'
    TrainingRunning = 'running'
    ConfusionMatrixSyncing = 'confusion_matrix_syncing'
    TrainModelUploading = 'train_model_uploading'
    Detecting = 'detecting'
    DetectionUploading = 'detection_uploading'
    ReadyForCleanup = 'ready_for_cleanup'
