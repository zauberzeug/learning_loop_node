class CriticalError(Exception):
    """
    CriticalError is raised when the training cannot be continued.
    In this case the trainer jumps to the TrainerState.ReadyForCleanup and tries to upload the latest model.
    """


class ModelUploadFailedError(Exception):
    """Raised when the loop could not store an uploaded model, e.g. because it answered with a server error.
    Unlike a CriticalError it keeps the training, so the upload is tried again later.
    """


class TrainingRefusedError(Exception):
    """Raised when a trainer cannot take on a training it was asked to begin; nothing was set up."""


class HyperparameterMismatchError(Exception):
    """Raised when a training's hyperparameters do not fit the trainer's declaration; nothing was set up.

    Unlike a refusal, the training would fail on any trainer of the same declaration.
    """


class TrainingSlotTakenError(TrainingRefusedError):
    """Raised when a sibling node holds the training slot."""

    def __init__(self, holder: str) -> None:
        super().__init__(f'the training slot is held by {holder}')
        self.holder = holder


class InsufficientMemoryError(RuntimeError):
    """Raised when not even the smallest unit of work fits in memory."""


class NodeNeedsRestartError(Exception):
    """
    NodeNeedsRestartError is raised when the node needs to be restarted.
    This is e.g. the case when the GPU is not available anymore.
    """
