class CriticalError(Exception):
    """
    CriticalError is raised when the training cannot be continued.
    In this case the trainer jumps to the TrainerState.ReadyForCleanup and tries to upload the latest model.
    """


class ModelAlreadyUploadedError(CriticalError):
    """Raised when the loop already stores the model of a training in the format being uploaded.
    A caller that does not count the format as uploaded ends the training, as with any other rejected upload.
    """


class ModelUploadFailedError(Exception):
    """Raised when the loop could not store an uploaded model, e.g. because it answered with a server error.
    Unlike a CriticalError it keeps the training, so the upload is tried again later.
    """


class InsufficientMemoryError(RuntimeError):
    """Raised when not even the smallest unit of work fits in memory."""


class NodeNeedsRestartError(Exception):
    """
    NodeNeedsRestartError is raised when the node needs to be restarted.
    This is e.g. the case when the GPU is not available anymore.
    """
