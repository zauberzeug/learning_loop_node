class CriticalError(Exception):
    """
    CriticalError is raised when the training cannot be continued.
    In this case the trainer jumps to the TrainerState.ReadyForCleanup and tries to upload the latest model.
    """


class TrainingRefusedError(Exception):
    """Raised when a trainer cannot take on a training it was asked to begin; nothing was set up."""


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
