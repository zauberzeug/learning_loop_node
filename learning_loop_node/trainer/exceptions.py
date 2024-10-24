class CriticalError(Exception):
    '''
    CriticalError is raised when the training cannot be continued.
    In this case the trainer jumps to the TrainerState.ReadyForCleanup and tries to upload the latest model.
    '''


class NodeNeedsRestartError(Exception):
    '''
    NodeNeedsRestartError is raised when the node needs to be restarted.
    This is e.g. the case when the GPU is not available anymore.
    '''
