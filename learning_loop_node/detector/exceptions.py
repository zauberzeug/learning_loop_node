

class NodeNeedsRestartError(Exception):
    '''
    NodeNeedsRestartError is raised when the node needs to be restarted.
    This is e.g. the case when the GPU is not available anymore.
    '''
