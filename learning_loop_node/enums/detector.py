from enum import Enum


class VersionMode(str, Enum):
    FollowLoop = 'follow_loop'  # will follow the loop
    SpecificVersion = 'specific_version'  # will follow the specific version
    Pause = 'pause'  # will pause the updates


class OperationMode(str, Enum):
    Startup = 'startup'  # used until model is loaded
    Idle = 'idle'  # will check and perform updates
    Detecting = 'detecting'  # Blocks updates


class OutboxMode(Enum):
    CONTINUOUS_UPLOAD = 'continuous_upload'
    STOPPED = 'stopped'
