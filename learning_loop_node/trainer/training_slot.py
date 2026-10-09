"""The right to use the GPU, shared between the trainer nodes on one machine.

A node holds the slot from the moment it starts a training until the training is cleared;
siblings that find the slot taken report `busy` and a sibling that was handed a training
waits for the slot instead of competing for the card.
"""
import fcntl
import os
from abc import ABC, abstractmethod
from pathlib import Path

ENV_VAR = 'TRAINING_SLOT_LOCK'


class TrainingSlot(ABC):

    @abstractmethod
    def try_acquire(self, holder: str) -> bool:
        """Take the slot without waiting; True if this slot now holds it (also if it already did)."""

    @abstractmethod
    def release(self) -> None:
        """Give the slot back; a no-op if this slot does not hold it."""

    @property
    @abstractmethod
    def held(self) -> bool:
        """Whether this slot object holds the slot."""

    @abstractmethod
    def holder(self) -> str | None:
        """Who holds the slot, None if it is free."""

class AlwaysFreeSlot(TrainingSlot):
    """The slot of a node that has the GPU to itself."""

    def __init__(self) -> None:
        self._holder: str | None = None

    def try_acquire(self, holder: str) -> bool:
        self._holder = holder
        return True

    def release(self) -> None:
        self._holder = None

    @property
    def held(self) -> bool:
        return self._holder is not None

    def holder(self) -> str | None:
        return self._holder

class FileTrainingSlot(TrainingSlot):
    """A slot backed by an exclusive `flock` on a file every sibling node mounts.

    The kernel drops the lock with the process, so a node that dies in any way frees the slot
    without a heartbeat or a timeout. The lock belongs to the inode: the file is only ever
    truncated and rewritten in place, never replaced, and the mounted path is its directory.

    The file's content names the holder and is diagnostic only — `holder()` tells a free slot
    from a taken one by trying the lock, and that probe holds the lock for an instant, so an
    acquirer that loses against a probe simply tries again.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._fd: int | None = None
        self._holder: str | None = None

    def try_acquire(self, holder: str) -> bool:
        if self._fd is not None:
            return True
        fd = self._open()
        if not self._try_lock(fd):
            os.close(fd)
            return False
        os.ftruncate(fd, 0)
        os.write(fd, holder.encode())
        self._fd = fd
        self._holder = holder
        return True

    def release(self) -> None:
        if self._fd is None:
            return
        os.ftruncate(self._fd, 0)
        os.close(self._fd)
        self._fd = None
        self._holder = None

    @property
    def held(self) -> bool:
        return self._fd is not None

    def holder(self) -> str | None:
        if self._fd is not None:
            return self._holder
        fd = self._open()
        try:
            if self._try_lock(fd):
                return None
            return os.read(fd, 4096).decode(errors='replace') or 'unknown'
        finally:
            os.close(fd)

    def _open(self) -> int:
        return os.open(self.path, os.O_RDWR | os.O_CREAT, 0o666)

    @staticmethod
    def _try_lock(fd: int) -> bool:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        return True

def training_slot_from_env() -> TrainingSlot:
    """The slot `TRAINING_SLOT_LOCK` names, or a free one if the variable is unset.

    :raise FileNotFoundError: if the lock file's directory does not exist — it is meant to be a
        mount shared with the sibling nodes, and a lock in a directory of the node's own would
        hold nobody off.
    """
    path = os.environ.get(ENV_VAR)
    if not path:
        return AlwaysFreeSlot()
    if not Path(path).parent.is_dir():
        raise FileNotFoundError(f'{ENV_VAR}={path}: the directory does not exist; mount it from the host')
    return FileTrainingSlot(path)
