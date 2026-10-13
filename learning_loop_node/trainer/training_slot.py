"""The right to use the GPU, shared between the trainer nodes on one machine.

A node holds the slot from the moment it accepts a training until the training is cleared.
"""
import fcntl
import logging
import os
import weakref
from abc import ABC, abstractmethod
from pathlib import Path

from .exceptions import TrainingSlotTakenError

ENV_VAR = 'TRAINING_SLOT_LOCK'

logger = logging.getLogger('learning_loop_node.training_slot')


class TrainingSlot(ABC):

    @abstractmethod
    def acquire(self, holder: str) -> None:
        """Take the slot without waiting; a no-op if this slot already holds it.

        :raise TrainingSlotTakenError: if a sibling holds the slot.
        """

    @abstractmethod
    def release(self) -> None:
        """Give the slot back; a no-op if this slot does not hold it."""

    @property
    @abstractmethod
    def held(self) -> bool:
        """Whether this slot object holds the slot."""

    @abstractmethod
    def holder(self) -> str | None:
        """Who holds the slot, None if it is free or cannot be inspected."""

class AlwaysFreeSlot(TrainingSlot):
    """The slot of a node that has the GPU to itself."""

    def __init__(self) -> None:
        self._holder: str | None = None

    def acquire(self, holder: str) -> None:
        self._holder = holder

    def release(self) -> None:
        self._holder = None

    @property
    def held(self) -> bool:
        return self._holder is not None

    def holder(self) -> str | None:
        return self._holder

class FileTrainingSlot(TrainingSlot):
    """A slot backed by an exclusive `flock` on a file every sibling node mounts.

    The kernel drops the lock with the process, so a node that dies in any way frees the slot.
    The lock belongs to the inode: the file is only ever truncated and rewritten in place, never
    replaced, and the mounted path is its directory.

    The file's content names the holder and is diagnostic only — `holder()` tells a free slot
    from a taken one by trying the lock, and that probe holds the lock for an instant, so an
    acquirer that loses against a probe simply tries again. The file is made writable for everyone,
    since sibling containers may run as different users.

    A forked child shares the parent's lock and would keep it alive after the parent releases it,
    so the child closes its copy right after the fork; the parent's lock is unaffected.
    """

    _instances: 'weakref.WeakSet[FileTrainingSlot]' = weakref.WeakSet()

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._fd: int | None = None
        self._holder: str | None = None
        FileTrainingSlot._instances.add(self)

    def acquire(self, holder: str) -> None:
        if self._fd is not None:
            return
        fd = self._open()
        try:
            self._lock(fd)
            os.ftruncate(fd, 0)
            os.write(fd, holder.encode())
        except BaseException:
            os.close(fd)
            raise
        self._fd = fd
        self._holder = holder

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            os.ftruncate(self._fd, 0)
        except OSError:
            logger.warning('cannot clear the holder from training slot %s', self.path, exc_info=True)
        finally:
            os.close(self._fd)
            self._fd = None
            self._holder = None

    @property
    def held(self) -> bool:
        return self._fd is not None

    def holder(self) -> str | None:
        if self._fd is not None:
            return self._holder
        try:
            fd = os.open(self.path, os.O_RDONLY)
        except FileNotFoundError:
            return None
        except OSError:
            logger.warning('cannot inspect training slot %s - treating it as free', self.path, exc_info=True)
            return None
        try:
            self._lock(fd)
        except TrainingSlotTakenError as e:
            return e.holder
        finally:
            os.close(fd)
        return None

    @classmethod
    def _drop_inherited_locks(cls) -> None:
        for slot in list(cls._instances):
            if slot._fd is not None:
                os.close(slot._fd)
                slot._fd = None
                slot._holder = None

    def _open(self) -> int:
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o666)
        try:
            os.fchmod(fd, 0o666)
        except PermissionError:
            pass
        return fd

    @staticmethod
    def _lock(fd: int) -> None:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise TrainingSlotTakenError(os.read(fd, 4096).decode(errors='replace') or 'unknown') from e

os.register_at_fork(after_in_child=FileTrainingSlot._drop_inherited_locks)  # pylint: disable=protected-access


def training_slot_from_env() -> TrainingSlot:
    """The slot `TRAINING_SLOT_LOCK` names, or a free one if the variable is unset.

    :raise FileNotFoundError: if the lock file's directory does not exist.
    """
    path = os.environ.get(ENV_VAR)
    if not path:
        return AlwaysFreeSlot()
    if not Path(path).parent.is_dir():
        raise FileNotFoundError(f'{ENV_VAR}={path}: the directory does not exist; mount it from the host')
    return FileTrainingSlot(path)
