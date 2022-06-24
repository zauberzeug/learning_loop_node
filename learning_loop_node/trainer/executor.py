
from typing import List
import psutil
import os
import subprocess
import signal
from icecream import ic
import logging
import signal
from sys import platform


def create_signal_handler(sig=signal.SIGTERM):
    if platform == "linux" or platform == "linux2":
        # "The system will send a signal to the child once the parent exits for any reason (even sigkill)."
        # https://stackoverflow.com/a/19448096
        import ctypes
        libc = ctypes.CDLL("libc.so.6")

        def callable():
            os.setsid()
            return libc.prctl(1, sig)

        return callable
    return os.setsid


class Executor:

    def __init__(self, base_path) -> None:
        self.path = base_path
        os.makedirs(self.path, exist_ok=True)
        self.process = None

    def start(self, cmd: str):
        self.process = subprocess.Popen(
            f'cd {self.path}; {cmd} >> last_training.log 2>&1',
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            executable='/bin/bash',
            preexec_fn=create_signal_handler(),
        )

    def is_process_running(self):
        if self.process is None:
            return False

        if self.process.poll() is not None:
            return False

        try:
            psutil.Process(self.process.pid)
        except psutil.NoSuchProcess:
            return False

        return True

    def get_log(self) -> str:
        try:
            with open(f'{self.path}/last_training.log') as f:
                return f.read()
        except:
            return ''

    def get_log_by_lines(self) -> List[str]:
        try:
            with open(f'{self.path}/last_training.log') as f:
                return f.readlines()
        except:
            return []

    def stop(self):
        if self.process is None:
            logging.info('no process running ... nothing to stop')
            return

        logging.info('terminating process')

        try:
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass

        self.process.terminate()
        out, err = self.process.communicate(timeout=3)

    @property
    def return_code(self):
        if not self.process:
            return None
        if self.is_process_running():
            return None
        return self.process.poll()
