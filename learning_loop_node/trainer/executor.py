
import ctypes
import logging
import os
import signal
import subprocess
from sys import platform
from typing import List, Optional

import psutil


def create_signal_handler(sig=signal.SIGTERM):
    if platform == "linux" or platform == "linux2":
        # "The system will send a signal to the child once the parent exits for any reason (even sigkill)."
        # https://stackoverflow.com/a/19448096
        libc = ctypes.CDLL("libc.so.6")

        def callable_():
            os.setsid()
            return libc.prctl(1, sig)

        return callable_
    return os.setsid


class Executor:
    def __init__(self, base_path: str) -> None:
        self.path = base_path
        os.makedirs(self.path, exist_ok=True)
        self.process: Optional[subprocess.Popen[bytes]] = None

    def start(self, cmd: str):
        with open(f'{self.path}/last_training.log', 'a') as f:
            f.write(f'\nStarting executor with command: {cmd}\n')
        # pylint: disable=subprocess-popen-preexec-fn
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
            # self.process.terminate() # TODO does this make sense?
            # self.process = None
            return False

        return True

    def get_log(self) -> str:
        try:
            with open(f'{self.path}/last_training.log') as f:
                return f.read()
        except Exception:
            return ''

    def get_log_by_lines(self, since_last_start=False) -> List[str]:  # TODO do not read whole log again
        try:
            with open(f'{self.path}/last_training.log') as f:
                lines = f.readlines()
            if since_last_start:
                lines_since_last_start = []
                for line in reversed(lines):
                    lines_since_last_start.append(line)
                    if line.startswith('Starting executor with command:'):
                        break
                return list(reversed(lines_since_last_start))
            return lines
        except Exception:
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
        _, _ = self.process.communicate(timeout=3)

    @property
    def return_code(self):
        if not self.process:
            return None
        if self.is_process_running():
            return None
        return self.process.poll()
