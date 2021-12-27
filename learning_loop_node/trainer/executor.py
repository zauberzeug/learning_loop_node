
import psutil
import os
import subprocess
import signal
from icecream import ic
import logging


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
            preexec_fn=os.setsid
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

    def get_log(self):
        with open(f'{self.path}/last_training.log') as f:
            return f.read()

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
