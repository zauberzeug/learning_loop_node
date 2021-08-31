
import psutil
import os
import subprocess
import signal
from icecream import ic


class Executor:

    def __init__(self, base_path) -> None:
        self.path = base_path
        os.makedirs(self.path, exist_ok=True)
        self.process = None

    def start(self, cmd: str):
        # NOTE we have to write the pid inside the bash command to get the correct pid.
        self.process = subprocess.Popen(
            f'cd {self.path}; {cmd} >> last_training.log 2>&1',
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            executable='/bin/bash',
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
        try:
            logging.info('terminating process')
            os.kill(self.process.pid, signal.SIGTERM)
            self.process.terminate()
            out, err = self.process.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            os.kill(self.process.pid, signal.SIGKILL)
            logging.info('could not terminate process -- trying to kill')
            self.process.kill()
            out, err = self.process.communicate()
