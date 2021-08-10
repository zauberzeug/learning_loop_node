
import psutil
import os
import subprocess


class Executor:

    def __init__(self, base_path) -> None:
        self.path = base_path

    def start(self, cmd: str):
        # NOTE we have to write the pid inside the bash command to get the correct pid.
        p = subprocess.Popen(
            f'cd {self.path};nohup {cmd} >> last_training.log 2>&1 & echo $! > last_training.pid',
            shell=True
        )
        _, err = p.communicate()
        if p.returncode != 0:
            raise Exception(f'Failed to start training with error: {err}')

    def is_process_running(self):
        if self.training is None:
            return False
        pid_path = f'{self.path}/last_training.pid'
        if not os.path.exists(pid_path):
            return False
        with open(pid_path, 'r') as f:
            pid = f.read().strip()
        try:
            p = psutil.Process(int(pid))
        except psutil.NoSuchProcess as e:
            return False

        return True

    def get_log(self):
        with open(f'{self.path}/last_training.log') as f:
            return f.read()
