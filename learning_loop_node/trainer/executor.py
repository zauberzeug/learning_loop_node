import asyncio
import logging
import os
import shlex
from io import BufferedWriter
from typing import List, Optional, Dict


class Executor:
    def __init__(self, base_path: str, log_name='last_training.log') -> None:
        """An executor that runs a command in a separate async subprocess.
        The log of the process is written to 'last_training.log' in the base_path.
        Tthe process is executed in the base_path directory.
        The process should be awaited to finish using `wait` or stopped using `stop` to 
        avoid zombie processes and close the log file."""

        self.path = base_path
        self.log_file_path = f'{self.path}/{log_name}'
        self.log_file: Optional[BufferedWriter] = None
        self._process: Optional[asyncio.subprocess.Process] = None  # pylint: disable=no-member
        os.makedirs(self.path, exist_ok=True)

    def _get_running_process(self) -> Optional[asyncio.subprocess.Process]:  # pylint: disable=no-member
        """Get the running process if available."""
        if self._process is not None and self._process.returncode is None:
            return self._process
        return None

    async def start(self, cmd: str, env: Optional[Dict[str, str]] = None) -> None:
        """Start the process with the given command and environment variables."""

        full_env = os.environ.copy()
        if env is not None:
            full_env.update(env)

        logging.info(f'Starting executor with command: {cmd} in {self.path} - logging to {self.log_file_path}')
        self.log_file = open(self.log_file_path, 'ab')

        self._process = await asyncio.create_subprocess_exec(
            *shlex.split(cmd),
            cwd=self.path,
            stdout=self.log_file,
            stderr=asyncio.subprocess.STDOUT,  # Merge stderr with stdout
            env=full_env
        )

    def is_running(self) -> bool:
        """Check if the process is still running."""
        return self._process is not None and self._process.returncode is None

    def terminate(self) -> None:
        """Terminate the process."""

        if process := self._get_running_process():
            try:
                process.terminate()
                return
            except ProcessLookupError:
                logging.error('No process to terminate')
        self._process = None

    async def wait(self) -> Optional[int]:
        """Wait for the process to finish. Returns the return code of the process or None if no process is running."""

        if not self._process:
            logging.info('No process to wait for')
            return None

        return_code = await self._process.wait()

        self.close_log()
        self._process = None

        return return_code

    async def stop_and_wait(self) -> Optional[int]:
        """Terminate the process and wait for it to finish. Returns the return code of the process."""

        if not self.is_running():
            logging.info('No process to stop')
            return None

        self.terminate()
        return await self.wait()

    # -------------------------------------------------------------------------------------------- LOGGING

    def get_log(self) -> str:
        """Get the log of the process as a string."""
        if not os.path.exists(self.log_file_path):
            return ''
        with open(self.log_file_path, 'r') as f:
            return f.read()

    def get_log_by_lines(self, tail: Optional[int] = None) -> List[str]:
        """Get the log of the process as a list of lines."""
        if not os.path.exists(self.log_file_path):
            return []
        with open(self.log_file_path) as f:
            lines = f.readlines()
        if tail is not None:
            lines = lines[-tail:]
        return lines

    def close_log(self):
        """Close the log file."""
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None
