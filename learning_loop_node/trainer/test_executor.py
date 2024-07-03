import os
import subprocess
from time import sleep
from uuid import uuid4

import psutil
import pytest

from .executor import Executor


@pytest.fixture(autouse=True)
def cleanup():
    cleanup_process = subprocess.Popen(
        "ps aux | grep some_executable.sh | awk '{print $2}'  | xargs kill",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        executable='/bin/bash',
    )
    cleanup_process.communicate()


@pytest.mark.asyncio
async def test_executor_lifecycle():
    assert_process_is_running('some_executable.sh', False)

    executor = Executor('/tmp/test_executor/' + str(uuid4())+'/')
    cmd = 'bash some_executable.sh'
    executable_path = executor.path+'some_executable.sh'
    with open(executable_path, 'w') as f:
        f.write('/bin/bash -c "while true; do sleep 1; echo some output; done"')
    os.chmod(executable_path, 0o755)

    await executor.start(cmd)

    assert executor.is_running()
    assert_process_is_running('some_executable.sh')

    sleep(5)
    assert 'some output' in executor.get_log()

    await executor.stop_and_wait()

    assert not executor.is_running()
    sleep(1)
    # NOTE: It happend that this process became a zombie process which leads to repeated test failures -> restart machine required
    assert_process_is_running('some_executable.sh', False)


def assert_process_is_running(process_name, running=True):
    if running:
        for process in psutil.process_iter():
            print(process.name(), process.cmdline())
            process_name_match = process_name in process.name()
            process_cmd_match = process_name in str(process.cmdline())
            if process_name_match or process_cmd_match:
                return
        assert False, f"Process '{process_name}' should be runnning."

    if not running:
        for process in psutil.process_iter():
            process_name_match = process_name in process.name()
            process_cmd_match = process_name in str(process.cmdline())
            if process_name_match or process_cmd_match:
                assert False, f"Process '{process_name} should not be running. Process name : '{process.name()}', process command line: {process.cmdline()}"
