from learning_loop_node.trainer.executor import Executor
from uuid import uuid4
import os
from time import sleep
import pytest
from icecream import ic
import subprocess
import psutil


@pytest.fixture(autouse=True)
def cleanup():
    cleanup = subprocess.Popen(
        "ps aux | grep some_executable.sh | awk '{print $2}'  | xargs kill",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        executable='/bin/bash',
    )
    cleanup.communicate()


def test_executor_lifecycle():
    assert_process_is_running('some_executable.sh', False)

    executor = Executor('/tmp/test_executor/' + str(uuid4()))
    cmd = executor.path + '/some_executable.sh'
    with open(cmd, 'w') as f:
        f.write('while true; do echo "some output"; sleep 1; done')
    os.chmod(cmd, 0o755)

    executor.start(cmd)

    assert executor.is_process_running()
    assert_process_is_running('some_executable.sh')

    sleep(1)
    assert 'some output' in executor.get_log()

    executor.stop()

    assert executor.is_process_running() == False
    sleep(1)
    assert_process_is_running('some_executable.sh', False)


def assert_process_is_running(process_name, running=True):
    if running:
        for process in psutil.process_iter():
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
