from learning_loop_node.trainer.executor import Executor
from uuid import uuid4
from time import sleep
import os


def test_executor_lifecycle():
    executor = Executor('/tmp/test_executor/' + str(uuid4()))
    cmd = executor.path + '/some_executable.sh'
    with open(cmd, 'w') as f:
        f.write('while true; do echo "some output"; sleep 1; done')
    os.chmod(cmd, 0o755)
    executor.start(cmd)
    assert executor.is_process_running()
    sleep(1)
    assert 'some output' in executor.get_log()
    executor.stop()
    assert executor.is_process_running() == False
