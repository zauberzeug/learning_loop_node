from learning_loop_node.trainer.executor import Executor
from uuid import uuid4
from time import sleep
import os


def test_executor_lifecycle():
    executor = Executor('/tmp/test_executor/' + str(uuid4()))
    cmd = executor.path + '/some_executable.sh'
    with open(cmd, 'w') as f:
        f.write('while true; do ls; sleep 1; done')
    os.chmod(cmd, 0o755)
    executor.start(cmd)
    assert executor.is_process_running()
    sleep(2)
    assert 'last_training.log' in executor.get_log()
    executor.stop()
    assert executor.is_process_running() == False
