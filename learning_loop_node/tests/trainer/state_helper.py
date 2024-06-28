import logging

from learning_loop_node.tests.test_helper import condition, update_attributes
from learning_loop_node.trainer.trainer_logic import TrainerLogic

from ...data_classes import Training


def create_active_training_file(trainer: TrainerLogic, **kwargs) -> None:
    update_attributes(trainer._training, **kwargs)  # pylint: disable=protected-access
    trainer.node.last_training_io.save(training=trainer.training)


async def assert_training_state(training: Training, state: str, timeout: float, interval: float) -> None:
    try:
        await condition(lambda: training.training_state == state, timeout=timeout, interval=interval)
    except TimeoutError as exc:
        msg = f"Trainer state should be '{state}' after {timeout} seconds, but is {training.training_state}"
        raise AssertionError(msg) from exc
    except Exception:
        logging.exception('##### was ist das hier?')
        raise
