import logging

from learning_loop_node.data_classes import Context, Training
from learning_loop_node.tests.test_helper import condition, update_attributes
from learning_loop_node.trainer.tests.testing_trainer import TestingTrainer


def create_active_training_file(**kwargs) -> None:
    trainer = TestingTrainer()
    details = {'categories': [],
               'id': '917d5c7f-403d-7e92-f95f-577f79c2273a',  # version 1.2 of demo project
               'training_number': 0,
               'resolution': 800,
               'flip_rl': False,
               'flip_ud': False}
    trainer.init(Context(organization='zauberzeug', project='demo'),
                 details, node_uuid='00000000-0000-0000-0000-000000000000')

    assert trainer._training is not None
    assert trainer._active_training_io is not None

    update_attributes(trainer._training, **kwargs)
    trainer._active_training_io.save(training=trainer._training)


async def assert_training_state(training: Training, state: str, timeout: float, interval: float) -> None:
    try:
        await condition(lambda: training.training_state == state, timeout=timeout, interval=interval)
    except TimeoutError as exc:
        msg = f"Trainer state should be '{state}' after {timeout} seconds, but is {training.training_state}"
        raise AssertionError(msg) from exc
    except Exception:
        logging.exception('##### was ist das hier?')
        raise
