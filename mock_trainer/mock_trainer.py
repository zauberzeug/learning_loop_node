from learning_loop_node.trainer.error_configuration import ErrorConfiguration
from typing import List, Optional
from learning_loop_node.trainer.model import BasicModel
from learning_loop_node.trainer.trainer import Trainer
import progress_simulator
import time


class MockTrainer(Trainer):
    latest_known_confusion_matrix: dict = {}
    error_configuration: ErrorConfiguration = ErrorConfiguration()

    async def start_training(self) -> None:
        if self.error_configuration.begin_training:
            raise Exception()
        self.executor.start('while true; do sleep 1; done')

    def get_error(self) -> str:
        if self.error_configuration.crash_training:
            return 'mocked crash'

    def get_model_files(self, model_id) -> List[str]:
        if self.error_configuration.save_model:
            raise Exception()

        time.sleep(1)  # NOTE reduce flakyness in Backend tests du to wrong order of events.
        fake_weight_file = '/tmp/weightfile.weights'
        with open(fake_weight_file, 'wb') as f:
            f.write(b'\x42')

        more_data_file = '/tmp/some_more_data.txt'
        with open(more_data_file, 'w') as f:
            f.write('zweiundvierzig')

        return [fake_weight_file, more_data_file]

    def get_new_model(self) -> Optional[BasicModel]:
        if self.error_configuration.get_new_model:
            raise Exception()
        return progress_simulator.increment_time(self, self.latest_known_confusion_matrix)

    def on_model_published(self, basic_model: BasicModel, uuid: str) -> None:
        self.latest_known_confusion_matrix = basic_model.confusion_matrix

    def stop_training(self) -> None:
        if self.error_configuration.stop_training:
            raise Exception()
        self.executor.stop()
        self.latest_known_confusion_matrix = None
