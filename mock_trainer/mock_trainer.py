from learning_loop_node.trainer.error_configuration import ErrorConfiguration
from typing import List, Optional, Union
from learning_loop_node.trainer.model import BasicModel, PretrainedModel
from learning_loop_node.trainer.trainer import Trainer
import progress_simulator
import time
from learning_loop_node.model_information import ModelInformation
from learning_loop_node.detector.box_detection import BoxDetection
from learning_loop_node.detector.point_detection import PointDetection
from icecream import ic


class MockTrainer(Trainer):
    latest_known_confusion_matrix: dict = {}
    error_configuration: ErrorConfiguration = ErrorConfiguration()
    max_iterations = 100
    current_iteration = 0

    async def start_training(self) -> None:
        self.current_iteration = 0
        if self.error_configuration.begin_training:
            raise Exception()
        self.executor.start('while true; do sleep 1; done')

    async def start_training_from_scratch(self, identifier: str) -> None:
        self.current_iteration = 0
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

    async def _detect(self, model_information: ModelInformation, images:  List[str], model_folder: str) -> List:
        detections = []
        for image in images:
            image_id = image.split('/')[-1].replace('.jpg', '')

            box_detections = []
            point_detections = []
            image_entry = {'image_id': image_id, 'box_detections': box_detections, 'point_detections': point_detections}
            for c in model_information.categories:
                if c.type == 'box':
                    d = BoxDetection(c.name, x=1, y=2, width=30, height=40,
                                     net=model_information.version, confidence=.99, category_id=c.id)
                    box_detections.append(d)
                elif c.type == 'point':
                    d = PointDetection(c.name, x=100, y=200,
                                       net=model_information.version, confidence=.97, category_id=c.id)
                    point_detections.append(d)

            detections.append(image_entry)
        return detections

    async def clear_training_data(self, training_folder: str):
        pass

    @property
    def provided_pretrained_models(self) -> List[PretrainedModel]:
        return [
            PretrainedModel(name='small', label='Small', description='a small model'),
            PretrainedModel(name='medium', label='Medium', description='a medium model'),
            PretrainedModel(name='large', label='Large', description='a large model')]

    @property
    def progress(self) -> float:
        return self.current_iteration / self.max_iterations

    def get_new_model(self) -> Optional[BasicModel]:
        if self.error_configuration.get_new_model:
            raise Exception()
        self.current_iteration += 1
        return progress_simulator.increment_time(self, self.latest_known_confusion_matrix)

    def on_model_published(self, basic_model: BasicModel, uuid: str) -> None:
        self.latest_known_confusion_matrix = basic_model.confusion_matrix

    @property
    def model_architecture(self) -> Union[str, None]:
        return "mocked"
