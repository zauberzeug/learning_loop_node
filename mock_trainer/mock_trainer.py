
from dacite import from_dict
import asyncio
import time
from typing import Dict, List, Optional, Union

from learning_loop_node.data_classes import (BasicModel, CategoryType,
                                             ErrorConfiguration,
                                             ModelInformation, PretrainedModel)
from learning_loop_node.data_classes.detections import (
    BoxDetection, ClassificationDetection, Detections, Point, PointDetection,
    SegmentationDetection, Shape)
from learning_loop_node.trainer.trainer_logic import TrainerLogic
from mock_trainer import progress_simulator


class MockTrainer(TrainerLogic):
    latest_known_confusion_matrix: Dict = {}
    error_configuration: ErrorConfiguration = ErrorConfiguration()
    max_iterations = 100
    current_iteration = 0
    provide_new_model = True

    def can_resume(self) -> bool:
        return False

    async def resume(self) -> None:
        pass

    async def start_training(self) -> None:
        self.current_iteration = 0
        if self.error_configuration.begin_training:
            raise Exception()
        self.executor.start('while true; do sleep 1; done')

    async def start_training_from_scratch(self, identifier: str) -> None:
        self.current_iteration = 0
        self.executor.start('while true; do sleep 1; done')

    def get_executor_error_from_log(self) -> Optional[str]:
        if self.error_configuration.crash_training:
            return 'mocked crash'
        return None

    def get_latest_model_files(self) -> Union[List[str], Dict[str, List[str]]]:
        if self.error_configuration.save_model:
            raise Exception()

        time.sleep(1)  # NOTE reduce flakyness in Backend tests du to wrong order of events.
        fake_weight_file = '/tmp/weightfile.weights'
        with open(fake_weight_file, 'wb') as f:
            f.write(b'\x42')

        more_data_file = '/tmp/some_more_data.txt'
        with open(more_data_file, 'w') as f:
            f.write('zweiundvierzig')
        return {'mocked': [fake_weight_file, more_data_file], 'mocked_2': [fake_weight_file, more_data_file]}

    async def _detect(self, model_information: ModelInformation, images:  List[str], model_folder: str) -> List[Detections]:
        detections = []

        await asyncio.sleep(1)

        for image in images:
            image_id = image.split('/')[-1].replace('.jpg', '')

            box_detections = []
            point_detections = []
            segmentation_detections = []
            classification_detections = []
            det_entry = {
                'image_id': image_id, 'box_detections': box_detections, 'point_detections': point_detections,
                'segmentation_detections': segmentation_detections,
                'classification_detections': classification_detections}
            for c in model_information.categories:
                if c.type == CategoryType.Box:
                    d = BoxDetection(c.name, x=1, y=2, width=30, height=40,
                                     model_name=model_information.version, confidence=.99, category_id=c.id)
                    box_detections.append(d)
                elif c.type == CategoryType.Point:
                    d = PointDetection(c.name, x=100, y=200,
                                       model_name=model_information.version, confidence=.97, category_id=c.id)
                    point_detections.append(d)
                elif c.type == CategoryType.Segmentation:
                    d = SegmentationDetection(c.name, shape=Shape(points=[Point(x=1, y=2), Point(
                        x=3, y=4)]), model_name=model_information.version, confidence=.96, category_id=c.id)
                    segmentation_detections.append(d)
                elif c.type == CategoryType.Classification:
                    d = ClassificationDetection(c.name, model_name=model_information.version,
                                                confidence=.95, category_id=c.id)
                    classification_detections.append(d)
            detections.append(from_dict(data_class=Detections, data=det_entry))
            # det = Detections(box_detections=box_detections, point_detections=point_detections,
            #                  segmentation_detections=segmentation_detections,
            #                  classification_detections=classification_detections,
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
        if not self.provide_new_model:
            return None
        self.current_iteration += 1
        return progress_simulator.increment_time(self, self.latest_known_confusion_matrix)

    def on_model_published(self, basic_model: BasicModel) -> None:
        assert isinstance(basic_model.confusion_matrix, Dict)
        self.latest_known_confusion_matrix = basic_model.confusion_matrix

    @property
    def model_architecture(self) -> Union[str, None]:
        return "mocked"
