from learning_loop_node.trainer.capability import Capability
from learning_loop_node.node import Node
from learning_loop_node.trainer.downloader_factory import DownloaderFactory
from learning_loop_node.trainer.downloader import Downloader
from learning_loop_node.context import Context
from learning_loop_node.trainer.trainer import Trainer
import shutil
import pytest
import main
import darknet_tests.test_helper as test_helper
import model_updater
import learning_loop_node.trainer.tests.trainer_test_helper as trainer_test_helper
from learning_loop_node import node
from darknet_trainer import DarknetTrainer


@pytest.fixture
def downloader() -> Downloader:
    context = Context(organization='zauberzeug', project='pytest')
    return DownloaderFactory.create(server_base_url=node.SERVER_BASE_URL_DEFAULT, headers={}, context=context, capability=Capability.Box)


@pytest.mark.asyncio
async def test_parse_latest_confusion_matrix(downloader: Downloader):
    model_id = trainer_test_helper.assert_upload_model(
        ['darknet_tests/test_data/tiny_yolo.cfg', 'darknet_tests/test_data/fake_weightfile.weights'])
    context = Context(organization='zauberzeug', project='pytest')
    training = Trainer.generate_training(context, {'id': 'some_uuid'})
    training.data = await downloader.download_data(training.images_folder, training.training_folder, model_id)

    shutil.copy('darknet_tests/test_data/last_training.log', f'{training.training_folder}/last_training.log')

    new_model = model_updater._parse_latest_iteration(training.id, training.data)
    assert new_model
    assert new_model['iteration'] == 1089
    confusion_matrix = new_model['confusion_matrix']
    assert len(confusion_matrix) == 2
    purple_matrix = confusion_matrix[training.data.box_categories[0]['id']]

    assert purple_matrix['ap'] == 42
    assert purple_matrix['tp'] == 1
    assert purple_matrix['fp'] == 2
    assert purple_matrix['fn'] == 3

    weightfile = new_model['weightfile']
    assert weightfile == 'backup/tiny_yolo_best_mAP_0.000000_iteration_1089_avgloss_-nan_.weights'


@pytest.mark.parametrize("filename,is_valid_model", [
    ('last_training.log', True),
    ('last_training_missing_weightfile.log', False),
    ('last_training_missing_confusion_matrix.log', False)
])
@pytest.mark.asyncio
async def test_iteration_needs_weightfile_to_be_valid(filename: str, is_valid_model: bool, downloader: Downloader):
    model_id = trainer_test_helper.assert_upload_model(
        ['darknet_tests/test_data/tiny_yolo.cfg', 'darknet_tests/test_data/fake_weightfile.weights'])
    context = Context(organization='zauberzeug', project='pytest')
    training = Trainer.generate_training(context, {'id': 'some_uuid'})
    training.data = await downloader.download_data(training.images_folder, training.training_folder, model_id)
    trainer = DarknetTrainer(capability=Capability.Box)
    trainer.training = training

    shutil.copy(f'darknet_tests/test_data/{filename}', f'{training.training_folder}/last_training.log')

    new_model = trainer.get_new_model()
    assert is_valid_model == (new_model is not None)


def get_box_categories():
    content = test_helper.LiveServerSession().get('/api/zauberzeug/projects/pytest/data/data2').json()
    categories = content['box_categories']
    return categories


def get_model_ids_from__latest_training():
    content = test_helper.LiveServerSession().get('/api/zauberzeug/projects/pytest/trainings')
    content_json = content.json()
    datapoints = [datapoint['model_id'] for datapoint in content_json['charts'][0]['data']]
    return datapoints
