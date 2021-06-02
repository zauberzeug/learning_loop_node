from learning_loop_node.trainer.capability import Capability
from learning_loop_node.trainer.downloader_factory import DownloaderFactory
import pytest
from learning_loop_node.tests import test_helper
from learning_loop_node.trainer.tests import trainer_test_helper
from learning_loop_node.trainer.downloader import DataDownloader
from learning_loop_node import node, node_helper
from learning_loop_node.context import Context
from learning_loop_node.conftest import create_project
from icecream import ic
import asyncio


@pytest.fixture(autouse=True, scope='module')
def create_project_for_module():
    # TODO can we use the 'create_project' fixture here?
    test_helper.LiveServerSession().delete(f"/api/zauberzeug/projects/pytest?keep_images=true")
    project_configuration = {'project_name': 'pytest', 'inbox': 0, 'annotate': 0, 'review': 0, 'complete': 3, 'image_style': 'beautiful',
                             'categories': 2, 'thumbs': False, 'tags': 0, 'trainings': 1, 'detections': 3, 'annotations': 0, 'skeleton': False}
    assert test_helper.LiveServerSession().post(f"/api/zauberzeug/projects/generator",
                                                json=project_configuration).status_code == 200
    yield
    test_helper.LiveServerSession().delete(f"/api/zauberzeug/projects/pytest?keep_images=true")


@pytest.fixture
def downloader() -> DataDownloader:
    context = Context(organization='zauberzeug', project='pytest')
    return DownloaderFactory.create(context=context, capability=Capability.Box)


@pytest.mark.asyncio
async def test_download_model():
    _, _, trainings_folder = trainer_test_helper.create_needed_folders()
    model_id = await trainer_test_helper.assert_upload_model()

    await node_helper.download_model(trainings_folder,
                                     'zauberzeug', 'pytest', model_id, 'mocked')

    files = test_helper.get_files_in_folder('/data')
    assert len(files) == 2

    assert files[0] == "/data/zauberzeug/pytest/trainings/some_uuid/file_1.txt"
    assert files[1] == "/data/zauberzeug/pytest/trainings/some_uuid/file_2.txt"

    assert open(files[0], 'r').read() == 'content of file one'
    assert open(files[1], 'r').read() == 'content of file two'


@pytest.mark.asyncio
async def test_download_basic_data(downloader: DataDownloader):
    basic_data = await downloader.download_basic_data()

    assert len(basic_data.image_ids) == 3
    assert len(basic_data.box_categories) == 2


@pytest.mark.asyncio
async def test_download_images(downloader: DataDownloader):
    _, image_folder, _ = trainer_test_helper.create_needed_folders()

    basic_data = await downloader.download_basic_data()
    await downloader.download_images(asyncio.get_event_loop(), basic_data.image_ids, image_folder)
    files = test_helper.get_files_in_folder('/data')

    assert len(files) == 3


@pytest.mark.asyncio
async def test_download_training_data(downloader: DataDownloader):
    _, image_folder, _ = trainer_test_helper.create_needed_folders()
    basic_data = await downloader.download_basic_data()
    image_data = await downloader.download_image_data(basic_data.image_ids)

    assert len(image_data) == 3
