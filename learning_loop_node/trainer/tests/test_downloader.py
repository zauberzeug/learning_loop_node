from learning_loop_node.globals import GLOBALS
from learning_loop_node.trainer.downloader_factory import DownloaderFactory
import pytest
from learning_loop_node.tests import test_helper
from learning_loop_node.trainer.tests import trainer_test_helper
from learning_loop_node.trainer.downloader import DataDownloader
from learning_loop_node import node_helper
from learning_loop_node.context import Context
from icecream import ic


@pytest.fixture(autouse=True, scope='module')
def create_project_for_module():
    # TODO can we use the 'create_project' fixture here?
    test_helper.LiveServerSession().delete(f"/api/zauberzeug/projects/pytest?keep_images=true")
    project_configuration = {'project_name': 'pytest', 'inbox': 0, 'annotate': 0, 'review': 0, 'complete': 3, 'image_style': 'beautiful',
                             'box_categories': 2, 'point_categories': 2, 'segmentation_categories': 2, 'thumbs': False, 'tags': 0, 'trainings': 1, 'box_detections': 3, 'box_annotations': 0}
    assert test_helper.LiveServerSession().post(f"/api/zauberzeug/projects/generator",
                                                json=project_configuration).status_code == 200
    yield
    test_helper.LiveServerSession().delete(f"/api/zauberzeug/projects/pytest?keep_images=true")


@pytest.fixture
def downloader() -> DataDownloader:
    context = Context(organization='zauberzeug', project='pytest')
    return DownloaderFactory.create(context=context)


@pytest.mark.asyncio
async def test_download_model():
    data_folder = GLOBALS.data_folder
    _, _, trainings_folder = trainer_test_helper.create_needed_folders(data_folder)
    model_id = await trainer_test_helper.assert_upload_model()

    await node_helper.download_model(trainings_folder, Context(organization='zauberzeug', project='pytest'), model_id, 'mocked')

    files = test_helper.get_files_in_folder(data_folder)
    assert len(files) == 2, str(files)

    assert files[0] == data_folder + "/zauberzeug/pytest/trainings/some_uuid/file_1.txt"
    assert files[1] == data_folder + "/zauberzeug/pytest/trainings/some_uuid/file_2.txt"

    assert open(files[0], 'r').read() == 'content of file one'
    assert open(files[1], 'r').read() == 'content of file two'


@pytest.mark.asyncio
async def test_download_basic_data(downloader: DataDownloader):
    basic_data = await downloader.download_basic_data()

    assert len(basic_data.image_ids) == 3
    assert len(basic_data.categories) == 6, 'Two box, two segmentation and two point categories'


@pytest.mark.asyncio
async def test_download_images(downloader: DataDownloader):
    _, image_folder, _ = trainer_test_helper.create_needed_folders(GLOBALS.data_folder)

    basic_data = await downloader.download_basic_data()
    await downloader.download_images(basic_data.image_ids, image_folder)
    files = test_helper.get_files_in_folder(GLOBALS.data_folder)

    assert len(files) == 3


@pytest.mark.asyncio
async def test_download_training_data(downloader: DataDownloader):
    basic_data = await downloader.download_basic_data()
    image_data = await downloader.download_image_data(basic_data.image_ids)

    assert len(image_data) == 3
