from learning_loop_node.globals import GLOBALS
import pytest
from learning_loop_node.tests import test_helper
from learning_loop_node.trainer.tests import trainer_test_helper
from learning_loop_node.tests import test_helper
from learning_loop_node.trainer.downloader import DataDownloader
from learning_loop_node.context import Context
from icecream import ic
import learning_loop_node.rest.downloads as downloads


@pytest.fixture(autouse=True, scope='module')
def create_project_for_module():
    # TODO can we use the 'create_project' fixture here?
    test_helper.LiveServerSession().delete(
        f"/api/zauberzeug/projects/pytest?keep_images=true")
    project_configuration = {'project_name': 'pytest', 'inbox': 0, 'annotate': 0, 'review': 0, 'complete': 3, 'image_style': 'beautiful',
                             'box_categories': 2, 'point_categories': 2, 'segmentation_categories': 2, 'thumbs': False, 'tags': 0, 'trainings': 1, 'box_detections': 3, 'box_annotations': 0}
    assert test_helper.LiveServerSession().post(f"/api/zauberzeug/projects/generator",
                                                json=project_configuration).status_code == 200
    yield
    test_helper.LiveServerSession().delete(
        f"/api/zauberzeug/projects/pytest?keep_images=true")


@pytest.fixture
def data_downloader() -> DataDownloader:
    context = Context(organization='zauberzeug', project='pytest')
    return DataDownloader(context)


async def test_download_model():
    data_folder = GLOBALS.data_folder
    _, _, trainings_folder = trainer_test_helper.create_needed_folders(
        data_folder)
    model_id = await test_helper.get_latest_model_id()

    await downloads.download_model(trainings_folder, Context(organization='zauberzeug', project='pytest'), model_id, 'mocked')

    files = test_helper.get_files_in_folder(data_folder)
    assert len(files) == 3, str(files)

    file_1 = f'{data_folder}/zauberzeug/pytest/trainings/some_uuid/file_1.txt'
    file_2 = f'{data_folder}/zauberzeug/pytest/trainings/some_uuid/file_2.txt'
    model_json = f'{data_folder}/zauberzeug/pytest/trainings/some_uuid/model.json'

    assert file_1 in files
    assert file_2 in files
    assert model_json in files

    assert open(file_1, 'r').read() == 'content of file one'
    assert open(file_2, 'r').read() == 'content of file two'
    assert '"format": "mocked"' in open(model_json, 'r').read(), 'should have base_model.json'


async def test_fetching_image_ids(data_downloader: DataDownloader):
    ids = await data_downloader.fetch_image_ids()
    assert len(ids) == 3


async def test_download_images(data_downloader: DataDownloader):
    _, image_folder, _ = trainer_test_helper.create_needed_folders(
        GLOBALS.data_folder)
    image_ids = await data_downloader.fetch_image_ids()
    await data_downloader.download_images(image_ids, image_folder)
    files = test_helper.get_files_in_folder(GLOBALS.data_folder)
    assert len(files) == 3


async def test_download_training_data(data_downloader: DataDownloader):
    image_ids = await data_downloader.fetch_image_ids()
    image_data = await data_downloader.download_images_data(image_ids)
    assert len(image_data) == 3
