import os
import shutil

from ...data_classes import Context
from ...data_exchanger import DataExchanger
from ...globals import GLOBALS
from ...helpers.misc import delete_corrupt_images
from .. import test_helper

# Used by all Nodes


async def test_download_model(data_exchanger: DataExchanger):

    _, _, trainings_folder = create_needed_folders()
    model_id = await test_helper.get_latest_model_id(project='pytest_nodelib_general')

    await data_exchanger.download_model(trainings_folder, Context(organization='zauberzeug', project='pytest_nodelib_general'), model_id, 'mocked')

    files = test_helper.get_files_in_folder(GLOBALS.data_folder)
    assert len(files) == 3, str(files)

    file_1 = f'{GLOBALS.data_folder}/zauberzeug/pytest_nodelib_general/trainings/some_uuid/file_1.txt'
    file_2 = f'{GLOBALS.data_folder}/zauberzeug/pytest_nodelib_general/trainings/some_uuid/file_2.txt'
    model_json = f'{GLOBALS.data_folder}/zauberzeug/pytest_nodelib_general/trainings/some_uuid/model.json'

    assert file_1 in files
    assert file_2 in files
    assert model_json in files

    assert open(file_1, 'r').read() == 'content of file one'
    assert open(file_2, 'r').read() == 'content of file two'
    assert '"format": "mocked"' in open(model_json, 'r').read(), 'should have base_model.json'


# pylint: disable=redefined-outer-name
async def test_fetching_image_ids(data_exchanger: DataExchanger):
    ids = await data_exchanger.fetch_image_uuids()
    assert len(ids) == 3


async def test_download_images(data_exchanger: DataExchanger):
    _, image_folder, _ = create_needed_folders()
    image_ids = await data_exchanger.fetch_image_uuids()
    await data_exchanger.download_images(image_ids, image_folder)
    files = test_helper.get_files_in_folder(GLOBALS.data_folder)
    assert len(files) == 3


async def test_download_training_data(data_exchanger: DataExchanger):
    image_ids = await data_exchanger.fetch_image_uuids()
    image_data = await data_exchanger.download_images_data(image_ids)
    assert len(image_data) == 3


async def test_removal_of_corrupted_images(data_exchanger: DataExchanger):
    image_ids = await data_exchanger.fetch_image_uuids()

    shutil.rmtree('/tmp/img_folder', ignore_errors=True)
    os.makedirs('/tmp/img_folder', exist_ok=True)
    await data_exchanger.download_images(image_ids, '/tmp/img_folder')
    num_images = len(os.listdir('/tmp/img_folder'))

    # Generate two corrupted images
    with open('/tmp/img_folder/c0.jpg', 'w') as f:
        f.write('')
    with open('/tmp/img_folder/c1.jpg', 'w') as f:
        f.write('I am no image')

    await delete_corrupt_images('/tmp/img_folder', True)

    assert len(os.listdir('/tmp/img_folder')) == num_images if data_exchanger.check_jpeg else num_images - 1
    shutil.rmtree('/tmp/img_folder', ignore_errors=True)


# ---------------------- HELPERS


def create_needed_folders(training_uuid: str = 'some_uuid'):  # pylint: disable=unused-argument
    project_folder = test_helper.create_project_folder(
        Context(organization='zauberzeug', project='pytest_nodelib_general'))
    image_folder = test_helper.create_image_folder(project_folder)
    training_folder = test_helper.create_training_folder(project_folder, training_uuid)
    return project_folder, image_folder, training_folder
