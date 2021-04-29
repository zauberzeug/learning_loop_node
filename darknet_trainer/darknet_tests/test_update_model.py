import shutil
import pytest
import main
import darknet_tests.test_helper as test_helper
from uuid import uuid4
import yolo_helper
import os
import asyncio
import model_updater
from learning_loop_node.trainer.training import Training
from learning_loop_node.trainer.model import Model


@pytest.fixture(autouse=True, scope="module")
def event_loop():
    loop = asyncio.get_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True, scope="module")
async def connect_node_fixture():
    await main.node.connect()
    yield
    await main.node.sio.disconnect()


@pytest.mark.asyncio
async def test_parse_latest_confusion_matrix():
    training_uuid = "some_uuid"
    model_id = test_helper.assert_upload_model()
    main.node.training = Training(id=training_uuid,
                                  base_model=Model(id=model_id),
                                  organization='zauberzeug',
                                  project='pytest',
                                  project_folder="",
                                  images_folder="",
                                  training_folder=""
                                  )

    _, image_folder, training_path = test_helper.create_needed_folders(training_uuid)
    data = test_helper.get_data2()
    main.node.training.data = await main.download_data(main.node, data, image_folder, training_path)

    shutil.copy('darknet_tests/test_data/last_training.log', f'{training_path}/last_training.log')

    new_model = model_updater._parse_latest_iteration(training_uuid, main.node)
    assert new_model
    assert new_model['iteration'] == 1089
    confusion_matrix = new_model['confusion_matrix']
    assert len(confusion_matrix) == 2
    purple_matrix = confusion_matrix[main.node.training.data.box_categories[0]['id']]

    assert purple_matrix['ap'] == 42
    assert purple_matrix['tp'] == 1
    assert purple_matrix['fp'] == 2
    assert purple_matrix['fn'] == 3

    weightfile = new_model['weightfile']
    assert weightfile == 'backup//tiny_yolo_best_mAP_0.000000_iteration_1089_avgloss_-nan_.weights'


@pytest.mark.asyncio
async def test_model_is_updated():
    model_id = test_helper.assert_upload_model()

    model_ids = get_model_ids_from__latest_training()
    assert(len(model_ids)) == 1

    training_uuid = str(uuid4())
    main.node.training = Training(id=training_uuid,
                                  base_model=Model(id=model_id),
                                  organization='zauberzeug',
                                  project='pytest',
                                  project_folder="",
                                  images_folder="",
                                  training_folder=""
                                  )

    _, image_folder, training_path = test_helper.create_needed_folders(training_uuid)
    data = test_helper.get_data2()
    main.node.training.data = await main.download_data(main.node, data, image_folder, training_path)

    shutil.copy('darknet_tests/test_data/last_training.log', f'{training_path}/last_training.log')

    yolo_helper.create_backup_dir(training_path)
    open(f'{training_path}/backup//tiny_yolo_best_mAP_0.000000_iteration_1089_avgloss_-nan_.weights', 'a').close()

    assert main.node.training.last_published_iteration == None
    await main._check_state()
    assert main.node.training.last_published_iteration == 1089
    model_ids = get_model_ids_from__latest_training()
    assert(len(model_ids)) == 2

    new_model_id = [id for id in model_ids if id != model_id][0]
    assert os.path.exists(f'{training_path}/{new_model_id}.weights'), 'there is no weightfile for this model'


@pytest.mark.asyncio
async def test_get_files_for_model_on_save():
    model_id = test_helper.assert_upload_model()
    training_uuid = str(uuid4())

    main.node.training = Training(id=training_uuid,
                                  base_model=Model(id=model_id),
                                  organization='zauberzeug',
                                  project='pytest',
                                  project_folder="",
                                  images_folder="",
                                  training_folder=""
                                  )

    _, image_folder, training_folder = test_helper.create_needed_folders(training_uuid)
    data = test_helper.get_data2()
    training_data = await main.download_data(main.node, data, image_folder, training_folder)
    await main._prepare_training(main.node, training_folder, image_folder,  training_data, training_uuid)

    shutil.copy('darknet_tests/test_data/last_training.log', f'{training_folder}/last_training.log')

    yolo_helper.create_backup_dir(training_folder)
    open(f'{training_folder}/backup//tiny_yolo_best_mAP_0.000000_iteration_1089_avgloss_-nan_.weights', 'a').close()

    main.node.training.data = training_data

    await main._check_state()

    model_ids = get_model_ids_from__latest_training()
    assert(len(model_ids)) == 2

    new_model_id = [id for id in model_ids if id != model_id][0]

    save_model_handler = main.node.sio.handlers['/']['save']
    model_to_save = {'id': new_model_id}
    assert save_model_handler('zauberzeug', 'pytest', model_to_save) == True


def get_box_categories():
    content = test_helper.LiveServerSession().get('/api/zauberzeug/projects/pytest/data/data2').json()
    categories = content['box_categories']
    return categories


def get_model_ids_from__latest_training():
    content = test_helper.LiveServerSession().get('/api/zauberzeug/projects/pytest/trainings')
    content_json = content.json()
    datapoints = [datapoint['model_id'] for datapoint in content_json['charts'][0]['data']]
    return datapoints
