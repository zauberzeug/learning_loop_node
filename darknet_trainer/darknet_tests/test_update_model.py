import shutil
from mAP_parser import MAPParser
import pytest
import main
import darknet_tests.test_helper as test_helper
from uuid import uuid4


@pytest.fixture(autouse=True, scope='module')
def create_project():
    test_helper.LiveServerSession().delete(f"/api/zauberzeug/projects/pytest?keep_images=true")
    project_configuration = {'project_name': 'pytest', 'inbox': 0, 'annotate': 0, 'review': 0, 'complete': 2, 'image_style': 'beautiful',
                             'categories': 2, 'thumbs': False, 'tags': 0, 'trainings': 1, 'detections': 3, 'annotations': 0, 'skeleton': False}
    assert test_helper.LiveServerSession().post(f"/api/zauberzeug/projects/generator",
                                                json=project_configuration).status_code == 200
    yield
    test_helper.LiveServerSession().delete(f"/api/zauberzeug/projects/pytest?keep_images=true")


def test_parse_latest_confusion_matrix():
    training_uuid = "some_uuid"
    _, _, training_path = test_helper.create_needed_folders(training_uuid)
    shutil.copy('darknet_tests/test_data/last_training.log', f'{training_path}/last_training.log')

    new_model = main.parse_latest_confusion_matrix(training_uuid)
    assert new_model
    assert new_model['iteration'] == 1089
    confusion_matrix = new_model['confusion_matrix']
    assert len(confusion_matrix) == 2
    purple_matrix = confusion_matrix['purple']

    assert purple_matrix['ap'] == 42
    assert purple_matrix['tp'] == 1
    assert purple_matrix['fp'] == 2
    assert purple_matrix['fn'] == 3


@pytest.mark.asyncio
async def test_model_is_updated():
    def get_box_categories():
        content = test_helper.LiveServerSession().get('/api/zauberzeug/projects/pytest/data').json()
        categories = content['box_categories']
        return categories

    def get_model_ids_from__latest_training():
        content = test_helper.LiveServerSession().get('/api/zauberzeug/projects/pytest/trainings')
        content_json = content.json()
        datapoints = [datapoint['model_id'] for datapoint in content_json['charts'][0]['data']]
        return datapoints

    await main.node.connect()
    await main.node.sio.sleep(1.0)

    model_id = test_helper.assert_upload_model()

    model_ids = get_model_ids_from__latest_training()
    assert(len(model_ids)) == 1

    training_uuid = "some_uuid"
    _, _, training_path = test_helper.create_needed_folders(training_uuid)
    shutil.copy('darknet_tests/test_data/last_training.log', f'{training_path}/last_training.log')

    main.node.status.organization = 'zauberzeug'
    main.node.status.project = 'pytest'
    main.node.status.model = {'id': model_id,
                              'training_id': training_uuid}
    main.node.status.train_images = []
    main.node.status.test_images = []
    main.node.status.box_categories = get_box_categories()

    assert main.node.status.model.get('last_published_iteration') == None
    await main._check_state()
    assert main.node.status.model.get('last_published_iteration') == 1089
    model_ids = get_model_ids_from__latest_training()
    assert(len(model_ids)) == 2
