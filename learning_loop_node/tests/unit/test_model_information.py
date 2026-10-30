import json
from types import SimpleNamespace

from ...data_classes import Category, ModelInformation
from ...data_classes.general import DEFAULT_CONF_THRESHOLD
from ...trainer.trainer_logic_generic import TrainerLogicGeneric

MODEL_JSON = {'id': 'model-uuid', 'host': 'localhost', 'organization': 'zauberzeug', 'project': 'demo',
              'version': '1.2', 'categories': [{'id': 'c1', 'name': 'weed'}], 'resolution': 640}


def test_conf_threshold_is_read_from_model_json(tmp_path):
    (tmp_path / 'model.json').write_text(json.dumps({**MODEL_JSON, 'conf_threshold': 0.37}))

    model_information = ModelInformation.load_from_disk(str(tmp_path))

    assert model_information is not None
    assert model_information.conf_threshold == 0.37


def test_a_model_without_a_conf_threshold_gets_the_default(tmp_path):
    (tmp_path / 'model.json').write_text(json.dumps(MODEL_JSON))

    model_information = ModelInformation.load_from_disk(str(tmp_path))

    assert model_information is not None
    assert model_information.conf_threshold == DEFAULT_CONF_THRESHOLD


def test_model_json_carries_the_trainers_model_metadata():
    trainer = SimpleNamespace(_training=SimpleNamespace(categories=[Category(id='c1', name='weed')]),
                              _get_model_metadata=lambda: {'conf_threshold': 0.37})

    with open(TrainerLogicGeneric._dump_model_json(trainer)) as f:  # type: ignore[arg-type]
        content = json.load(f)

    assert content['conf_threshold'] == 0.37
    assert [c['id'] for c in content['categories']] == ['c1']
