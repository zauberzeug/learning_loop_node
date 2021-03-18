import pytest
import main
import inferences_helper
from icecream import ic
import numpy as np


def test_get_files():
    files = main._get_model_files()
    assert files[0] == '/data/names.txt'
    assert files[1] == '/data/training_final.weights'
    assert files[2] == '/data/training.cfg'


def test_get_names():
    names = inferences_helper.get_names_of_classes('/data/names.txt')
    assert names == ['dirt', 'obstacle', 'animal', 'person', 'robot', 'marker_vorne',
                     'marker_mitte', 'marker_hinten_links', 'marker_hinten_rechts']


def test_load_network():
    net = inferences_helper.load_network('/data/training.cfg', '/data/training_final.weights')
    assert len(net.getLayerNames()) == 94


def test_get_inferences():
    net = inferences_helper.load_network('/data/training.cfg', '/data/training_final.weights')
    image_path = '/data/2462abd538f8_2021-01-17_08-33-49.800.jpg'
    image = inferences_helper._read_image(image_path)
    outs = inferences_helper.get_inferences(net, image)
    assert len(outs) == 3


def test_parse_outs():
    net = inferences_helper.load_network('/data/training.cfg', '/data/training_final.weights')
    image_path = '/data/2462abd538f8_2021-01-17_08-33-49.800.jpg'
    image = inferences_helper._read_image(image_path)
    outs = inferences_helper.get_inferences(net, image)
    indices, class_ids, boxes, confidences = inferences_helper.parse_inferences(outs, net, 608, 608)
    assert indices == [1, 0]
    assert class_ids == [0, 0]
    assert len(boxes) == 2
    assert boxes == [[397, 142, 15, 6], [604, 458, 4, 6]]
    assert confidences == [0.609, 0.798]


def test_create_json_from_outs():
    net = inferences_helper.load_network('/data/training.cfg', '/data/training_final.weights')
    image_path = '/data/2462abd538f8_2021-01-17_08-33-49.800.jpg'
    image = inferences_helper._read_image(image_path)
    outs = inferences_helper.get_inferences(net, image)
    indices, class_ids, boxes, confidences = inferences_helper.parse_inferences(outs, net, 608, 608)

    json = inferences_helper.convert_to_json(indices, class_ids, boxes, confidences)
    ic(json)
    assert json == '{"0": {"box": [397, 142, 15, 6], "class_id": 0, "confidence": 0.609}, "1": {"box": [604, 458, 4, 6], "class_id": 0, "confidence": 0.798}}'
