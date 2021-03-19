import pytest
import main
import inferences_helper
from icecream import ic
import numpy as np
import requests

base_path = '/data/yolo4_tiny_3lspp_12_76844'
image_path = f'{base_path}/2462abd538f8_2021-01-17_08-33-49.800.jpg'


def test_get_files():
    files = main._get_model_files()
    assert files[0].split('/')[-1] == 'names.txt'
    assert files[1].split('/')[-1] == 'project.json'
    assert files[2].split('/')[-1] == 'training_final.weights'
    assert files[3].split('/')[-1] == 'training.cfg'


def test_get_model_id():
    model_id = inferences_helper._get_model_id(base_path)
    assert model_id == 'tiny_3l_23_2775'


def test_get_names():
    names = inferences_helper.get_names_of_classes(f'{base_path}/names.txt')
    assert names == ['dirt', 'obstacle', 'animal', 'person', 'robot', 'marker_vorne',
                     'marker_mitte', 'marker_hinten_links', 'marker_hinten_rechts']


def test_load_network():
    net = main.node.net
    assert len(net.getLayerNames()) == 94


def test_get_inferences():
    net = net = main.node.net
    image = inferences_helper._read_image(image_path)
    outs = inferences_helper.get_inferences(net, image)
    assert len(outs) == 3


def test_parse_outs():
    net = main.node.net
    image = inferences_helper._read_image(image_path)
    outs = inferences_helper.get_inferences(net, image)
    indices, class_ids, boxes, confidences = inferences_helper.parse_inferences(outs, net, 608, 608)
    assert indices == [1, 0]
    assert class_ids == [0, 0]
    assert len(boxes) == 2
    assert boxes == [[397, 142, 15, 6], [604, 458, 4, 6]]
    assert confidences == [0.609, 0.798]


def test_create_json_from_outs():
    net = main.node.net
    image = inferences_helper._read_image(image_path)
    outs = inferences_helper.get_inferences(net, image)
    indices, class_ids, boxes, confidences = inferences_helper.parse_inferences(outs, net, 608, 608)

    net_id = inferences_helper._get_model_id(main.node.path)
    json = inferences_helper.convert_to_json(indices, class_ids, boxes, net_id, confidences)
    ic(json)
    assert json == '{"0": {"class_id": 0, "x": 397, "y": 142, "width": 15, "height": 6, "net": "tiny_3l_23_2775", "confidence": 0.609}, "1": {"class_id": 0, "x": 604, "y": 458, "width": 4, "height": 6, "net": "tiny_3l_23_2775", "confidence": 0.798}}'


def test_calculate_inferences_from_sent_images():
    file = image_path
    data = [('file', open(file, 'rb')), ('file', open(file, 'rb'))]
    request = requests.post('http://detection_node/images/', files=data)
    assert request.status_code == 200
    content = request.json()
    calculated_inferences = '{"0": {"class_id": 0, "x": 587, "y": 450, "width": 6, "height": 7, "net": "tiny_3l_23_2775", "confidence": 0.635}, "1": {"class_id": 0, "x": 603, "y": 457, "width": 4, "height": 7, "net": "tiny_3l_23_2775", "confidence": 0.804}}'
    assert content == [calculated_inferences, calculated_inferences]
