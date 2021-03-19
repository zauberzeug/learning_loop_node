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


def test_get_network_input_image_size():
    width, height = inferences_helper._get_network_input_image_size(base_path)
    assert width == 800
    assert height == 800


def test_load_network():
    net = main.node.net
    assert len(net.getLayerNames()) == 94


def test_get_inferences():
    net = net = main.node.net
    image = inferences_helper._read_image(image_path)
    outs = inferences_helper.get_inferences(net, image, 800, 800)
    assert len(outs) == 3


def test_parse_inferences():
    net = main.node.net
    image = inferences_helper._read_image(image_path)
    outs = inferences_helper.get_inferences(net, image, 800, 800)
    net_id = inferences_helper._get_model_id(main.node.path)
    inferences = inferences_helper.parse_inferences(outs, net, image.shape[0], image.shape[1], net_id)
    assert len(inferences) == 5
    assert inferences[0] == {'class_id': 0,
                             'confidence': 0.643,
                             'height': 14,
                             'net': 'tiny_3l_23_2775',
                             'width': 10,
                             'x': 1110,
                             'y': 1149}

