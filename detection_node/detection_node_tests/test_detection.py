from pydantic.types import Json
import main
import detections_helper
import requests
from icecream import ic
import cv2
import helper
from glob import glob
import os
import json

base_path = '/model'
image_path = f'{base_path}/2462abd538f8_2021-01-17_08-33-49.800.jpg'


def test_get_model_id():
    model_id = detections_helper._get_model_id(base_path)
    assert model_id == 'some_weightfile'


def test_get_names():
    names = detections_helper.get_category_names(base_path)
    assert names == ['dirt', 'obstacle', 'animal', 'person', 'robot', 'marker_vorne',
                     'marker_mitte', 'marker_hinten_links', 'marker_hinten_rechts']


def test_get_network_input_image_size():
    width, height = detections_helper.get_network_input_image_size(base_path)
    assert width == 800
    assert height == 800


def test_load_network():
    net = main.node.net
    assert len(net.getLayerNames()) == 94


def test_get_inferences():
    net = net = main.node.net
    image = detections_helper._read_image(image_path)
    classes, confidences, boxes = detections_helper.get_inferences(net, image, 800, 800, swapRB=True)
    assert len(classes) == 8


def test_parse_inferences():
    net = main.node.net
    category_names = detections_helper.get_category_names(main.node.path)
    image = detections_helper._read_image(image_path)
    classes, confidences, boxes = detections_helper.get_inferences(net, image, 800, 800, swapRB=True)
    net_id = detections_helper._get_model_id(main.node.path)
    inferences = detections_helper.parse_detections(
        zip(classes, confidences, boxes), net, category_names, image.shape[1], image.shape[0], net_id)
    assert len(inferences) == 8
    assert inferences[0] == {'category': 'dirt',
                             'confidence': 0.855,
                             'height': 24,
                             'net': 'some_weightfile',
                             'width': 37,
                             'x': 1366,
                             'y': 1017}


def test_calculate_inferences_from_sent_images():
    data = {('file', open(image_path, 'rb'))}
    request = requests.post('http://detection_node/images', files=data)
    assert request.status_code == 200
    content = request.json()
    inferences = content['box_detections']
    ic(inferences)
    assert len(inferences) == 8
    assert inferences[0] == {'category': 'dirt',
                             'confidence': 0.855,
                             'height': 24,
                             'net': 'some_weightfile',
                             'width': 37,
                             'x': 1366,
                             'y': 1017}


def test_save_detections_and_image():
    detections = [
        {"category": "dirt",
         "x": 1366,
         "y": 1017,
         "width": 37,
         "height": 24,
         "net": "some_weightfile",
         "confidence": 0.852},
        {"category": "dirt",
         "x": 1479,
         "y": 862,
         "width": 14,
         "height": 11,
         "net": "some_weightfile",
         "confidence": 0.676}]

    image = cv2.imread(image_path)
    mac_address = '0:0:0:0:0'

    helper.save_detections_and_image(detections, image, mac_address)
    saved_files = glob('/data/*', recursive=True)
    assert len(saved_files) == 2
    filename = saved_files[0].rsplit('.', 1)[0]
    with open(f'{filename}.json') as f:
        content = json.load(f)

    ic(content)
    ic(detections)

    assert content == {"box_detections": detections}

    for file in saved_files:
        os.remove(file)


def test_save_image_and_detections_if_mac_was_sent():
    data = {('file', open(image_path, 'rb'))}
    mac_adress = {"mac": '0:0:0:0'}
    request = requests.post('http://detection_node/images', files=data, data=mac_adress)
    assert request.status_code == 200
    content = request.json()
    inferences = content['box_detections']

    assert len(inferences) == 8
    assert inferences[0] == {'category': 'dirt',
                             'confidence': 0.855,
                             'height': 24,
                             'net': 'some_weightfile',
                             'width': 37,
                             'x': 1366,
                             'y': 1017}

    saved_files = glob('/data/*', recursive=True)
    assert len(saved_files) == 2

    for file in saved_files:
        os.remove(file)
