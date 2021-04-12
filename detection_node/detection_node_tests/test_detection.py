import asyncio
from active_learner.learner import Learner
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
import pytest
import time

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
    net = main.node.model
    image = detections_helper._read_image(image_path)
    classes, confidences, boxes = detections_helper.get_inferences(net, image)
    assert len(classes) == 8


def test_parse_inferences():
    model = main.node.model
    category_names = detections_helper.get_category_names(main.node.path)
    image = detections_helper._read_image(image_path)
    classes, confidences, boxes = detections_helper.get_inferences(model, image)
    net_id = detections_helper._get_model_id(main.node.path)
    net = main.node.net
    inferences = detections_helper.parse_detections(
        zip(classes, confidences, boxes), net, category_names, net_id)
    assert len(inferences) == 8
    assert inferences[0].__dict__ == {'category_name': 'dirt',
                                      'confidence': 85.5,
                                      'height': 24,
                                      'model_name': 'some_weightfile',
                                      'width': 37,
                                      'x': 1366,
                                      'y': 1017}


@pytest.mark.asyncio()
async def test_save_image_and_detections_if_mac_was_sent():
    request = requests.put('http://detection_node/reset')
    assert request.status_code == 200

    data = {('file', open(image_path, 'rb'))}
    headers = {'mac': '0:0:0:0', 'tags':  'some_tag'}
    request = requests.post('http://detection_node/detect', files=data, headers=headers)
    assert request.status_code == 200
    content = request.json()
    inferences = content['box_detections']

    expected_detection = {'category_name': 'dirt',
                          'confidence': 85.5,
                          'height': 24,
                          'model_name': 'some_weightfile',
                          'width': 37,
                          'x': 1366,
                          'y': 1017}
    assert len(inferences) == 8
    assert inferences[0] == expected_detection

    for try_to_get_files in range(20):
        saved_files = helper.get_data_files()
        time.sleep(0.2)

    assert len(saved_files) == 2

    json_filename = [file for file in saved_files if file.endswith('.json')][0]
    with open(json_filename, 'r') as f:
        json_content = json.load(f)

    box_detections = json_content['box_detections']
    assert len(box_detections) == 8
    assert box_detections[0] == expected_detection

    tags = json_content['tags']
    assert len(tags) == 3
    assert tags == ['0:0:0:0', 'some_tag', 'lowConfidence']


def test_extract_macs_and_filenames():
    files = ['/data/00000_2021-03-31_07:05:54.849.jpg',
             '/data/00000_2021-03-31_07:04:51.314.jpg',
             '/data/00000_2021-03-31_07:05:54.849.json',
             '/data/00000_2021-03-31_07:04:51.314.json',
             '/data/00001_2021-03-31_07:04:51.316.json']

    macs_and_filenames = helper.extract_macs_and_filenames(files)
    assert macs_and_filenames == {'00000': ['00000_2021-03-31_07:05:54.849', '00000_2021-03-31_07:04:51.314'],
                                  '00001': ['00001_2021-03-31_07:04:51.316']}


def test_files_are_deleted_after_sending():
    with open('/data/test.json', 'w') as f:
        f.write('Json testfile')
        f.close()

    with open('/data/test.jpg', 'w') as f:
        f.write('Jpg testfile')
        f.close()

    saved_files = helper.get_data_files()
    assert len(saved_files) == 2

    main._handle_detections()

    saved_files = helper.get_data_files()
    assert len(saved_files) == 0
