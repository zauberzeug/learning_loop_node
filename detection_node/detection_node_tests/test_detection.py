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
        zip(classes, confidences, boxes), net, category_names, net_id)
    assert len(inferences) == 8
    assert inferences[0] == {'category_name': 'dirt',
                             'confidence': 85.5,
                             'height': 24,
                             'model_name': 'some_weightfile',
                             'width': 37,
                             'x': 1366,
                             'y': 1017}


def test_calculate_inferences_from_sent_images():
    data = {('file', open(image_path, 'rb'))}
    request = requests.post('http://detection_node/images', files=data)
    assert request.status_code == 200
    content = request.json()
    inferences = content['box_detections']

    assert len(inferences) == 8
    assert inferences[0] == {'category_name': 'dirt',
                             'confidence': 85.5,
                             'height': 24,
                             'model_name': 'some_weightfile',
                             'width': 37,
                             'x': 1366,
                             'y': 1017}


def test_save_detections_and_image():
    detections = [
        {"category_name": "dirt",
         "x": 1,
         "y": 1,
         "width": 1,
         "height": 1,
         "model_name": "some_weightfile",
         "confidence": 30.0}]

    image = cv2.imread(image_path)
    mac_address = '00000'
    filename = 'test'
    dir = '/test_data'

    helper.save_detections_and_image(dir, detections, image, filename, mac_address)
    saved_files = glob(f'{dir}/*', recursive=True)
    assert len(saved_files) == 2
    filename = saved_files[0].rsplit('.', 1)[0]
    with open(f'{filename}.json') as f:
        content = json.load(f)

    assert content['box_detections'] == detections
    assert content['tags'][0] == mac_address


@pytest.mark.reset_active_learners()
def test_save_image_and_detections_if_mac_was_sent():
    request = requests.put('http://detection_node/reset')
    assert request.status_code == 200

    data = {('file', open(image_path, 'rb'))}
    mac_adress = {"mac": '0:0:0:0'}
    request = requests.post('http://detection_node/images', files=data, data=mac_adress)
    assert request.status_code == 200
    content = request.json()
    inferences = content['box_detections']

    assert len(inferences) == 8
    assert inferences[0] == {'category_name': 'dirt',
                             'confidence': 85.5,
                             'height': 24,
                             'model_name': 'some_weightfile',
                             'width': 37,
                             'x': 1366,
                             'y': 1017}

    saved_files = glob('/data/*', recursive=True)
    assert len(saved_files) == 2


def test_extract_macs_and_filenames():
    files = ['/data/00000_2021-03-31_07:05:54.849.jpg',
             '/data/00000_2021-03-31_07:04:51.314.jpg',
             '/data/00000_2021-03-31_07:05:54.849.json',
             '/data/00000_2021-03-31_07:04:51.314.json',
             '/data/00001_2021-03-31_07:04:51.316.json']

    macs_and_filenames = helper.extract_macs_and_filenames(files)
    assert macs_and_filenames == {'00000': ['00000_2021-03-31_07:05:54.849', '00000_2021-03-31_07:04:51.314'],
                                  '00001': ['00001_2021-03-31_07:04:51.316']}


def test_get_filename():
    filpaths = ['/data/2462abd538f8_2021-01-17_08-33-49.800.jpg', '/data/2462abd538f8_2021-01-17_08-33-49.800.json']
    _filepaths = helper.get_file_paths(filpaths)
    assert _filepaths == {'/data/2462abd538f8_2021-01-17_08-33-49.800'}


def test_files_are_deleted_after_sending():
    with open('/data/test.json', 'w') as f:
        f.write('Json testfile')
        f.close()

    with open('/data/test.jpg', 'w') as f:
        f.write('Jpg testfile')
        f.close()

    saved_files = glob('/data/*', recursive=True)
    assert len(saved_files) == 2

    main._handle_detections()

    saved_files = glob('/data/*', recursive=True)
    assert len(saved_files) == 0
