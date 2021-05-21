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
from helper import data_dir
from ctypes import *

base_path = '/model'
image_path = f'{base_path}/2462abd538f8_2021-01-17_08-33-49.800.jpg'


# def test_export_weights():
#     return_code = detections_helper.export_weights(helper.find_cfg_file(base_path), helper.find_weight_file(base_path))
#     assert return_code == 0


def test_get_number_of_classes():
    classes = detections_helper.get_number_of_classes(base_path)
    assert classes == 9


def test_create_darknet_image():
    image = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    darknet_image = detections_helper.create_darknet_image(img_rgb)
    assert darknet_image.w == 1600
    assert darknet_image.h == 1200
    assert darknet_image.c == 3


def test_detect_image():
    image = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    darknet_image = detections_helper.create_darknet_image(img_rgb)
    model_id = detections_helper.get_model_id(base_path)
    detections = detections_helper.detect_image(main.node.net, darknet_image, model_id)
    assert len(detections) == 7
    assert detections[0] == ('dirt', 0.852836549282074, 1366.3819580078125,
                             1017.0836181640625, 36.920166015625, 24.57421875, 'some_weightfile')


def test_parse_detections():
    image = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    darknet_image = detections_helper.create_darknet_image(img_rgb)
    model_id = detections_helper.get_model_id(base_path)
    detections = detections_helper.detect_image(main.node.net, darknet_image, model_id)
    parsed_detections = detections_helper.parse_detections(detections)
    assert len(parsed_detections) == 7
    assert parsed_detections[0].__dict__ == {'category_name': 'dirt',
                                             'confidence': 85.3,
                                             'height': 24,
                                             'model_name': 'some_weightfile',
                                             'width': 36,
                                             'x': 1366,
                                             'y': 1017}


def test_get_model_id():
    model_id = detections_helper.get_model_id(base_path)
    assert model_id == 'some_weightfile'


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

    # Wait for async file saving
    for try_to_get_files in range(20):
        saved_files = helper.get_data_files()
        await asyncio.sleep(.2)
        if saved_files == 2:
            break

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


def test_files_are_deleted_after_sending():
    with open(f'{data_dir}/test.json', 'w') as f:
        f.write('Json testfile')
        f.close()

    with open(f'{data_dir}/test.jpg', 'w') as f:
        f.write('Jpg testfile')
        f.close()

    saved_files = helper.get_data_files()
    assert len(saved_files) == 2

    main._handle_detections()

    saved_files = helper.get_data_files()
    assert len(saved_files) == 0
