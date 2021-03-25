import main
import inferences_helper
import requests
from icecream import ic

base_path = '/model'
image_path = f'{base_path}/2462abd538f8_2021-01-17_08-33-49.800.jpg'


def test_get_model_id():
    model_id = inferences_helper._get_model_id(base_path)
    assert model_id == 'some_weightfile'


def test_get_names():
    names = inferences_helper.get_category_names(base_path)
    assert names == ['dirt', 'obstacle', 'animal', 'person', 'robot', 'marker_vorne',
                     'marker_mitte', 'marker_hinten_links', 'marker_hinten_rechts']


def test_get_network_input_image_size():
    width, height = inferences_helper.get_network_input_image_size(base_path)
    assert width == 800
    assert height == 800


def test_load_network():
    net = main.node.net
    assert len(net.getLayerNames()) == 94


def test_get_inferences():
    net = net = main.node.net
    image = inferences_helper._read_image(image_path)
    classes, confidences, boxes = inferences_helper.get_inferences(net, image, 800, 800)
    assert len(classes) == 8


def test_parse_inferences():
    net = main.node.net
    category_names = inferences_helper.get_category_names(main.node.path)
    image = inferences_helper._read_image(image_path)
    classes, confidences, boxes = inferences_helper.get_inferences(net, image, 800, 800)
    net_id = inferences_helper._get_model_id(main.node.path)
    inferences = inferences_helper.parse_inferences(
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
    request = requests.post('http://detection_node/detect', files=data)
    assert request.status_code == 200
    content = request.json()
    inferences = content['box_detections']
    ic(inferences)
    assert len(inferences) == 4
    assert inferences[0] == {'category': 'dirt',
                             'confidence': 0.676,
                             'height': 11,
                             'net': 'some_weightfile',
                             'width': 14,
                             'x': 1479,
                             'y': 861}
