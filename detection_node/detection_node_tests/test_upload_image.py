
import requests
import json
from glob import glob
import helper
from helper import data_dir


base_path = '/model'
image_name = '2462abd538f8_2021-01-17_08-33-49.800.jpg'
json_name = '2462abd538f8_2021-01-17_08-33-49.800.json'
image_path = f'{base_path}/{image_name}'
json_path = f'/tmp/{json_name}'


def test_upload_image():
    assert len(helper.get_data_files()) == 0
    json_content = {'some_key': 'some_value'}

    with open(json_path, 'w') as f:
        json.dump(json_content, f)

    data = [('files', open(image_path, 'rb')),
            ('files', open(json_path, 'r')), ]

    response = requests.post('http://detection_node/upload', files=data)
    assert response.status_code == 200

    data_files = helper.get_data_files()
    assert len(data_files) == 2

    # we do not check the .jpg file.
    assert f'{data_dir}/{image_name}' in data_files
    assert f'{data_dir}/{json_name}' in data_files

    with open(f'{data_dir}/{json_name}', 'r') as f:
        uploaded_json_content = json.load(f)

    assert uploaded_json_content == json_content
