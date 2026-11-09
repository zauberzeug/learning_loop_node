import io

import numpy as np
import pytest
from PIL import Image

from ...data_classes import CaptureMetadata
from ...detector.detector_node import _capture_metadata
from ...helpers.misc import image_from_dict, numpy_image_from_dict


def test_a_raw_dict_is_reshaped_without_a_format_key():
    array = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    image = numpy_image_from_dict({'bytes': array.tobytes(order='C'), 'dtype': 'uint8', 'shape': (2, 3, 3)})
    assert np.array_equal(image, array)


def test_a_jpeg_dict_is_decoded():
    array = np.full((8, 16, 3), (73, 109, 137), dtype=np.uint8)
    image = numpy_image_from_dict({'format': 'jpeg', 'bytes': _jpeg_bytes(array)})
    assert image.shape == (8, 16, 3)
    assert image.dtype == np.uint8
    assert np.abs(image.astype(int) - array.astype(int)).max() <= 3  # JPEG is lossy, but a flat color barely changes


def test_image_from_dict_keeps_jpeg_bytes_and_parses_raw_arrays():
    array = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    jpeg_bytes = _jpeg_bytes(array)
    assert image_from_dict({'format': 'jpeg', 'bytes': jpeg_bytes}) == jpeg_bytes
    assert np.array_equal(image_from_dict({'bytes': array.tobytes(order='C'), 'dtype': 'uint8', 'shape': (2, 3, 3)}), array)


def test_a_jpeg_is_rotated_as_its_orientation_tag_says():
    array = np.zeros((8, 16, 3), dtype=np.uint8)
    exif = Image.Exif()
    exif[0x0112] = 6  # shown rotated by 90 degrees, so the 16 wide image appears 16 tall
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format='JPEG', exif=exif.tobytes())
    image = numpy_image_from_dict({'format': 'jpeg', 'bytes': buffer.getvalue()})
    assert image.shape == (16, 8, 3)


def test_an_unknown_format_is_rejected():
    with pytest.raises(ValueError):
        numpy_image_from_dict({'format': 'png', 'bytes': b''})


_BATCH = {'camera_id': 'batch_cam', 'tags': ['robot'], 'source': 'robot', 'creation_date': '2024-01-01T00:00:00'}


def test_an_image_replaces_the_values_of_the_batch_and_adds_its_tags():
    image_data = {'camera_id': 'cam_a', 'tags': ['cam_a'], 'source': 'robot/cam_a',
                  'creation_date': '2024-02-02T00:00:00'}
    assert _capture_metadata(_BATCH, image_data) == CaptureMetadata(
        camera_id='cam_a', tags=['robot', 'cam_a'], source='robot/cam_a', creation_date='2024-02-02T00:00:00')


@pytest.mark.parametrize('image_data', [{}, {'camera_id': None, 'tags': None, 'source': None, 'creation_date': None},
                                        {'tags': []}])
def test_what_an_image_does_not_set_comes_from_the_batch(image_data: dict):
    assert _capture_metadata(_BATCH, image_data) == CaptureMetadata(
        camera_id='batch_cam', tags=['robot'], source='robot', creation_date='2024-01-01T00:00:00')


def test_what_neither_sets_stays_unset():
    assert _capture_metadata({}, {}) == CaptureMetadata()


@pytest.mark.parametrize('tags', ['cam_a', ('cam_a',), 7])
def test_an_image_with_tags_that_are_no_list_is_rejected(tags: object):
    with pytest.raises(ValueError):
        _capture_metadata({}, {'tags': tags})


def _jpeg_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format='JPEG', quality=95)
    return buffer.getvalue()
