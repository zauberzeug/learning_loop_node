import io

import numpy as np
import pytest
from PIL import Image

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


def _jpeg_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format='JPEG', quality=95)
    return buffer.getvalue()
