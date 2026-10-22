import io

import numpy as np
import pytest
from PIL import Image

from ...helpers.misc import numpy_image_from_dict


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


def test_an_unknown_format_is_rejected():
    with pytest.raises(ValueError):
        numpy_image_from_dict({'format': 'png', 'bytes': b''})


def _jpeg_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format='JPEG', quality=95)
    return buffer.getvalue()
