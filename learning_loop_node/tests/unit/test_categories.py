import pytest

from ...data_classes import Category, ModelInformation
from ...detector.categories import category_by_index, category_by_name
from ...enums import CategoryType

BOX = Category(id='uuid-box', name='car', type=CategoryType.Box)
POINT = Category(id='uuid-point', name='weed', type=CategoryType.Point)


def model_information(*categories: Category) -> ModelInformation:
    return ModelInformation(id='model-uuid', host='localhost', organization='zauberzeug',
                            project='pytest', version='1.2', categories=list(categories or (BOX, POINT)))


def test_category_is_resolved_by_index():
    assert category_by_index(model_information(), 1) is POINT


@pytest.mark.parametrize('index', [-1, 2, 99])
def test_an_index_outside_the_model_categories_is_an_error(index: int):
    # a mismatch between model and metadata must not be silently skipped
    with pytest.raises(ValueError, match='out of range'):
        category_by_index(model_information(), index)


def test_category_is_resolved_by_name():
    assert category_by_name(model_information(), 'weed') is POINT


def test_an_unknown_category_name_lists_the_known_ones():
    with pytest.raises(ValueError, match='car, weed'):
        category_by_name(model_information(), 'tractor')
