from dataclasses import asdict

from dacite import from_dict
from fastapi.encoders import jsonable_encoder

from ...data_classes import AnnotationData, AnnotationEventType, Category, Context, Point

# Used by all Nodes


def test_basemodel_functionality():
    obj = AnnotationData(
        coordinate=Point(x=0, y=0),
        event_type=AnnotationEventType.LeftMouseDown,
        context=Context(organization='zauberzeug', project='pytest'),
        image_uuid='285a92db-bc64-240d-50c2-3212d3973566',
        category=Category(id='some_id', name='category_1', description=''),
        key_down='test',
    )

    assert from_dict(data_class=AnnotationData, data=jsonable_encoder(asdict(obj))) == obj
