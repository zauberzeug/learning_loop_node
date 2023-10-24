from learning_loop_node.data_classes import (AnnotationData,
                                             AnnotationEventType, Category,
                                             CategoryType, Context, Point,
                                             UserInput)


class MockAsyncClient():  # pylint: disable=too-few-public-methods
    def __init__(self):
        self.history = []

    async def call(self, *args, **kwargs):
        self.history.append((args, kwargs))
        return True


def default_user_input() -> UserInput:
    context = Context(organization='zauberzeug', project='pytest_dst')
    category = Category(id='some_id', name='category_1', description='',
                        hotkey='', color='', type=CategoryType.Segmentation, point_size=None)
    annotation_data = AnnotationData(
        coordinate=Point(x=0, y=0),
        event_type=AnnotationEventType.LeftMouseDown,
        context=context,
        image_uuid='285a92db-bc64-240d-50c2-3212d3973566',
        category=category,
        is_shift_key_pressed=None
    )
    return UserInput(frontend_id='some_id', data=annotation_data)

# TODO: test missing
