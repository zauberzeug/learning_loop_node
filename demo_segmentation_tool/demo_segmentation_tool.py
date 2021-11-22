from annotation_node import AnnotationTool, Point, SegmentationAnnotation, Shape, ToolOutput, UserInput
from uuid import uuid4


class DemoSegmentationTool(AnnotationTool):

    async def handle_user_input(self, user_input: UserInput) -> ToolOutput:

        points = [Point(x=0, y=0), Point(x=100, y=100), Point(x=0, y=100)]

        annotation = SegmentationAnnotation(id=str(uuid4()), shape=Shape(points=points),
                                            image_id=user_input.data.image_uuid, category_id=user_input.data.category.id)
        out = ToolOutput(svg="", annotation=annotation)

        return out
