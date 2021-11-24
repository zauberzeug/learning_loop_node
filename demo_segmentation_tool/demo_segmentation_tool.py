from typing import List, Optional
from learning_loop_node.annotation_node.annotation_tool import AnnotationTool
from learning_loop_node.annotation_node.data_classes import EventType, Point, SegmentationAnnotation, Shape, ToolOutput, UserInput
from uuid import uuid4
from pydantic import BaseModel
import numpy as np
import cv2
from icecream import ic


class History(BaseModel):
    inside_clicks: List[Point] = []
    outside_clicks: List[Point] = []
    annotation: Optional[SegmentationAnnotation]


class DemoSegmentationTool(AnnotationTool):
    history: Optional[History]

    async def handle_user_input(self, user_input: UserInput) -> ToolOutput:

        # points = [Point(x=0, y=0), Point(x=100, y=100), Point(x=0, y=100)]

        if user_input.data.event_type == EventType.MouseDown:
            if not self.history:
                self.history = History()
                # currently we only support outside clicks.
            self.history.outside_clicks.append(user_input.data.coordinate)

            points = autofit(f'/data/{user_input.data.context.organization}/{user_input.data.context.project}/images/{user_input.data.image_uuid}.jpg',
                             self.history)

            if not self.history.annotation or self.history.annotation.image_id != user_input.data.image_uuid:
                self.history.annotation = SegmentationAnnotation(id=str(uuid4()), shape=Shape(points=[]),
                                                                 image_id=user_input.data.image_uuid, category_id=user_input.data.category.id)
            self.history.annotation.shape.points = points
            ic(self.history.annotation)
            return ToolOutput(svg="", annotation=self.history.annotation)

        return ToolOutput(svg="", annotation=None)


def autofit(image_path, history: History) -> List[Point]:
    img = cv2.imread(image_path)

    # region-of-interest
    H, W = img.shape[:2]
    ic(f'Image width, height : {W}, {H} ')
    ic(f'ClickPoint : {Point.__dict__}')

    mask = np.ones(img.shape[:2], np.uint8) * cv2.GC_BGD

    array_points = []
    polygon = np.array([array_points], dtype=np.int32)

    if history.annotation:
        array_points = [[p.x, p.y] for p in history.annotation.shape.points]

    inside_points = []
    for p in history.inside_clicks:
        inside_points.append([p.x, p.y])

    outside_points = []
    for p in history.outside_clicks:
        outside_points.append([p.x, p.y])

    bbox_points = array_points + outside_points

    bbox = bounding_box(bbox_points)
    box_size = 100
    bbox[0] = max(bbox[0] - box_size, 0)
    bbox[1] = max(bbox[1] - box_size, 0)
    bbox[2] = min(bbox[2] + box_size, W)
    bbox[3] = min(bbox[3] + box_size, H)

    mask[bbox[1]:bbox[1]+bbox[3],
         bbox[0]: bbox[0]+bbox[2]] = cv2.GC_PR_FGD

    if array_points:
        polygon = np.array([array_points], dtype=np.int32)
        # cv2.fillPoly(mask, polygon, cv2.GC_PR_FGD)
        cv2.fillPoly(mask, polygon, cv2.GC_FGD)
    for p in outside_points:
        mask[p[1], p[0]] = cv2.GC_FGD

    for p in inside_points:
        mask[p[1], p[0]] = cv2.GC_BGD

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(img, mask, bbox, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_MASK)

    mask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1).astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    greatest_contour = max(contours, key=cv2.contourArea)
    points = [Point(x=something[0][0], y=something[0][1]) for something in greatest_contour]

    return points


def bounding_box(points):
    x_coordinates, y_coordinates = zip(*points)
    return [min(x_coordinates), min(y_coordinates), max(x_coordinates), max(y_coordinates)]
