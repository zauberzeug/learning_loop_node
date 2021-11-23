from typing import List
from learning_loop_node.annotation_node.annotation_tool import AnnotationTool
from learning_loop_node.annotation_node.data_classes import EventType, Point, SegmentationAnnotation, Shape, ToolOutput, UserInput
from uuid import uuid4
from pydantic import BaseModel
import numpy as np
import cv2
from icecream import ic


class DemoSegmentationTool(AnnotationTool):

    async def handle_user_input(self, user_input: UserInput) -> ToolOutput:

        points = [Point(x=0, y=0), Point(x=100, y=100), Point(x=0, y=100)]

        length = 100
        box = Box(x=user_input.data.coordinate.x - length/2,
                  y=user_input.data.coordinate.y - length/2, w=length, h=length)
        if user_input.data.event_type == EventType.MouseDown:
            points = autofit(f'/data/{user_input.data.context.organization}/{user_input.data.context.project}/images/{user_input.data.image_uuid}.jpg',
                             box, user_input.data.coordinate)
            ic(points)
            annotation = SegmentationAnnotation(id=str(uuid4()), shape=Shape(points=points),
                                                image_id=user_input.data.image_uuid, category_id=user_input.data.category.id)

            return ToolOutput(svg="", annotation=annotation)

        return ToolOutput(svg="", annotation=None)


class Box(BaseModel):
    x: int
    y: int
    w: int
    h: int


def autofit(image_path, box: Box, point: Point) -> List[Point]:
    img = cv2.imread(image_path)
    x_, y_, w_, h_ = int(box.x), int(box.y), box.w, box.h

    # region-of-interest
    padding_factor = 4
    H, W = img.shape[:2]
    # roi_l = max(x_ - padding_factor * w_, 0)
    # roi_t = max(y_ - padding_factor * h_, 0)
    # roi_r = min(x_ + (padding_factor + 1) * w_, W)
    # roi_b = min(y_ + (padding_factor + 1) * h_, H)
    # crop = img[roi_t:roi_b, roi_l:roi_r]
    crop = img
    # crop_rect = x_ - roi_l, y_ - roi_t, w_, h_
    crop_rect = x_, y_, w_, h_

    # grabcut
    mask = np.ones(crop.shape[:2], np.uint8) * cv2.GC_BGD
    mask[crop_rect[1]:crop_rect[1]+crop_rect[3],
         crop_rect[0]:crop_rect[0]+crop_rect[2]] = cv2.GC_PR_FGD
    mask[point.y, point.x] = cv2.GC_FGD
    # mask[crop_rect[1]+crop_rect[3]//4:crop_rect[1]+3*crop_rect[3]//4,
    #      crop_rect[0]+crop_rect[2]//4:crop_rect[0]+3*crop_rect[2]//4] = cv2.GC_FGD
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(crop, mask, crop_rect, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_MASK)

    mask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1).astype(np.uint8)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    greatest_contour = max(contours, key=cv2.contourArea)
    points = [Point(x=something[0][0], y=something[0][1]) for something in greatest_contour]

    return points
