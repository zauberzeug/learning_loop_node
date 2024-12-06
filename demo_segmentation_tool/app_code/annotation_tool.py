import logging
import sys
# pylint: disable=no-name-in-module
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional
from uuid import uuid4

import cv2
import numpy as np

from learning_loop_node.annotation.annotator_logic import AnnotatorLogic
from learning_loop_node.data_classes import Point, SegmentationAnnotation, Shape, ToolOutput, UserInput
from learning_loop_node.enums import AnnotationEventType

KWONLY_SLOTS = {'kw_only': True, 'slots': True} if sys.version_info >= (3, 10) else {}


@dataclass(**KWONLY_SLOTS)
class Box():
    x: int
    y: int
    w: int
    h: int

    def to_svg_rect(self) -> str:
        return f'<rect x="{self.x}" y="{self.y}" width="{abs(self.w)}" height="{abs(self.h)}" stroke="blue" fill="transparent">'


class AnnotationState(str, Enum):
    NONE = "NONE"
    CREATING = "CREATING"
    IDLE = "IDLE"
    EDITING = "EDITING"


@dataclass(**KWONLY_SLOTS)
class History():
    bbox: Optional[Box] = None
    bg_pixel: List[Point] = field(default_factory=list)
    fg_pixel: List[Point] = field(default_factory=list)
    path_pixels: List[Point] = field(default_factory=list)
    annotation: Optional[SegmentationAnnotation] = None
    state: AnnotationState = AnnotationState.NONE

    def to_svg_path(self, shift_pressed: bool) -> str:
        def create_path(pixel: List[Point]) -> str:
            path = f'M {pixel[0].x} {pixel[0].y}'
            for point in pixel[1:]:
                path += f' L {point.x} {point.y}'
            return path

        svg_path = ''
        if self.path_pixels:
            if shift_pressed:
                path = create_path(self.path_pixels)
                svg_path = f'<path d="{path}" stroke="white" stroke-width="3" fill="none">'
            else:
                path = create_path(self.path_pixels)
                svg_path = f'<path d="{path}" stroke="red" stroke-width="3" fill="none">'
        return svg_path


class SegmentationTool(AnnotatorLogic):
    # TODO: fix signature
    async def handle_user_input(self, user_input: UserInput, history: History) -> ToolOutput:  # type: ignore
        coordinate = user_input.data.coordinate
        output = ToolOutput(svg="", annotation=None)

        if history.state == AnnotationState.NONE and user_input.data.event_type == AnnotationEventType.LeftMouseDown:
            # start creating bbox
            history.state = AnnotationState.CREATING
            history.bbox = Box(x=coordinate.x, y=coordinate.y, w=0, h=0)
            output.svg = history.bbox.to_svg_rect()
            return output

        elif history.state == AnnotationState.CREATING and user_input.data.event_type == AnnotationEventType.MouseMove:
            # update bbox
            assert history.bbox is not None
            history.bbox.w = coordinate.x - history.bbox.x
            history.bbox.h = coordinate.y - history.bbox.y
            output.svg = history.bbox.to_svg_rect()
            return output

        elif history.state == AnnotationState.CREATING and user_input.data.event_type == AnnotationEventType.LeftMouseUp:
            # end update bbox
            history.state = AnnotationState.IDLE
            assert history.bbox is not None
            history.bbox.w = coordinate.x - history.bbox.x
            history.bbox.h = coordinate.y - history.bbox.y

            points = autofit(
                f'/data/{user_input.data.context.organization}/{user_input.data.context.project}/images/{user_input.data.image_uuid}.jpg',
                history)

            history.annotation = SegmentationAnnotation(id=str(uuid4()), shape=Shape(
                points=[]), image_id=user_input.data.image_uuid, category_id=user_input.data.category.id)
            history.annotation.shape.points = points
            assert isinstance(history.bbox, Box)
            output.svg = history.bbox.to_svg_rect()
            output.annotation = history.annotation
            return output

        elif history.state == AnnotationState.IDLE and user_input.data.event_type == AnnotationEventType.LeftMouseDown:
            # start gathering bg vg points
            history.state = AnnotationState.EDITING
            if user_input.data.is_shift_key_pressed:
                history.fg_pixel.append(user_input.data.coordinate)
            else:
                history.bg_pixel.append(user_input.data.coordinate)

        elif history.state == AnnotationState.EDITING and user_input.data.event_type == AnnotationEventType.MouseMove:
            # gathering bg vg points
            if user_input.data.is_shift_key_pressed:
                history.fg_pixel.append(user_input.data.coordinate)
            else:
                history.bg_pixel.append(user_input.data.coordinate)

            history.path_pixels.append(user_input.data.coordinate)

            output.svg = history.to_svg_path(user_input.data.is_shift_key_pressed or False)
            return output

        elif history.state == AnnotationState.EDITING and user_input.data.event_type == AnnotationEventType.LeftMouseUp:
            # gathering complete
            history.state = AnnotationState.IDLE
            history.path_pixels = []
            if user_input.data.is_shift_key_pressed:
                history.fg_pixel.append(user_input.data.coordinate)
            else:
                history.bg_pixel.append(user_input.data.coordinate)

            # crabcut
            points = autofit(
                f'/data/{user_input.data.context.organization}/{user_input.data.context.project}/images/{user_input.data.image_uuid}.jpg',
                history)
            if not history.annotation:
                return output
            history.annotation.shape.points = points
            output.annotation = history.annotation
            return output

        elif (history.state in [AnnotationState.NONE, AnnotationState.IDLE]) and user_input.data.event_type == AnnotationEventType.MouseMove:
            return output

        elif (history.state in [AnnotationState.NONE, AnnotationState.IDLE]) and user_input.data.key_down == 'Enter':
            history.state = AnnotationState.NONE
        else:
            logging.error(
                f"Invalid state transition: Current state: '{history.state}', user input : {asdict(user_input)} ")

        return ToolOutput(svg="", annotation=None)

    def create_empty_history(self):
        return History()

    def logout_user(self, sid):
        logging.info(sid)
        return True

# pylint: disable=no-member


def autofit(image_path, history: History) -> List[Point]:
    logging.debug('inside grab cut')
    img = cv2.imread(image_path)
    assert history.bbox is not None
    x_, y_, w_, h_ = int(history.bbox.x), int(history.bbox.y), int(history.bbox.w), int(history.bbox.h)

    # define region-of-interest
    padding_factor = 4
    H, W = img.shape[:2]
    roi_l = max(x_ - padding_factor * w_, 0)
    roi_t = max(y_ - padding_factor * h_, 0)
    roi_r = min(x_ + (padding_factor + 1) * w_, W)
    roi_b = min(y_ + (padding_factor + 1) * h_, H)
    crop = img[roi_t:roi_b, roi_l:roi_r]
    crop_rect = x_ - roi_l, y_ - roi_t, w_, h_

    # grabcut
    mask = np.ones(crop.shape[:2], np.uint8) * cv2.GC_BGD
    mask[crop_rect[1]:crop_rect[1]+crop_rect[3],
         crop_rect[0]:crop_rect[0]+crop_rect[2]] = cv2.GC_PR_FGD

    for p in history.fg_pixel:
        mask[p.y - roi_t, p.x - roi_l] = cv2.GC_FGD

    for p in history.bg_pixel:
        mask[p.y - roi_t, p.x - roi_l] = cv2.GC_BGD

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(crop, mask, crop_rect, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_MASK)

    mask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1).astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    greatest_contour = max(contours, key=cv2.contourArea)
    points = [Point(x=something[0][0], y=something[0][1]) for something in greatest_contour]
    points = [Point(x=p.x + roi_l, y=p.y+roi_t) for p in points]

    return points


def bounding_box(points):
    x_coordinates, y_coordinates = zip(*points)
    return [min(x_coordinates), min(y_coordinates), max(x_coordinates), max(y_coordinates)]
