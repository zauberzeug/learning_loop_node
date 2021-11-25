from typing import List, Optional
from learning_loop_node.annotation_node.annotation_tool import AnnotationTool
from learning_loop_node.annotation_node.data_classes import EventType, Point, SegmentationAnnotation, Shape, ToolOutput, UserInput
from uuid import uuid4
from pydantic import BaseModel
import numpy as np
import cv2
from icecream import ic
import logging


class Box(BaseModel):
    x: int
    y: int
    w: int
    h: int


class History(BaseModel):
    bbox: Optional[Box]
    rect_creation_finished: bool = False
    mouse_is_down: bool = False
    bg_pixel: List[Point] = []
    fg_pixel: List[Point] = []
    path_pixels: List[Point] = []
    annotation: Optional[SegmentationAnnotation]

    def to_svg_rect(self) -> str:

        # width = abs(self.x2-self.x)
        # height = abs(self.y2-self.y)
        # return f'<rect x="{min(self.x, self.x2)}" y="{min(self.y, self.y2)}" width="{width or 1}" height="{height or 1}" fill="blue">'
        return f'<rect x="{self.bbox.x}" y="{self.bbox.y}" width="{abs(self.bbox.w)}" height="{abs(self.bbox.h)}" stroke="blue" fill="transparent">'

    def to_svg_path(self, shift_pressed: bool) -> str:
        def create_path(pixel: List[Point]) -> str:
            path = f'M {pixel[0].x} {pixel[0].y}'
            for point in pixel[1:]:
                path += f' L {point.x} {point.y}'
            path += ' Z'
            return path

        svg_path = ''
        if self.path_pixels:
            if shift_pressed:
                path = create_path(self.path_pixels)
                svg_path = f'<path d="{path}" stroke="white" stroke-width="3" fill="white">'
            else:
                path = create_path(self.path_pixels)
                svg_path = f'<path d="{path}" stroke="red" stroke-width="3" fill="red">'
        return svg_path


class DemoSegmentationTool(AnnotationTool):
    history: Optional[History]

    async def handle_user_input(self, user_input: UserInput) -> ToolOutput:
        try:
            ic(self.history.rect_creation_finished)
            if self.history.rect_creation_finished == True and user_input.data.event_type == EventType.Enter_Pressed:
                self.history = None  # // simple resetting.
        except:
            pass

        coordinate = user_input.data.coordinate
        output = ToolOutput(svg="", annotation=None)
        if not self.history and user_input.data.event_type == EventType.MouseDown:
            self.history = History()
            self.history.bbox = Box(x=coordinate.x, y=coordinate.y, w=0, h=0)
            # start bbox Creation
            output.svg = self.history.to_svg_rect()
            return output

        elif self.history.rect_creation_finished == False and user_input.data.event_type == EventType.MouseMove:
            # update update bbox
            self.history.bbox.w = coordinate.x - self.history.bbox.x
            self.history.bbox.h = coordinate.y - self.history.bbox.y
            output.svg = self.history.to_svg_rect()
            return output
        elif self.history.rect_creation_finished == False and user_input.data.event_type == EventType.MouseUp:
            # end update bbox
            ic('hier')
            self.history.bbox.w = coordinate.x - self.history.bbox.x
            self.history.bbox.h = coordinate.y - self.history.bbox.y
            self.history.rect_creation_finished = True

            try:
                points = autofit(f'/data/{user_input.data.context.organization}/{user_input.data.context.project}/images/{user_input.data.image_uuid}.jpg',
                                 self.history)
            except:
                self.history = None
            self.history.annotation = SegmentationAnnotation(id=str(uuid4()), shape=Shape(points=[]),
                                                             image_id=user_input.data.image_uuid, category_id=user_input.data.category.id)
            self.history.annotation.shape.points = points
            ic(points)
            output.svg = self.history.to_svg_rect()
            output.annotation = self.history.annotation
            return output

        elif self.history.rect_creation_finished == True and user_input.data.event_type == EventType.MouseDown:
            self.history.mouse_is_down = True
            # start gathering bg vg points
            self.history.fg_pixel.append(
                user_input.data.coordinate) if user_input.data.is_shift_key_pressed else self.history.bg_pixel.append(user_input.data.coordinate)

        elif self.history.rect_creation_finished == True and user_input.data.event_type == EventType.MouseMove and self.history.mouse_is_down:
            # gathering bg vg points
            self.history.fg_pixel.append(
                user_input.data.coordinate) if user_input.data.is_shift_key_pressed else self.history.bg_pixel.append(user_input.data.coordinate)
            ic(self.history.fg_pixel)
            self.history.path_pixels.append(user_input.data.coordinate)
            output.svg = self.history.to_svg_path(user_input.data.is_shift_key_pressed)
            return output

        elif self.history.rect_creation_finished == True and user_input.data.event_type == EventType.MouseMove:
            return output

        elif self.history.rect_creation_finished == True and user_input.data.event_type == EventType.MouseUp:
            # editing complete
            self.history.mouse_is_down = False
            self.history.path_pixels = []
            # start gathering bg vg points
            self.history.fg_pixel.append(
                user_input.data.coordinate) if user_input.data.is_shift_key_pressed else self.history.bg_pixel.append(user_input.data.coordinate)
            # crabcut
            points = autofit(f'/data/{user_input.data.context.organization}/{user_input.data.context.project}/images/{user_input.data.image_uuid}.jpg',
                             self.history)
            self.history.annotation.shape.points = points
            output.annotation = self.history.annotation
            return output
        else:
            import logging
            logging.error('Invalid State')

        try:
            ic(user_input.data.image_uuid)
            ic(self.history.annotation.image_id)
        except:
            ic('Exception')

        if user_input.data.event_type == EventType.MouseDown:
            if not self.history:
                self.history = History()
                # currently we only support outside clicks.
            if user_input.data.is_shift_key_pressed:
                self.history.outside_clicks.append(user_input.data.coordinate)
            else:
                self.history.inside_clicks.append(user_input.data.coordinate)

            points = autofit(f'/data/{user_input.data.context.organization}/{user_input.data.context.project}/images/{user_input.data.image_uuid}.jpg',
                             self.history)

            if not self.history.annotation or self.history.annotation.image_id != user_input.data.image_uuid:
                ic('replacing annotation')
                self.history.annotation = SegmentationAnnotation(id=str(uuid4()), shape=Shape(points=[]),
                                                                 image_id=user_input.data.image_uuid, category_id=user_input.data.category.id)
            self.history.annotation.shape.points = points
            ic(self.history.annotation)
            return ToolOutput(svg="", annotation=self.history.annotation)

        return ToolOutput(svg="", annotation=None)


def autofit(image_path, history: History) -> List[Point]:
    logging.info('inside grab cut')
    img = cv2.imread(image_path)
    x_, y_, w_, h_ = int(history.bbox.x), int(history.bbox.y), int(history.bbox.w), int(history.bbox.h)
    # // 100*100 box
    # region-of-interest
    padding_factor = 4
    H, W = img.shape[:2]
    roi_l = max(x_ - padding_factor * w_, 0)  # x-400
    roi_t = max(y_ - padding_factor * h_, 0)  # y -400
    roi_r = min(x_ + (padding_factor + 1) * w_, W)  # x+500
    roi_b = min(y_ + (padding_factor + 1) * h_, H)  # y+500
    crop = img[roi_t:roi_b, roi_l:roi_r]
    crop_rect = x_ - roi_l, y_ - roi_t, w_, h_

    # grabcut
    mask = np.ones(crop.shape[:2], np.uint8) * cv2.GC_BGD
    mask[crop_rect[1]:crop_rect[1]+crop_rect[3],
         crop_rect[0]:crop_rect[0]+crop_rect[2]] = cv2.GC_PR_FGD
    mask[crop_rect[1]+crop_rect[3]//4:crop_rect[1]+3*crop_rect[3]//4,
         crop_rect[0]+crop_rect[2]//4:crop_rect[0]+3*crop_rect[2]//4] = cv2.GC_FGD

    for p in history.fg_pixel:
        mask[p.y, p.x] = cv2.GC_FGD

    for p in history.bg_pixel:
        mask[p.y, p.x] = cv2.GC_BGD

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(img, mask, crop_rect, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_MASK)
    # rows, cols = np.where(mask % 2 > 0)

    mask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1).astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    greatest_contour = max(contours, key=cv2.contourArea)
    points = [Point(x=something[0][0], y=something[0][1]) for something in greatest_contour]

    return points

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
        cv2.fillPoly(mask, polygon, cv2.GC_PR_FGD)
        # cv2.fillPoly(mask, polygon, cv2.GC_FGD)
    for p in outside_points:
        mask[p[1], p[0]] = cv2.GC_FGD
    offset = 10
    for p in inside_points:

        for y in range(p[1]-offset, p[1]+offset):
            for x in range(p[0]-offset, p[0]+offset):
                ic(x, y)
                # mask[y, x] = cv2.GC_PR_BGD
                mask[y, x] = cv2.GC_BGD
        ic(p)

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
