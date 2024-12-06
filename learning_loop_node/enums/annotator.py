from enum import Enum


class AnnotationEventType(str, Enum):
    LeftMouseDown = 'left_mouse_down'
    RightMouseDown = 'right_mouse_down'
    MouseMove = 'mouse_move'
    LeftMouseUp = 'left_mouse_up'
    RightMouseUp = 'right_mouse_up'
    KeyUp = 'key_up'
    KeyDown = 'key_down'
