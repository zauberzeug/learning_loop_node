"""Box and point clipping shared by every detector node.

The loop stores a box as its top-left corner plus a size, which is the form :func:`clip_box`
takes and produces.
"""


def clip_box(
    *,
    x1: float,
    y1: float,
    width: float,
    height: float,
    img_width: int,
    img_height: int,
) -> tuple[int, int, int, int]:
    """
    Clip a top-left-anchored box to the image bounds.

    Both corners are clamped into the image; a box wholly outside it comes back with a size of 0.

    :return: The clipped ``(x1, y1, width, height)``; the size is never negative.
    """
    clipped_x1 = round(_clamp(x1, img_width))
    clipped_y1 = round(_clamp(y1, img_height))
    clipped_x2 = round(_clamp(x1 + width, img_width))
    clipped_y2 = round(_clamp(y1 + height, img_height))

    clipped_width = max(clipped_x2 - clipped_x1, 0)
    clipped_height = max(clipped_y2 - clipped_y1, 0)

    return clipped_x1, clipped_y1, clipped_width, clipped_height


def clip_point(x: float, y: float, img_width: int, img_height: int) -> tuple[float, float]:
    """Clamp a point into the image bounds."""
    return _clamp(x, img_width), _clamp(y, img_height)


def _clamp(value: float, upper: int) -> float:
    return min(max(0.0, value), float(upper))
