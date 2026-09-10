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
    """Clip a top-left-anchored box to the image bounds.

    :return: The clipped ``(x1, y1, width, height)``; the size is never negative.
    """
    x2 = x1 + width
    y2 = y1 + height

    clipped_x1 = round(max(0.0, x1))
    clipped_y1 = round(max(0.0, y1))
    clipped_x2 = round(min(float(img_width), x2))
    clipped_y2 = round(min(float(img_height), y2))

    clipped_width = max(clipped_x2 - clipped_x1, 0)
    clipped_height = max(clipped_y2 - clipped_y1, 0)

    return clipped_x1, clipped_y1, clipped_width, clipped_height


def clip_point(x: float, y: float, img_width: int, img_height: int) -> tuple[float, float]:
    """Clamp a point into the image bounds."""
    x = min(max(0, x), img_width)
    y = min(max(0, y), img_height)
    return x, y
