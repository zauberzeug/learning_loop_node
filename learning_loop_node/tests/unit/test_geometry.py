from ...detector.geometry import clip_box, clip_box_centered, clip_point


def test_box_inside_the_image_is_unchanged():
    assert clip_box(x1=10, y1=20, width=30, height=40, img_width=100, img_height=100) == (10, 20, 30, 40)


def test_box_is_clipped_to_the_image_bounds():
    assert clip_box(x1=-20, y1=-20, width=60, height=60, img_width=100, img_height=100) == (0, 0, 40, 40)
    assert clip_box(x1=80, y1=80, width=40, height=40, img_width=100, img_height=100) == (80, 80, 20, 20)


def test_box_fully_outside_the_image_collapses_to_zero_size():
    # The corner is only clamped at the lower bound, so it stays at 200 — the zero size is
    # what marks the box as empty. Detector output cannot reach here, because
    # non_max_suppression already clips every box into the image.
    assert clip_box(x1=200, y1=200, width=10, height=10, img_width=100, img_height=100) == (200, 200, 0, 0)


def test_box_corners_are_rounded():
    assert clip_box(x1=10.4, y1=10.6, width=20.0, height=20.0, img_width=100, img_height=100) == (10, 11, 20, 20)


def test_centered_box_keeps_its_centre_when_it_fits():
    assert clip_box_centered(x=50, y=50, width=20, height=20, img_width=100, img_height=100) == (50, 50, 20, 20)


def test_clipping_a_centered_box_moves_its_centre():
    # only the right half of the box is inside the image, so the centre moves right
    assert clip_box_centered(x=0, y=50, width=20, height=20, img_width=100, img_height=100) == (5, 50, 10, 20)


def test_point_is_clamped_into_the_image():
    assert clip_point(50, 50, 100, 100) == (50, 50)
    assert clip_point(-10, 150, 100, 100) == (0, 100)
