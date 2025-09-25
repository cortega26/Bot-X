"""Unit tests for calibration utility helpers."""

from calibration_utils import compute_display_scale, map_display_to_image_coords


def test_compute_display_scale_limits_window_size():
    # Large region should be scaled down to fit bounds
    scale = compute_display_scale(2000, 1200, max_width=1000, max_height=800)
    assert 0 < scale <= 0.5


def test_compute_display_scale_defaults_to_one_for_small_regions():
    assert compute_display_scale(400, 300) == 1.0


def test_map_display_to_image_coords_rounds_correctly():
    assert map_display_to_image_coords(150, 90, 0.5) == (300, 180)


def test_map_display_to_image_coords_invalid_scale():
    try:
        map_display_to_image_coords(10, 10, 0)
    except ValueError as exc:
        assert "scale" in str(exc)
    else:
        raise AssertionError("Expected ValueError for zero scale")

