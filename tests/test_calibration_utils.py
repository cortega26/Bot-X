"""Unit tests for calibration utility helpers."""

import os
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from calibration_utils import (
    compute_display_scale,
    import_reference_templates,
    map_display_to_image_coords,
)


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


def _create_sample_image(path: Path, color: tuple[int, int, int]) -> str:
    image = np.full((20, 30, 3), color, dtype=np.uint8)
    cv2.putText(image, "X", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), image)
    return str(path)


def test_import_reference_templates_creates_grayscale(tmp_path):
    source = _create_sample_image(tmp_path / "x.png", (0, 0, 255))
    destination = tmp_path / "templates"

    saved = import_reference_templates([source], str(destination), "x_template")

    assert saved == ["x_template_1.png"]

    stored = cv2.imread(str(destination / saved[0]), cv2.IMREAD_UNCHANGED)
    assert stored is not None
    # Grayscale images have two dimensions
    assert stored.ndim == 2


def test_import_reference_templates_respects_existing_index(tmp_path):
    destination = tmp_path / "templates"
    os.makedirs(destination, exist_ok=True)

    # Pre-create an existing template with index 4
    existing = destination / "x_template_4.png"
    _create_sample_image(existing, (0, 255, 0))

    src1 = _create_sample_image(tmp_path / "money1.png", (255, 0, 0))
    src2 = _create_sample_image(tmp_path / "money2.png", (0, 255, 255))

    saved = import_reference_templates(
        [src1, src2], str(destination), "x_template"
    )

    assert saved == ["x_template_5.png", "x_template_6.png"]

