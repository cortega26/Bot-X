"""Utility helpers for calibration workflows."""

from __future__ import annotations

import os
import re
from typing import Iterable, List

try:
    import cv2 as _cv2
except ModuleNotFoundError:  # pragma: no cover - exercised indirectly in tests
    _cv2 = None


def _require_cv2():
    if _cv2 is None:
        raise RuntimeError(
            "OpenCV (cv2) is required for calibration utilities but is not installed."
        )
    return _cv2


def compute_display_scale(
    width: int,
    height: int,
    max_width: int = 1200,
    max_height: int = 900,
) -> float:
    """Compute a safe scale factor for preview windows.

    Parameters
    ----------
    width, height:
        Dimensions of the capture region in pixels.
    max_width, max_height:
        Maximum size for the preview window.

    Returns
    -------
    float
        Scale factor in the range ``(0, 1]`` that preserves aspect ratio.
    """

    width = max(int(width or 0), 1)
    height = max(int(height or 0), 1)

    width_scale = max_width / float(width)
    height_scale = max_height / float(height)

    scale = min(1.0, width_scale, height_scale)
    return max(scale, 1e-3)


def map_display_to_image_coords(x: int, y: int, scale: float) -> tuple[int, int]:
    """Translate coordinates from a scaled preview to the original frame."""

    if scale <= 0:
        raise ValueError("scale must be greater than zero")

    x_img = int(round(x / scale))
    y_img = int(round(y / scale))
    return x_img, y_img


def _next_template_index(
    destination_dir: str,
    prefix: str,
    existing: Iterable[str] | None = None,
) -> int:
    """Return the next sequential index for ``prefix`` templates.

    Parameters
    ----------
    destination_dir:
        Directory that contains the templates.
    prefix:
        Prefix used to build template file names (e.g. ``"x_template"``).
    existing:
        Optional iterable of filenames to inspect instead of reading the
        filesystem. Useful for testing.
    """

    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)\.\w+$", re.IGNORECASE)

    if existing is None:
        try:
            existing = os.listdir(destination_dir)
        except FileNotFoundError:
            existing = []

    indices = [
        int(match.group(1))
        for name in existing
        if (match := pattern.match(name)) is not None
    ]

    return max(indices, default=0) + 1


def import_reference_templates(
    file_paths: Iterable[str],
    destination_dir: str,
    prefix: str,
) -> List[str]:
    """Copy reference images into the template folder.

    The helper reads each ``file_paths`` entry, converts the image to
    grayscale and stores it in ``destination_dir`` using a sequential
    ``{prefix}_N.png`` naming scheme. The new filenames are returned in the
    order they were processed.
    """

    file_paths = list(file_paths)
    if not file_paths:
        raise ValueError("at least one reference image path is required")

    os.makedirs(destination_dir, exist_ok=True)
    opencv = _require_cv2()
    next_index = _next_template_index(destination_dir, prefix)

    saved_files: List[str] = []

    for path in file_paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"reference image not found: {path}")

        image = opencv.imread(path, opencv.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"unable to read image: {path}")

        if image.ndim == 2:
            gray = image
        else:
            channels = image.shape[2]
            if channels == 4:
                gray = opencv.cvtColor(image, opencv.COLOR_BGRA2GRAY)
            elif channels == 3:
                gray = opencv.cvtColor(image, opencv.COLOR_BGR2GRAY)
            else:
                raise ValueError(
                    "unsupported channel count for image conversion: "
                    f"{channels}"
                )

        filename = f"{prefix}_{next_index}.png"
        output_path = os.path.join(destination_dir, filename)
        if not opencv.imwrite(output_path, gray):
            raise ValueError(f"failed to write template: {output_path}")

        saved_files.append(filename)
        next_index += 1

    return saved_files

