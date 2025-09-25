"""Utility helpers for calibration workflows."""

from __future__ import annotations


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

