"""Utilities to classify UI states based on OCR output.

These helpers extract actionable cues from the ad flow showcased in
provided reference screenshots. They operate purely on OCR tokens so they can
be unit-tested without relying on OpenCV or PyAutoGUI side effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional, Sequence, Tuple


@dataclass(frozen=True)
class OCRToken:
    """Lightweight container for a single OCR word."""

    text: str
    left: int
    top: int
    width: int
    height: int
    confidence: float

    @property
    def normalized(self) -> str:
        return (self.text or "").strip().lower()

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    @property
    def center(self) -> Tuple[int, int]:
        return self.left + self.width // 2, self.top + self.height // 2


class ScreenState(Enum):
    """High-level states recognised from the reference screenshots."""

    UNKNOWN = "unknown"
    AD_DISMISS = "ad_dismiss"
    COUNTDOWN = "countdown"
    CONTINUE_ARROW = "continue_arrow"
    REWARD_COMPLETE = "reward_complete"
    MAIN_REWARD_BUTTON = "main_reward_button"


def _aggregate_text(tokens: Iterable[OCRToken]) -> str:
    return " ".join(token.normalized for token in tokens if token.normalized)


def _first_token(tokens: Sequence[OCRToken], keyword: str) -> Optional[OCRToken]:
    keyword = keyword.lower()
    for token in tokens:
        if keyword in token.normalized:
            return token
    return None


def _filter_tokens(tokens: Sequence[OCRToken], keywords: Sequence[str]) -> list[OCRToken]:
    keyset = tuple(word.lower() for word in keywords)
    filtered = []
    for token in tokens:
        text = token.normalized
        if text and any(key in text for key in keyset):
            filtered.append(token)
    return filtered


def analyze_tokens(
    tokens: Sequence[OCRToken],
    image_shape: Tuple[int, int, int],
    *,
    min_confidence: float = 40.0,
) -> tuple[ScreenState, Optional[Tuple[int, int]]]:
    """Classify the current screen and return an optional click target.

    Parameters
    ----------
    tokens:
        OCR tokens extracted from the screenshot.
    image_shape:
        Shape of the screenshot array (``height, width, channels``).
    min_confidence:
        Confidence threshold used to filter noisy OCR tokens.

    Returns
    -------
    tuple[ScreenState, Optional[Tuple[int, int]]]
        Detected state and an optional point in screenshot coordinates that
        should be clicked to progress the ad flow.
    """

    height, width = image_shape[:2]
    valid_tokens = [
        token
        for token in tokens
        if token.confidence >= min_confidence and token.normalized
    ]

    if not valid_tokens:
        return ScreenState.UNKNOWN, None

    aggregated = _aggregate_text(valid_tokens)

    if "recompensa en" in aggregated:
        # Countdown before the Continue arrow becomes clickable.
        return ScreenState.COUNTDOWN, None

    arrow_token = _first_token(valid_tokens, "continuar")
    if arrow_token is not None:
        # The arrow button sits immediately to the right of the "Continuar" label.
        offset = max(40, arrow_token.width // 2)
        arrow_x = min(width - 5, arrow_token.right + offset)
        arrow_y = arrow_token.top + arrow_token.height // 2
        return ScreenState.CONTINUE_ARROW, (arrow_x, arrow_y)

    if "recompensa conseguida" in aggregated:
        # Overlay with the final reward confirmation.
        return ScreenState.REWARD_COMPLETE, None

    bottom_tokens = [
        token
        for token in _filter_tokens(valid_tokens, [
            "moneda",
            "monedas",
            "anuncio",
            "200",
            "+200",
            "ver",
        ])
        if token.top >= int(height * 0.55)
    ]

    if bottom_tokens:
        left = min(token.left for token in bottom_tokens)
        top = min(token.top for token in bottom_tokens)
        right = max(token.right for token in bottom_tokens)
        bottom = max(token.bottom for token in bottom_tokens)
        center_x = max(0, min(width - 1, (left + right) // 2))
        center_y = max(0, min(height - 1, (top + bottom) // 2))
        return ScreenState.MAIN_REWARD_BUTTON, (center_x, center_y)

    return ScreenState.UNKNOWN, None


__all__ = ["OCRToken", "ScreenState", "analyze_tokens"]
