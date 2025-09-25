"""Tests for safety guards in :mod:`bot-x-detector`."""

import importlib
import sys
import types


if 'pyautogui' not in sys.modules:
    dummy_pyautogui = types.ModuleType('pyautogui')
    dummy_pyautogui.FAILSAFE = False
    dummy_pyautogui.PAUSE = 0

    def _position():
        return 0, 0

    def _screenshot(region=None):
        from PIL import Image
        import numpy as _np

        data = _np.zeros((10, 10, 3), dtype=_np.uint8)
        return Image.fromarray(data)

    def _moveTo(*args, **kwargs):
        return None

    def _click(*args, **kwargs):
        return None

    def _ease(*args, **kwargs):
        return None

    dummy_pyautogui.position = _position
    dummy_pyautogui.screenshot = _screenshot
    dummy_pyautogui.moveTo = _moveTo
    dummy_pyautogui.click = _click
    dummy_pyautogui.easeInOutQuad = _ease

    sys.modules['pyautogui'] = dummy_pyautogui


if 'pytesseract' not in sys.modules:
    dummy_pytesseract = types.ModuleType('pytesseract')
    dummy_pytesseract.Output = types.SimpleNamespace(DICT='DICT')

    def _image_to_data(*args, **kwargs):
        return {
            'text': [],
            'conf': [],
            'left': [],
            'width': [],
            'top': [],
            'height': [],
        }

    dummy_pytesseract.image_to_data = _image_to_data
    sys.modules['pytesseract'] = dummy_pytesseract


bot_detector = importlib.import_module("bot-x-detector")
np = bot_detector.np
XDetectorBot = bot_detector.XDetectorBot


def make_bot_without_templates():
    """Return a bot instance with empty template collections."""

    bot = XDetectorBot()
    bot.x_templates = []
    bot.money_templates = []
    return bot


def test_detect_money_button_ignores_larger_templates(monkeypatch):
    """Large templates should be skipped to avoid OpenCV assertions."""

    bot = make_bot_without_templates()

    oversized_template = np.zeros((150, 200), dtype=np.uint8)
    bot.money_templates = [{
        'name': 'oversized.png',
        'image': oversized_template,
        'shape': oversized_template.shape,
    }]

    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    detections = bot.detect_money_button(frame)

    assert detections == []


def test_detect_x_template_matching_skips_oversized_resized_template(monkeypatch):
    """Scaled templates larger than the frame must be ignored safely."""

    bot = make_bot_without_templates()

    monkeypatch.setattr(
        bot_detector.np,
        "linspace",
        lambda *args, **kwargs: bot_detector.np.array([1.2, 1.4])
    )

    base_template = np.zeros((100, 100), dtype=np.uint8)
    bot.x_templates = [{
        'name': 'base.png',
        'image': base_template,
        'shape': base_template.shape,
    }]

    frame = np.zeros((80, 80, 3), dtype=np.uint8)

    detections = bot.detect_x_template_matching(frame)

    assert detections == []
