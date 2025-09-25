# Bot-X User Guide

## Overview

Bot-X automates the reward flow of the Go Iv100 application by closing ads,
waiting for the countdown, tapping the *Continuar* arrow, dismissing the
"Recompensa conseguida" overlay, and finally triggering a new reward by
pressing the `+200 monedas por ver un anuncio` button. The behaviour matches the
reference screenshots supplied for this iteration.

The bot relies on OpenCV, PyAutoGUI and Tesseract OCR. Make sure Tesseract is
installed and available on your system path. On Windows you can install
[Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) and then add the
installation directory to the `PATH` environment variable. On macOS you can run
`brew install tesseract`; on Debian/Ubuntu systems run `sudo apt-get install
-tesseract-ocr`.

## First-time setup

1. **Install dependencies**
   ```bash
   pip install opencv-python pyautogui pytesseract pillow numpy
   ```
   The project assumes Python 3.10+.

2. **Calibrate the BlueStacks region**
   ```bash
   python bot-x-detector.py
   ```
   Choose option `1` and follow the on-screen instructions to draw the emulator
   window.

3. **Capture templates (optional but recommended)**
   - Option `2` now imports ready-made captures of the close button (`X`). Place
     the reference image(s) inside `templates/reference/` (for example the
     screenshot containing `X/Continuar/Recompensa en X segundos`) and provide
     the path when prompted. The helper copies them into the working templates
     folder without opening an extra window.
   - Option `3` works the same way for the `+200 monedas por ver un anuncio`
     button: point the prompt to the supplied screenshot and the bot will store
     grayscale templates automatically.

4. **Start the bot**
   - From the interactive menu choose option `4`.
   - Alternatively you can launch the quick mode after configuring the region:
     ```bash
     python bot-x-detector.py quick
     ```

## How the automation behaves

The detector now understands the ad flow shown in the latest screenshots:

| Screen | What the bot does |
| ------ | ----------------- |
| Ad close (`X` centred in the bottom half) | The bot prioritises that `X` detection and clicks it once. |
| Countdown (`Recompensa en N s`) | The bot waits without clicking until the message disappears. |
| Continue (`Continuar` label with arrow) | It clicks slightly to the right of the text to hit the arrow. |
| Reward confirmation (`Recompensa conseguida` with `X`) | It searches the upper overlay for an `X` and clicks it. |
| Main screen (`+200 monedas por ver un anuncio`) | It centres the CTA button and clicks it to trigger the next ad. |

The OCR-driven state machine prevents premature clicks and reduces repeated
inputs by throttling identical actions for 1.5 seconds.

## Tips for reliability

- Keep BlueStacks visible; overlaying windows may confuse the screenshot-based
  detection.
- Ensure the emulator resolution roughly matches the screenshots so that text
  remains legible to Tesseract.
- If PyAutoGUI raises a fail-safe error, move the mouse away from the corners or
  disable fail-safe mode in the configuration (`safe_mode: false`).
- Review `bot_detector.log` for diagnostic messages and to verify which
  detection path triggered each action.

## Troubleshooting

- **The bot cannot read text**: verify that Tesseract is installed and that your
  display scaling keeps text crisp. You can adjust BlueStacks resolution or use
  higher-quality templates.
- **It clicks the wrong place**: rerun calibration steps to capture fresh
  templates and double-check the configured region matches the emulator window.
- **No clicks happen**: confirm that the process has the necessary permissions
  to control the mouse (macOS requires enabling Accessibility access under
  System Preferences).

For further assistance inspect the log file and share the relevant snippets when
requesting support.
