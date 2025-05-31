# HeadBlinkMouse

**Hands‑free mouse control using head movements + eye blinks**  
Powered by [MediaPipe FaceMesh](https://developers.google.com/mediapipe) and pure‑Python libraries.

Move the cursor with subtle head motions and trigger **left / right clicks** by blinking one eye – no extra hardware beyond a standard webcam.

---

## Why use it?

| Benefit | Details |
|---------|---------|
| **Accessibility** | Lets people with limited hand mobility, limb injuries, repetitive‑strain conditions, or temporary restrictions (e.g. broken arm) use a computer without a physical mouse. |
| **Low‑cost setup** | Runs on any machine with a webcam and a common CPU (GPU not required). |
| **Customisable** | Adjustable smoothing windows, alpha values, blink thresholds and cooldowns. Tune it to be as snappy or stable as you need. |
| **Portable** | Pure‑Python, cross‑platform (Windows / macOS / Linux*). |
| **Open source** | MIT‑licensed – free for personal **or** commercial use. |

\* On Linux, `pyautogui` requires an X11 session (Wayland needs extra steps).

---

## Features

* **Interactive calibration** – look at each screen corner and press **SPACE** to map face coordinates to pixels.
* **Dual smoothing pipeline**  
  * *Normal mode* – short averaging window for fast large moves.  
  * *Precise mode* – longer averaging when the intended move is small (ideal for clicking tiny UI elements).
* **Blink‑based clicks**  
  * Left‑eye blink → **left click**  
  * Right‑eye blink → **right click**
* Configurable constants right at the top of `head_blink_mouse.py`.
* Clean exit with the **ESC** key.

---

## Quick start

```bash
# Clone the repo
git clone https://github.com/dbudanec/HeadBlinkMouse.git
cd HeadBlinkMouse

# (Optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run
python head_blink_mouse.py
```

Follow the on‑screen prompts for calibration.  
Afterwards, move your head to steer the cursor and blink an eye to click.

---

## Tuning parameters

Constant | Purpose | Typical range
---------|---------|--------------
`NORMAL_SMOOTHING_WINDOW` | Moving‑average window for large moves | 1 – 5
`NORMAL_ALPHA` | EMA factor for large moves | 0.1 – 0.6
`PRECISE_SMOOTHING_WINDOW` | Window for tiny moves | 10 – 30
`PRECISE_ALPHA` | EMA factor for tiny moves | 0.02 – 0.2
`PRECISE_MODE_DISTANCE_THRESHOLD` | Switch distance (pixels) | 30 – 100
`BLINK_THRESHOLD` | Landmark distance to detect blink | 0.003 – 0.01
`CLICK_COOLDOWN` | Seconds between clicks | 0.3 – 1.0

Edit them in **head_blink_mouse.py**, save, and rerun to test.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Cursor drifts / jitters | Increase `PRECISE_SMOOTHING_WINDOW` or lower `PRECISE_ALPHA`. |
| Clicks mis‑fire or double | Tweak `BLINK_THRESHOLD` and/or increase `CLICK_COOLDOWN`; ensure good lighting. |
| High CPU usage | Lower webcam resolution with `opencv`; install `opencv-python-headless`. |
| Nothing happens on Linux | Make sure you are on X11, or install `python3-tk` and allow `uinput` for Wayland. |

---

## Security & Safety

* `pyautogui.FAILSAFE` is **disabled** so the pointer can reach screen edges.  
  Use **ESC** to quit.
* No admin/root privileges needed.
* Remember to take eye breaks during long sessions.

---

## Requirements

All dependencies are listed in [`requirements.txt`](requirements.txt):

```
opencv-python
mediapipe
pyautogui
```

Install them with `pip install -r requirements.txt`.

---

## License

This project is released under the [MIT License](LICENSE).

---

## Contributing

Bug reports and pull requests welcome!
Open an issue if you plan a large change.

---

## Acknowledgements

* [MediaPipe](https://developers.google.com/mediapipe) for real‑time face‑mesh detection.
* [`pyautogui`](https://pypi.org/project/PyAutoGUI/) for cross‑platform mouse control.
