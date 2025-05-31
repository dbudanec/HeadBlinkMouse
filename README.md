# HeadBlinkMouse
A Python script for hands-free mouse control using MediaPipe FaceMesh.  Tracks head movements for cursor positioning and uses single-eye blinks for left/right clicks. Includes an interactive calibration and adjustable smoothing parameters for precise pointer control.

Hands-free mouse control powered by **MediaPipe FaceMesh**.  
Move the cursor with subtle head motions and trigger left/right clicks by blinking one eye — no hardware beyond a webcam needed.

---

## Why use it?

| Benefit | Details |
|---------|---------|
| **Accessibility** | Enables computer interaction for users with limited hand mobility, limb injuries, repetitive-strain conditions, or temporary restrictions (e.g. broken arm). |
| **Low-cost setup** | Works with any standard webcam and common CPUs (no GPU required). |
| **Customisable** | Adjustable smoothing and blink thresholds let you fine-tune responsiveness vs. stability. |
| **Portable** | Pure-Python, cross-platform (Windows / macOS / Linux¹). |
| **Open source** | MIT-licensed — free for personal or commercial use. |

> ¹ On Linux, `pyautogui` needs an X11 desktop session.

---

## Features

* **Interactive calibration** – guides you through looking at the four screen corners, mapping face coordinates to pixels.
* **Dual smoothing pipeline**  
  * *Normal mode* – quick response for large moves.  
  * *Precise mode* – heavy smoothing when the intended move is small (ideal for clicking tiny UI elements).
* **Blink-based clicks**  
  * Left-eye blink → **left click**  
  * Right-eye blink → **right click**
* **Configurable constants** at the top of the script (smoothing windows, alpha values, blink threshold, cooldown, etc.).
* Clean exit with **ESC**.

---

## Installation

1. **Clone** the repo  
   ```bash
   git clone https://github.com/dbudanec/HeadBlinkMouse.git
   cd HeadBlinkControl
