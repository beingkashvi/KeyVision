# KeyVision: Virtual Keyboard using Hand Tracking and OpenCV

## Overview

This project demonstrates a real-time virtual keyboard application using Python, OpenCV, and MediaPipe. Users can interact with an on-screen keyboard by pinching their index finger and thumb in front of the webcam. Each pinch selects a key, which is then displayed on-screen and can optionally be logged to a text file in real time.

![Virtual Keyboard Demo](demo/output.gif)

## Features

* **Real-time Hand Tracking**: Uses MediaPipe Hands to detect and track a single hand in the camera feed.
* **Customizable UI**: Semi-transparent keys with hover and active states, customizable colors, shadows, and icon support via Pillow and Font Awesome.
* **Pinch Gesture Input**: Select keys by pinching the index finger and thumb.
* **Backspace Icon**: Integrated Font Awesome backspace icon for deleting characters.
* **Live Text Logging**: Optionally log typed characters to a live-updated text file (`typed_text.txt`).
* **Session Logging**: Appends entire session text on exit, creating a persistent log of typed sessions.

## Requirements

* Python 3.7–3.10
* OpenCV (`opencv-python`)
* MediaPipe (`mediapipe`)
* NumPy (`numpy`)
* cvzone (optional, for rounded corner UI)
* Pillow (`Pillow`)
* Font Awesome Free (`fontawesome-free`)
* pynput (`pynput`)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/virtual-keyboard.git
   cd virtual-keyboard
   ```

2. Install required packages:

   ```bash
   pip install opencv-python mediapipe numpy cvzone Pillow fontawesome-free pynput
   ```

3. Download `fa-solid-900.ttf` from the Font Awesome Free repo and place it in a `fonts/` folder:

   ```
   virtual-keyboard/
   ├── fonts/
   │   └── fa-solid-900.ttf
   └── virtual_keyboard.py
   ```

## Usage

1. Open your editor and ensure the working directory contains `virtual_keyboard.py` and the `fonts/` folder.
2. (Optional) Open `typed_text.txt` in VS Code with auto-reload enabled to see live updates.
3. Run the script:

   ```bash
   python virtual_keyboard.py
   ```
4. Focus the “Virtual Keyboard” window, hover your index finger over a key, and pinch your index finger and thumb to select.
5. Press `q` (or `Esc`) in the keyboard window to quit. Your session’s typed text will be appended to `typed_text.txt`.

## Configuration

* **Key Layout & Positioning**: Modify `keys`, `origin_x`, `origin_y`, `spacing_x`, and `spacing_y` at the top of the script.
* **UI Colors**: Customize `drawAll` and `drawTextBox` functions by changing BGR color tuples and alpha values.
* **Pinch Threshold**: Adjust the distance threshold (`40` pixels by default) to tune sensitivity.

## Troubleshooting

* **No hand detected**: Ensure good lighting and avoid cluttered backgrounds. Adjust `min_detection_confidence`.
* **Pinch not recognized**: Increase threshold or add smoothing to landmarks.
* **Font not found**: Verify `fa-solid-900.ttf` is in the `fonts/` directory and `fa_path` is correct.

---

*Happy typing!*
