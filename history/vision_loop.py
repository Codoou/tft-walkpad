import time
import cv2
import numpy as np
import pytesseract
import mss
import requests
from collections import deque

API = "http://127.0.0.1:8765"

# smoothing
hp_window = deque(maxlen=5)
last_command_ts = 0

def map_hp_to_targets(hp: int):
    if hp >= 80:
        return 0.8, 0.0
    if hp >= 60:
        return 1.2, 2.0
    if hp >= 40:
        return 1.6, 4.0
    if hp >= 20:
        return 2.0, 6.0
    return 2.4, 8.0

with mss.mss() as sct:
    monitor = sct.monitors[1]

    while True:
        img = np.array(sct.grab(monitor))
        h, w, _ = img.shape

        # ---- ROI (top-right) ----
        x = int(0.80 * w)
        y = int(0.03 * h)
        rw = int(0.18 * w)
        rh = int(0.08 * h)

        roi = img[y:y+rh, x:x+rw]

        # preprocess
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

        # OCR digits only
        text = pytesseract.image_to_string(
            thresh,
            config="--psm 7 -c tessedit_char_whitelist=0123456789"
        ).strip()

        if text.isdigit():
            hp = int(text)
            if 0 <= hp <= 100:
                hp_window.append(hp)

        if len(hp_window) >= 3:
            smoothed_hp = int(np.median(hp_window))
            now = time.time()

            # rate limit commands
            if now - last_command_ts > 7:
                speed, incline = map_hp_to_targets(smoothed_hp)

                # requests.post(f"{API}/speed", json={"kmh": speed}, timeout=1)
                # requests.post(f"{API}/incline", json={"percent": incline}, timeout=1)

                last_command_ts = now
                print(f"HP={smoothed_hp} → speed={speed} incline={incline}")

        time.sleep(0.3)