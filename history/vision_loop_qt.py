#!/usr/bin/env python3
"""
TFT -> Walkpad OCR prototype (single file)

What this does:
- Finds the QuickTime mirroring window (so we DON'T capture your whole ultrawide)
- Crops the right-side TFT player list (sidebar)
- Uses OCR to find YOUR name in that list (e.g. "sipi")
- Reads the HP number for your row (HP is typically BELOW the name in this UI)
- Shows debug windows (SIDEBAR / THRESH / DIGIT_ROI)
- Scans at low FPS (default: 1 scan/sec)

What this does NOT do (yet):
- It does NOT send treadmill commands (requests are commented out on purpose)

Install:
    brew install tesseract
    python3 -m pip install mss opencv-python numpy pytesseract pyobjc-framework-Quartz requests

Run:
    python3 tft_ocr_quicktime.py

macOS permission:
- System Settings -> Privacy & Security -> Screen Recording -> enable Terminal (or your IDE)

Tuning:
- You already tuned sidebar crop:
    sx = int(0.85 * W)
    sy = int(0.09 * H)
    sw = int(0.15 * W)
    sh = int(0.50 * H)
  That is used below.

Debug:
- If it says "name not found", it will print the top OCR words it *did* see so we can
  adjust matching (or add fuzzy match if needed).
"""

import time
import re
from collections import deque
from typing import Optional, Dict, Any, Tuple

import cv2
import numpy as np
import pytesseract
import mss

# requests is optional; left here so you can uncomment later
# import requests

from Quartz import (
    CGWindowListCopyWindowInfo,
    kCGWindowListOptionOnScreenOnly,
    kCGNullWindowID,
)

# -----------------------------
# CONFIG
# -----------------------------
MY_NAME = "sipi"

WINDOW_OWNER = "QuickTime Player"
# If your QT window title differs, change this; otherwise we pick the largest QT window.
WINDOW_TITLE_CONTAINS = "Movie Recording"

SCAN_INTERVAL_SEC = 1.0  # 1 FPS
DEBUG_PREVIEW = True

# OCR smoothing (helps when OCR occasionally misreads)
SMOOTHING_WINDOW = 5
MIN_SAMPLES_TO_USE = 3

# Optional: if you later turn commands on, keep rate limits
COMMAND_INTERVAL_SEC = 3.0

API = "http://127.0.0.1:8765"


# -----------------------------
# Quartz: find QuickTime window bounds
# -----------------------------
def find_quicktime_window() -> Optional[Dict[str, Any]]:
    windows = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID)

    candidates = []
    for w in windows:
        owner = w.get("kCGWindowOwnerName", "")
        title = w.get("kCGWindowName", "") or ""
        if owner == WINDOW_OWNER and WINDOW_TITLE_CONTAINS.lower() in title.lower():
            b = w.get("kCGWindowBounds", {})
            candidates.append((w, b))

    # fallback: any QT window
    if not candidates:
        for w in windows:
            owner = w.get("kCGWindowOwnerName", "")
            if owner == WINDOW_OWNER:
                b = w.get("kCGWindowBounds", {})
                candidates.append((w, b))

    if not candidates:
        return None

    # pick the largest window (usually the recording view)
    candidates.sort(key=lambda wb: wb[1].get("Width", 0) * wb[1].get("Height", 0), reverse=True)
    w, b = candidates[0]

    return {
        "title": w.get("kCGWindowName", "") or "",
        "x": int(b.get("X", 0)),
        "y": int(b.get("Y", 0)),
        "w": int(b.get("Width", 0)),
        "h": int(b.get("Height", 0)),
    }


# -----------------------------
# Image preprocessing + OCR helpers
# -----------------------------
def preprocess_for_text(img_bgr: np.ndarray) -> np.ndarray:
    """
    Preprocess image for OCR over varying TFT backgrounds.
    Output is a binary image (white-ish text on black-ish background).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Upscale improves OCR on small text
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Adaptive threshold handles changing background better than a fixed cutoff
    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,   # block size
        5     # C
    )

    # Thicken strokes a little
    kernel = np.ones((2, 2), np.uint8)
    th = cv2.dilate(th, kernel, iterations=1)
    return th


def dump_ocr_words(thresh_img: np.ndarray, limit: int = 30) -> None:
    """
    Prints what Tesseract thinks the words are in this frame (helpful when "name not found").
    """
    data = pytesseract.image_to_data(
        thresh_img,
        output_type=pytesseract.Output.DICT,
        config="--psm 6"
    )

    rows = []
    for i, txt in enumerate(data["text"]):
        word = (txt or "").strip()
        if not word:
            continue
        conf = data["conf"][i]
        try:
            conf_val = float(conf)
        except Exception:
            conf_val = -1.0
        rows.append((conf_val, word))

    rows.sort(reverse=True, key=lambda x: x[0])
    print("Top OCR words (conf, word):")
    for conf, word in rows[:limit]:
        print(f"  {conf:>6.1f}  '{word}'")


def find_name_bbox(thresh_img: np.ndarray, target_name: str) -> Optional[Tuple[int, int, int, int]]:
    """
    Returns (x, y, w, h) for the OCR word that matches target_name, in THRESH image coords.
    """
    data = pytesseract.image_to_data(
        thresh_img,
        output_type=pytesseract.Output.DICT,
        config="--psm 6 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )

    target = target_name.lower()
    for i, txt in enumerate(data["text"]):
        word = (txt or "").strip().lower()
        if not word:
            continue

        # Clean OCR noise: keep letters/numbers only
        word_clean = re.sub(r"[^a-z0-9]", "", word)

        if word_clean == target:
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            return (x, y, w, h)

    return None


def ocr_digits(thresh_img: np.ndarray) -> Optional[int]:
    """
    OCR digits only from a thresholded (binary) image.
    """
    txt = pytesseract.image_to_string(
        thresh_img,
        config="--psm 7 -c tessedit_char_whitelist=0123456789"
    ).strip()
    return int(txt) if txt.isdigit() else None


def read_hp_from_quicktime_frame(frame_bgr: np.ndarray, my_name: str, debug: bool = False):
    """
    1) Crop right sidebar from the QuickTime frame
    2) OCR word boxes to find your name
    3) OCR digits from region just BELOW your name (HP sits below name in this UI)
    """
    H, W = frame_bgr.shape[:2]

    # ---- Sidebar crop (your tuned values) ----
    sx = int(0.85 * W)
    sy = int(0.09 * H)
    sw = int(0.15 * W)
    sh = int(0.50 * H)
    sidebar = frame_bgr[sy:sy+sh, sx:sx+sw]

    th = preprocess_for_text(sidebar)

    bbox = find_name_bbox(th, my_name)
    if not bbox:
        if debug:
            dump_ocr_words(th, limit=30)
            return None, {
                "sidebar": sidebar,
                "thresh": th,
                "digit_roi": None,
                "note": "name not found",
                "name_bbox": None,
                "roi_box": None,
                "sidebar_crop": (sx, sy, sw, sh),
            }
        return None, None

    x, y, bw, bh = bbox

    # ---- Digit ROI BELOW the name (not to the right) ----
    # Expand horizontally a bit, and look in a band under the name.
    roi_x1 = max(0, x - int(0.05 * th.shape[1]))
    roi_x2 = min(th.shape[1], x + bw + int(0.25 * th.shape[1]))

    roi_y1 = min(th.shape[0] - 1, y + bh + 4)
    roi_y2 = min(th.shape[0], roi_y1 + int(0.18 * th.shape[0]))  # band below

    digit_roi = th[roi_y1:roi_y2, roi_x1:roi_x2]

    if digit_roi.size > 0:
        digit_roi = cv2.medianBlur(digit_roi, 3)

    hp = ocr_digits(digit_roi) if digit_roi.size > 0 else None

    dbg = None
    if debug:
        dbg = {
            "sidebar": sidebar,
            "thresh": th,
            "digit_roi": digit_roi,
            "note": "ok" if hp is not None else "digits not read",
            "name_bbox": bbox,
            "roi_box": (roi_x1, roi_y1, roi_x2 - roi_x1, roi_y2 - roi_y1),
            "sidebar_crop": (sx, sy, sw, sh),
        }

    return hp, dbg


# -----------------------------
# (Later) health -> treadmill mapping
# -----------------------------
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


# -----------------------------
# Main loop
# -----------------------------
def main():
    hp_window = deque(maxlen=SMOOTHING_WINDOW)
    last_sent = 0.0

    with mss.mss() as sct:
        while True:
            t0 = time.time()

            win = find_quicktime_window()
            if not win or win["w"] < 200 or win["h"] < 200:
                print("QuickTime window not found / too small. Make sure QuickTime is visible.")
                time.sleep(1.0)
                continue

            region = {"left": win["x"], "top": win["y"], "width": win["w"], "height": win["h"]}
            img = np.array(sct.grab(region))  # BGRA
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            hp, dbg = read_hp_from_quicktime_frame(frame, my_name=MY_NAME, debug=True)

            if hp is not None and 0 <= hp <= 100:
                hp_window.append(hp)

            # Smoothing (median)
            smoothed_hp = None
            if len(hp_window) >= MIN_SAMPLES_TO_USE:
                smoothed_hp = int(np.median(np.array(hp_window)))

            # Console output
            if dbg:
                sc = dbg.get("sidebar_crop")
                if sc:
                    sx, sy, sw, sh = sc
                    print(f"[{win['title']}] sidebar crop sx={sx} sy={sy} sw={sw} sh={sh}")

            if hp is None:
                print(f"[{win['title']}] HP=??  (reason: {dbg.get('note') if dbg else 'unknown'})")
            else:
                print(f"[{win['title']}] HP={hp}  smooth={smoothed_hp}  (note: {dbg.get('note') if dbg else 'ok'})")

            # Debug windows
            if DEBUG_PREVIEW and dbg:
                cv2.imshow("SIDEBAR", dbg["sidebar"])
                cv2.imshow("SIDEBAR_THRESH", dbg["thresh"])
                if dbg["digit_roi"] is not None:
                    cv2.imshow("DIGIT_ROI", dbg["digit_roi"])
                else:
                    cv2.imshow("DIGIT_ROI", np.zeros((60, 200), dtype=np.uint8))
                cv2.waitKey(1)

            # (Commented out) send treadmill commands once OCR is stable
            # if smoothed_hp is not None:
            #     now = time.time()
            #     if now - last_sent >= COMMAND_INTERVAL_SEC:
            #         speed, incline = map_hp_to_targets(smoothed_hp)
            #         requests.post(f"{API}/speed", json={"kmh": speed}, timeout=1)
            #         requests.post(f"{API}/incline", json={"percent": incline}, timeout=1)
            #         print(f"  → treadmill speed={speed} km/h incline={incline}%")
            #         last_sent = now

            # 1 FPS loop
            elapsed = time.time() - t0
            time.sleep(max(0.0, SCAN_INTERVAL_SEC - elapsed))


if __name__ == "__main__":
    main()