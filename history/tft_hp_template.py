#!/usr/bin/env python3
"""
TFT HP reader via QuickTime window capture + TEMPLATE MATCHING + robust smoothing

What this does:
- Captures ONLY the QuickTime window (so your ultrawide doesn't matter)
- Crops the right-side TFT sidebar
- Template-matches your row (your template matches the whole row block)
- Crops the HP region INSIDE the matched box (using your tuned bounds)
- OCRs digits only
- Adds two safety layers:
    1) HOLD-LAST: if HP can't be read (different screen / no list), we keep last known HP
    2) MAJORITY-VOTE: over last N valid reads (default 10), accept an HP only if it appears >= K times (default 6)
       (prevents random OCR glitches from spiking treadmill commands later)

What this does NOT do (yet):
- It does NOT send treadmill commands (requests are commented out)

Install:
    brew install tesseract
    python3 -m pip install mss opencv-python numpy pytesseract pyobjc-framework-Quartz requests

Run:
    python3 tft_hp_template.py

macOS permission:
- System Settings -> Privacy & Security -> Screen Recording -> enable Terminal / Python
"""

import time
from collections import deque, Counter
from typing import Optional, Dict, Any, Tuple

import cv2
import numpy as np
import pytesseract
import mss

# requests optional; keep import commented until you're ready
# import requests

from Quartz import (
    CGWindowListCopyWindowInfo,
    kCGWindowListOptionOnScreenOnly,
    kCGNullWindowID,
)

# -----------------------------
# CONFIG
# -----------------------------
WINDOW_OWNER = "QuickTime Player"
WINDOW_TITLE_CONTAINS = "Movie Recording"

SCAN_INTERVAL_SEC = 1.0
DEBUG_PREVIEW = True

# Sidebar crop relative to QuickTime frame (your tuned values)
SIDEBAR_SX = 0.85
SIDEBAR_SY = 0.09
SIDEBAR_SW = 0.15
SIDEBAR_SH = 0.50

# Template image (matches your whole row block)
TEMPLATE_PATH = "portrait_template.png"
MATCH_MIN = 0.65

# HP bounds inside match box (YOUR NEW BOUNDS)
# These are ratios of match width/height.
HP_RX1 = 0.00
HP_RX2 = 0.45
HP_RY1 = 0.63
HP_RY2 = 0.95

# Robustness settings
HP_MIN = 0
HP_MAX = 100

# Majority vote filter
HISTORY_N = 10         # keep last 10 valid reads
MAJORITY_K = 6         # require same value to appear 6 times before accepting as "stable"

# Hold-last behavior
HOLD_LAST_ENABLED = True

# Optional: later treadmill API endpoint
API = "http://127.0.0.1:8765"


# -----------------------------
# QuickTime window finder
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

    if not candidates:
        for w in windows:
            owner = w.get("kCGWindowOwnerName", "")
            if owner == WINDOW_OWNER:
                b = w.get("kCGWindowBounds", {})
                candidates.append((w, b))

    if not candidates:
        return None

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
# Image helpers
# -----------------------------
def crop_sidebar(frame_bgr: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    H, W = frame_bgr.shape[:2]
    sx = int(SIDEBAR_SX * W)
    sy = int(SIDEBAR_SY * H)
    sw = int(SIDEBAR_SW * W)
    sh = int(SIDEBAR_SH * H)

    # clamp to bounds
    sx = max(0, min(sx, W - 1))
    sy = max(0, min(sy, H - 1))
    sw = max(1, min(sw, W - sx))
    sh = max(1, min(sh, H - sy))

    sidebar = frame_bgr[sy:sy + sh, sx:sx + sw]
    return sidebar, (sx, sy, sw, sh)


def load_template(path: str) -> np.ndarray:
    tpl = cv2.imread(path, cv2.IMREAD_COLOR)
    if tpl is None:
        raise RuntimeError(f"Could not read TEMPLATE_PATH='{path}'. Fix the path or place the file next to this script.")
    return tpl


def match_template(sidebar_bgr: np.ndarray, template_bgr: np.ndarray) -> Tuple[float, Tuple[int, int], Tuple[int, int]]:
    sb = cv2.cvtColor(sidebar_bgr, cv2.COLOR_BGR2GRAY)
    tp = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)

    sb = cv2.GaussianBlur(sb, (3, 3), 0)
    tp = cv2.GaussianBlur(tp, (3, 3), 0)

    res = cv2.matchTemplate(sb, tp, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)

    h, w = tp.shape[:2]
    return float(max_val), (int(max_loc[0]), int(max_loc[1])), (w, h)


def safe_crop(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    H, W = img.shape[:2]
    x1 = max(0, min(x1, W))
    x2 = max(0, min(x2, W))
    y1 = max(0, min(y1, H))
    y2 = max(0, min(y2, H))
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0, 3), dtype=img.dtype) if img.ndim == 3 else np.empty((0, 0), dtype=img.dtype)
    return img[y1:y2, x1:x2]


def preprocess_digits(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((2, 2), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
    return th


def ocr_digits(thresh_img: np.ndarray) -> Optional[int]:
    txt = pytesseract.image_to_string(
        thresh_img,
        config="--psm 7 -c tessedit_char_whitelist=0123456789"
    ).strip()
    return int(txt) if txt.isdigit() else None


def show_or_blank(win_name: str, img: Optional[np.ndarray], blank_shape=(120, 260), gray=False) -> None:
    if img is None or img.size == 0:
        if gray:
            blank = np.zeros(blank_shape, dtype=np.uint8)
        else:
            blank = np.zeros((blank_shape[0], blank_shape[1], 3), dtype=np.uint8)
        cv2.imshow(win_name, blank)
    else:
        cv2.imshow(win_name, img)


def hp_roi_from_match_box(sidebar_bgr: np.ndarray, top_left: Tuple[int, int], wh: Tuple[int, int]) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    x, y = top_left
    mw, mh = wh

    rx1 = x + int(HP_RX1 * mw)
    rx2 = x + int(HP_RX2 * mw)
    ry1 = y + int(HP_RY1 * mh)
    ry2 = y + int(HP_RY2 * mh)

    roi = safe_crop(sidebar_bgr, rx1, ry1, rx2, ry2)
    return roi, (rx1, ry1, rx2, ry2)


# -----------------------------
# Robust HP state
# -----------------------------
class HPState:
    def __init__(self, history_n: int, majority_k: int, hold_last: bool):
        self.history = deque(maxlen=history_n)  # only valid reads (ints)
        self.majority_k = majority_k
        self.hold_last = hold_last
        self.last_stable: Optional[int] = None
        self.last_raw: Optional[int] = None
        self.missing_count = 0

    def update(self, raw_hp: Optional[int]) -> Tuple[Optional[int], Dict[str, Any]]:
        """
        Returns (stable_hp, debug_info)
        stable_hp is the value you should use for treadmill logic later.
        """
        dbg: Dict[str, Any] = {}

        if raw_hp is None:
            self.missing_count += 1
            dbg["raw"] = None
            dbg["missing_count"] = self.missing_count
            dbg["history"] = list(self.history)

            # HOLD-LAST behavior: keep last stable even when missing
            return (self.last_stable if self.hold_last else None), dbg

        # valid read
        self.missing_count = 0
        self.last_raw = raw_hp
        self.history.append(raw_hp)

        # majority vote
        counts = Counter(self.history)
        value, freq = counts.most_common(1)[0]

        dbg["raw"] = raw_hp
        dbg["history"] = list(self.history)
        dbg["mode"] = value
        dbg["mode_count"] = freq
        dbg["required"] = self.majority_k

        # Only accept stable if it appears >= K times
        if freq >= self.majority_k:
            self.last_stable = value
            dbg["stable_update"] = True
        else:
            dbg["stable_update"] = False

        # If we don't have a stable update yet, still hold last stable if enabled
        if self.last_stable is None:
            return (value if freq >= self.majority_k else None), dbg

        return (self.last_stable if self.hold_last else (value if freq >= self.majority_k else None)), dbg


# -----------------------------
# Main
# -----------------------------
def main():
    template = load_template(TEMPLATE_PATH)
    hp_state = HPState(history_n=HISTORY_N, majority_k=MAJORITY_K, hold_last=HOLD_LAST_ENABLED)

    with mss.mss() as sct:
        while True:
            t0 = time.time()

            win = find_quicktime_window()
            if not win:
                print("QuickTime window not found. Make sure it's visible.")
                time.sleep(1.0)
                continue

            region = {"left": win["x"], "top": win["y"], "width": win["w"], "height": win["h"]}
            img = np.array(sct.grab(region))  # BGRA
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            sidebar, _ = crop_sidebar(frame)

            score, top_left, wh = match_template(sidebar, template)

            raw_hp: Optional[int] = None
            stable_hp: Optional[int] = None

            digit_roi = None
            digit_th = None
            roi_box = None

            if score >= MATCH_MIN:
                digit_roi, roi_box = hp_roi_from_match_box(sidebar, top_left, wh)
                if digit_roi.size > 0:
                    digit_th = preprocess_digits(digit_roi)
                    raw_hp = ocr_digits(digit_th)

                # sanity check
                if raw_hp is not None and not (HP_MIN <= raw_hp <= HP_MAX):
                    raw_hp = None

            stable_hp, dbg = hp_state.update(raw_hp)

            # Console output
            if score < MATCH_MIN:
                print(f"[{win['title']}] match={score:.3f} (no lock) stable_hp={stable_hp}")
            else:
                mode = dbg.get("mode")
                mode_count = dbg.get("mode_count")
                req = dbg.get("required")
                missing = dbg.get("missing_count", 0)
                print(
                    f"[{win['title']}] match={score:.3f} raw_hp={raw_hp} stable_hp={stable_hp} "
                    f"(mode={mode} {mode_count}/{req}, missing={missing})"
                )

            # Debug preview
            if DEBUG_PREVIEW:
                dbg_img = sidebar.copy()

                # Match box (green)
                x, y = top_left
                mw, mh = wh
                x1 = max(0, min(x, dbg_img.shape[1] - 1))
                y1 = max(0, min(y, dbg_img.shape[0] - 1))
                x2 = max(0, min(x + mw, dbg_img.shape[1] - 1))
                y2 = max(0, min(y + mh, dbg_img.shape[0] - 1))
                cv2.rectangle(dbg_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(dbg_img, f"{score:.3f}", (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # HP ROI (blue)
                if roi_box:
                    rx1, ry1, rx2, ry2 = roi_box
                    rx1 = max(0, min(rx1, dbg_img.shape[1] - 1))
                    rx2 = max(0, min(rx2, dbg_img.shape[1] - 1))
                    ry1 = max(0, min(ry1, dbg_img.shape[0] - 1))
                    ry2 = max(0, min(ry2, dbg_img.shape[0] - 1))
                    if rx2 > rx1 and ry2 > ry1:
                        cv2.rectangle(dbg_img, (rx1, ry1), (rx2, ry2), (255, 0, 0), 2)

                # Overlay stable HP on screen
                if stable_hp is not None:
                    cv2.putText(
                        dbg_img,
                        f"STABLE HP: {stable_hp}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2,
                    )
                else:
                    cv2.putText(
                        dbg_img,
                        f"STABLE HP: (none)",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 0, 255),
                        2,
                    )

                cv2.imshow("SIDEBAR", dbg_img)
                show_or_blank("DIGIT_ROI", digit_roi, blank_shape=(120, 260), gray=False)
                show_or_blank("DIGIT_THRESH", digit_th, blank_shape=(120, 260), gray=True)
                cv2.waitKey(1)

            elapsed = time.time() - t0
            time.sleep(max(0.0, SCAN_INTERVAL_SEC - elapsed))


if __name__ == "__main__":
    main()