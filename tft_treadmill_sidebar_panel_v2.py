#!/usr/bin/env python3
"""
TFT -> Walkpad controller (single file)
- Captures QuickTime window
- Crops TFT right sidebar
- Template-matches YOUR row (template matches whole row block)
- OCRs HP from inside the matched box (your tuned bounds)
- Robustness:
    - HOLD-LAST stable HP when UI hides HP
    - MAJORITY-VOTE over last N valid reads (10) requiring K occurrences (6)
- Treadmill integration ENABLED:
    - Calls your local API:
        POST /speed   {"kmh": <float>}
        POST /incline {"percent": <float>}
    - Safety:
        - clamps to max/min speed + incline
        - rate limits commands
        - ramps changes (limits per command)
        - only acts on STABLE HP (not raw HP)

Install:
    brew install tesseract
    python3 -m pip install mss opencv-python numpy pytesseract pyobjc-framework-Quartz requests

Run:
    python3 tft_walkpad_driver.py

macOS permission:
- System Settings -> Privacy & Security -> Screen Recording -> enable Terminal / Python

IMPORTANT SAFETY NOTE:
- Keep a hand on your treadmill controls / emergency stop.
- Start with conservative MAX_SPEED/INCLINE.
"""

import time
from collections import deque, Counter
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import cv2
import numpy as np
import pytesseract
import mss
import requests

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
DEBUG_PANEL_HEIGHT_PX = 300  # extra space below sidebar for readable stats

# Sidebar crop relative to QuickTime frame (your tuned values)
SIDEBAR_SX = 0.85
SIDEBAR_SY = 0.09
SIDEBAR_SW = 0.15
SIDEBAR_SH = 0.50

# Template image (matches your whole row block)
TEMPLATE_PATH = "portrait_template.png"
MATCH_MIN = 0.5

# Lock hysteresis for template matching
MATCH_MIN_LOCKED = 0.55   # once locked, allow slightly lower scores
LOCK_SEARCH_RADIUS = 80   # px search radius around last lock
MATCH_USE_EDGES = True    # edge-based matching is less sensitive to color/outline changes

# HP bounds inside match box (YOUR tuned bounds)
HP_RX1 = 0.00
HP_RX2 = 0.45
HP_RY1 = 0.63
HP_RY2 = 0.95

HP_MIN = 0
HP_MAX = 100

# Majority vote filter
HISTORY_N = 10
MAJORITY_K = 6

# Hold-last behavior
HOLD_LAST_ENABLED = True

# -----------------------------
# Treadmill API
# -----------------------------
API = "http://127.0.0.1:8765"
ENABLE_TREADMILL = False

# Command cadence
COMMAND_INTERVAL_SEC = 5.0  # don't spam control point
REQUEST_TIMEOUT_SEC = 1.0

# Clamp + ramp limits (start conservative)
MIN_SPEED_KMH = 0.6
MAX_SPEED_KMH = 2.4
MIN_INCLINE_PCT = 0.0
MAX_INCLINE_PCT = 8.0

MAX_SPEED_STEP_PER_CMD = 0.2     # km/h per command
MAX_INCLINE_STEP_PER_CMD = 0.5   # percent per command

# If we lose stable HP entirely (no stable yet), what do we do?
# "hold" = keep last targets; "safe" = drift toward these safe values
ON_UNKNOWN_HP = "hold"  # "hold" or "safe"
SAFE_SPEED_KMH = 0.8
SAFE_INCLINE_PCT = 0.0


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
    """Return (score, top_left, (w,h)).

    If MATCH_USE_EDGES is True, match on Canny edges to be less sensitive to
    UI color/styling changes (e.g. portrait ring disappearing).
    """
    sb = cv2.cvtColor(sidebar_bgr, cv2.COLOR_BGR2GRAY)
    tp = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)

    sb = cv2.GaussianBlur(sb, (3, 3), 0)
    tp = cv2.GaussianBlur(tp, (3, 3), 0)

    if MATCH_USE_EDGES:
        sb = cv2.Canny(sb, 60, 160)
        tp = cv2.Canny(tp, 60, 160)

    res = cv2.matchTemplate(sb, tp, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)

    h, w = tp.shape[:2]
    return float(max_val), (int(max_loc[0]), int(max_loc[1])), (w, h)


def match_template_local(sidebar_bgr: np.ndarray, template_bgr: np.ndarray, approx_top_left: Tuple[int, int], radius: int) -> Tuple[float, Tuple[int, int], Tuple[int, int]]:
    """Search for the template in a local window around approx_top_left."""
    H, W = sidebar_bgr.shape[:2]
    ax, ay = approx_top_left

    # Expand window by radius in each direction
    x1 = max(0, ax - radius)
    y1 = max(0, ay - radius)
    x2 = min(W, ax + radius + template_bgr.shape[1])
    y2 = min(H, ay + radius + template_bgr.shape[0])

    roi = sidebar_bgr[y1:y2, x1:x2]
    score, loc, wh = match_template(roi, template_bgr)
    return score, (loc[0] + x1, loc[1] + y1), wh


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


def hp_roi_from_match_box(sidebar_bgr: np.ndarray, top_left: Tuple[int, int], wh: Tuple[int, int]) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    x, y = top_left
    mw, mh = wh

    rx1 = x + int(HP_RX1 * mw)
    rx2 = x + int(HP_RX2 * mw)
    ry1 = y + int(HP_RY1 * mh)
    ry2 = y + int(HP_RY2 * mh)

    roi = safe_crop(sidebar_bgr, rx1, ry1, rx2, ry2)
    return roi, (rx1, ry1, rx2, ry2)


def show_or_blank(win_name: str, img: Optional[np.ndarray], blank_shape=(120, 260), gray=False) -> None:
    if img is None or img.size == 0:
        if gray:
            blank = np.zeros(blank_shape, dtype=np.uint8)
        else:
            blank = np.zeros((blank_shape[0], blank_shape[1], 3), dtype=np.uint8)
        cv2.imshow(win_name, blank)
    else:
        cv2.imshow(win_name, img)


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
        dbg: Dict[str, Any] = {}

        if raw_hp is None:
            self.missing_count += 1
            dbg["raw"] = None
            dbg["missing_count"] = self.missing_count
            dbg["history"] = list(self.history)
            return (self.last_stable if self.hold_last else None), dbg

        self.missing_count = 0
        self.last_raw = raw_hp
        self.history.append(raw_hp)

        counts = Counter(self.history)
        value, freq = counts.most_common(1)[0]

        dbg["raw"] = raw_hp
        dbg["history"] = list(self.history)
        dbg["mode"] = value
        dbg["mode_count"] = freq
        dbg["required"] = self.majority_k

        if freq >= self.majority_k:
            self.last_stable = value
            dbg["stable_update"] = True
        else:
            dbg["stable_update"] = False

        if self.last_stable is None:
            return (value if freq >= self.majority_k else None), dbg

        return (self.last_stable if self.hold_last else (value if freq >= self.majority_k else None)), dbg


# -----------------------------
# Treadmill control (safe ramping)
# -----------------------------
def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def step_toward(current: float, target: float, max_step: float) -> float:
    if target > current:
        return min(target, current + max_step)
    if target < current:
        return max(target, current - max_step)
    return current


def map_hp_to_targets(hp: int) -> Tuple[float, float]:
    """
    Better doing -> lower speed/incline
    Losing -> higher speed/incline

    Adjust these buckets however you want.
    """
    if hp >= 80:
        return 0.8, 0.0
    if hp >= 60:
        return 1.2, 2.0
    if hp >= 40:
        return 1.6, 4.0
    if hp >= 20:
        return 2.0, 6.0
    return 2.4, 8.0


@dataclass
class TreadmillState:
    speed_kmh: float = SAFE_SPEED_KMH
    incline_pct: float = SAFE_INCLINE_PCT
    last_cmd_ts: float = 0.0


class TreadmillController:
    def __init__(self, api_base: str):
        self.api_base = api_base.rstrip("/")
        self.state = TreadmillState()

    def maybe_send(self, target_speed: float, target_incline: float) -> Optional[Tuple[float, float]]:
        now = time.time()
        if now - self.state.last_cmd_ts < COMMAND_INTERVAL_SEC:
            return None

        # clamp targets
        target_speed = clamp(target_speed, MIN_SPEED_KMH, MAX_SPEED_KMH)
        target_incline = clamp(target_incline, MIN_INCLINE_PCT, MAX_INCLINE_PCT)

        # ramp toward target
        new_speed = step_toward(self.state.speed_kmh, target_speed, MAX_SPEED_STEP_PER_CMD)
        new_incline = step_toward(self.state.incline_pct, target_incline, MAX_INCLINE_STEP_PER_CMD)

        # if no meaningful change, skip
        if abs(new_speed - self.state.speed_kmh) < 1e-6 and abs(new_incline - self.state.incline_pct) < 1e-6:
            self.state.last_cmd_ts = now
            return None

        if ENABLE_TREADMILL:
            try:
                requests.post(f"{self.api_base}/speed", json={"kmh": float(new_speed)}, timeout=REQUEST_TIMEOUT_SEC)
                requests.post(f"{self.api_base}/incline", json={"percent": float(new_incline)}, timeout=REQUEST_TIMEOUT_SEC)
            except Exception as e:
                print(f"[TREADMILL] ERROR sending commands: {e}")
                return None

        self.state.speed_kmh = new_speed
        self.state.incline_pct = new_incline
        self.state.last_cmd_ts = now
        return new_speed, new_incline



def _wrap_text_to_width(text: str, max_width_px: int, font, scale: float, thickness: int) -> list[str]:
    """Greedy word wrap for cv2.putText."""
    words = text.split(" ")
    lines: list[str] = []
    cur = ""

    for w in words:
        trial = (cur + " " + w).strip()
        (tw, _), _ = cv2.getTextSize(trial, font, scale, thickness)
        if tw <= max_width_px or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w

    if cur:
        lines.append(cur)
    return lines


def render_sidebar_with_panel(sidebar_bgr: np.ndarray, panel_lines: list[str]) -> np.ndarray:
    """Return an image = sidebar stacked on top of a dark stats panel.

    The panel is NARROW (same width as the sidebar crop), so we wrap text to fit.
    """
    h, w = sidebar_bgr.shape[:2]
    out = np.zeros((h + DEBUG_PANEL_HEIGHT_PX, w, 3), dtype=np.uint8)
    out[0:h, 0:w] = sidebar_bgr

    # Panel background
    y0 = h
    out[y0:, :] = (18, 18, 18)

    # Border line
    cv2.line(out, (0, y0), (w - 1, y0), (60, 60, 60), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.72
    thickness = 2
    x_pad = 12
    y = y0 + 36
    line_h = 30
    max_text_w = w - 2 * x_pad

    def wrap_to_width(text: str) -> list[str]:
        # Wrap by spaces, and if a token is still too long, hard-truncate it.
        words = text.split(' ')
        lines: list[str] = []
        cur = ''
        for word in words:
            trial = word if not cur else (cur + ' ' + word)
            tw = cv2.getTextSize(trial, font, font_scale, thickness)[0][0]
            if tw <= max_text_w:
                cur = trial
                continue
            if cur:
                lines.append(cur)
                cur = word
            else:
                # single word too wide -> truncate
                truncated = word
                while truncated and cv2.getTextSize(truncated + '…', font, font_scale, thickness)[0][0] > max_text_w:
                    truncated = truncated[:-1]
                lines.append((truncated + '…') if truncated else '…')
                cur = ''
        if cur:
            lines.append(cur)
        return lines

    # Render wrapped lines until we run out of vertical space
    max_lines = max(1, int((DEBUG_PANEL_HEIGHT_PX - 20) / line_h))
    rendered = 0
    for raw in panel_lines:
        for wrapped in wrap_to_width(str(raw)):
            if rendered >= max_lines:
                break
            cv2.putText(out, wrapped, (x_pad, y + rendered * line_h), font, font_scale, (240, 240, 240), thickness, cv2.LINE_AA)
            rendered += 1
        if rendered >= max_lines:
            break

    return out


# -----------------------------
# Main
# -----------------------------
def main():
    template = load_template(TEMPLATE_PATH)
    hp_state = HPState(history_n=HISTORY_N, majority_k=MAJORITY_K, hold_last=HOLD_LAST_ENABLED)
    treadmill = TreadmillController(API)

    # Template lock state (helps when match score dips due to UI changes)
    lock_top_left: Optional[Tuple[int, int]] = None

    last_target_speed = SAFE_SPEED_KMH
    last_target_incline = SAFE_INCLINE_PCT

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

            # Prefer a local search around our last lock to avoid bouncing
            if lock_top_left is not None:
                score, top_left, wh = match_template_local(sidebar, template, lock_top_left, LOCK_SEARCH_RADIUS)
                if score < MATCH_MIN_LOCKED:
                    # fall back to full search if local lock isn't good enough
                    score, top_left, wh = match_template(sidebar, template)
                    if score >= MATCH_MIN:
                        lock_top_left = top_left
                    else:
                        lock_top_left = None
                else:
                    lock_top_left = top_left
            else:
                score, top_left, wh = match_template(sidebar, template)
                if score >= MATCH_MIN:
                    lock_top_left = top_left

            raw_hp: Optional[int] = None
            digit_roi = None
            digit_th = None
            roi_box = None
            match_ok = (score >= MATCH_MIN) or (lock_top_left is not None and score >= MATCH_MIN_LOCKED)

            if match_ok:
                digit_roi, roi_box = hp_roi_from_match_box(sidebar, top_left, wh)
                if digit_roi.size > 0:
                    digit_th = preprocess_digits(digit_roi)
                    raw_hp = ocr_digits(digit_th)

                if raw_hp is not None and not (HP_MIN <= raw_hp <= HP_MAX):
                    raw_hp = None

            stable_hp, dbg = hp_state.update(raw_hp)

            # Decide targets
            if stable_hp is None:
                if ON_UNKNOWN_HP == "safe":
                    target_speed, target_incline = SAFE_SPEED_KMH, SAFE_INCLINE_PCT
                else:
                    target_speed, target_incline = last_target_speed, last_target_incline
            else:
                target_speed, target_incline = map_hp_to_targets(stable_hp)
                last_target_speed, last_target_incline = target_speed, target_incline

            # Apply treadmill ramped commands
            sent = treadmill.maybe_send(target_speed, target_incline)

            # Console output
            if score < MATCH_MIN:
                print(f"[{win['title']}] match={score:.3f} (no lock) stable_hp={stable_hp} -> target={target_speed:.1f}kmh/{target_incline:.1f}% sent={sent}")
            else:
                mode = dbg.get("mode")
                mode_count = dbg.get("mode_count")
                req = dbg.get("required")
                missing = dbg.get("missing_count", 0)
                print(
                    f"[{win['title']}] match={score:.3f} raw_hp={raw_hp} stable_hp={stable_hp} "
                    f"(mode={mode} {mode_count}/{req}, missing={missing}) "
                    f"-> target={target_speed:.1f}kmh/{target_incline:.1f}% sent={sent}"
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

                # Build stats panel text (big + readable)
                mode = dbg.get("mode")
                mode_count = dbg.get("mode_count")
                req = dbg.get("required")
                missing = dbg.get("missing_count", 0)
                hist = dbg.get("history", [])
                hist_tail = hist[-10:] if isinstance(hist, list) else []

                stable_str = "None" if stable_hp is None else str(stable_hp)
                raw_str = "None" if raw_hp is None else str(raw_hp)

                lines = [
                    f"HP: stable={stable_str} raw={raw_str}",
                    f"Match: {score:.3f}  (min={MATCH_MIN:.2f} locked_min={MATCH_MIN_LOCKED:.2f})",
                    f"Vote: mode={mode} {mode_count}/{req}  missing={missing}",
                    f"TM now: {treadmill.state.speed_kmh:.1f} km/h  {treadmill.state.incline_pct:.1f}%",
                    f"Target: {target_speed:.1f} km/h  {target_incline:.1f}%  sent={sent}",
                    f"History({len(hist_tail)}): {hist_tail}",
                ]

                sidebar_view = render_sidebar_with_panel(dbg_img, lines)

                cv2.imshow("SIDEBAR", sidebar_view)
                show_or_blank("DIGIT_ROI", digit_roi, blank_shape=(120, 260), gray=False)
                show_or_blank("DIGIT_THRESH", digit_th, blank_shape=(120, 260), gray=True)
                cv2.waitKey(1)

            elapsed = time.time() - t0
            time.sleep(max(0.0, SCAN_INTERVAL_SEC - elapsed))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting (KeyboardInterrupt).")
        # I’m NOT auto-stopping the treadmill here because I don’t know your API’s stop semantics.
        # Use the treadmill’s physical stop controls if needed.
        pass