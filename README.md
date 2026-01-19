# TFT → Treadmill Controller 🏃‍♂️🎮

This project connects **Teamfight Tactics gameplay** to a **walking treadmill** in real time.

Your TFT performance directly affects treadmill difficulty:
- Doing well → slower speed / lower incline
- Losing → faster speed / higher incline

The system **does not interact with game memory** (anti-cheat safe).
Instead, it:
1. Mirrors an iPad running TFT via **QuickTime**
2. Uses **computer vision + OCR** to read your HP from the screen
3. Smooths noisy readings for safety
4. Sends **rate-limited commands** to a local treadmill API

---

## High-Level Architecture

```
iPad (TFT)
   ↓  (USB screen mirror)
QuickTime Player (macOS)
   ↓  (screen capture)
Python (OpenCV + OCR)
   ↓  (smoothed HP signal)
Local Treadmill API
   ↓
Walkpad / Treadmill
```

---

## Why This Exists

- TFT has no public API
- Anti-cheat prevents memory inspection
- Player ordering changes dynamically
- HP UI disappears on some screens
- Player portrait appearance changes as HP drops

Design choices reflect these constraints:
- Template matching instead of pure OCR
- Edge-based matching to tolerate UI changes
- HP hold-last behavior when UI is hidden
- Majority-vote smoothing to prevent OCR spikes

This intentionally favors **stability and safety** over instant reaction.

---

## Features

### Robust HP Detection
- Template matching on your TFT row
- Digit-only OCR for HP
- Edge-based matching for late-game stability
- Lock + hysteresis to avoid losing your row

### Signal Smoothing
- Sliding window of recent HP reads
- Requires majority agreement before updating
- Prevents treadmill spikes from bad OCR frames

### Treadmill Safety
- Speed and incline clamps
- Rate-limited commands
- Gradual ramping
- Safe fallback behavior when HP is unknown

### Debug UI
- Live sidebar preview
- Match box (green)
- HP ROI box (blue)
- Bottom stats panel with:
  - Raw HP
  - Stable HP
  - Vote state
  - Treadmill state
  - Target values
  - History

---

## Requirements

### Hardware
- macOS
- iPad running Teamfight Tactics
- USB cable (iPad → Mac)
- Walking treadmill / walkpad

### Software
- Python 3.9+
- QuickTime Player
- Tesseract OCR

### Python Dependencies
```
pip install opencv-python numpy pytesseract mss requests pyobjc-framework-Quartz
```

---

## macOS Permissions

You **must** enable Screen Recording:
```
System Settings → Privacy & Security → Screen Recording
→ enable Terminal / Python
```

---

## Running

1. Mirror your iPad to QuickTime (Movie Recording)
2. Ensure the TFT sidebar is visible
3. Start your treadmill API locally
4. Run:
```
python tft_treadmill_sidebar_panel_v2.py
```

---

## Safety Notes ⚠️

- Start with conservative max speed/incline
- Keep a hand near the treadmill controls
- Use the physical emergency stop if needed
- This is an experimental project — use at your own risk

---

## Disclaimer

This project is for **personal experimentation** only.
It is not affiliated with Riot Games.
No game memory or network traffic is inspected.
