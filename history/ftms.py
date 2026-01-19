from dataclasses import dataclass
from typing import Optional

def u16(b: bytes, i: int) -> int:
    return b[i] | (b[i+1] << 8)

def s16(b: bytes, i: int) -> int:
    v = u16(b, i)
    return v - 0x10000 if v & 0x8000 else v

def u24(b: bytes, i: int) -> int:
    return b[i] | (b[i+1] << 8) | (b[i+2] << 16)

@dataclass
class TreadmillTelemetry:
    flags: int
    speed_kmh: Optional[float] = None
    avg_speed_kmh: Optional[float] = None
    total_distance_m: Optional[int] = None
    incline_percent: Optional[float] = None
    ramp_angle_deg: Optional[float] = None
    total_energy_kcal: Optional[int] = None
    energy_per_hour: Optional[int] = None
    energy_per_min: Optional[int] = None
    heart_rate_bpm: Optional[int] = None
    elapsed_time_s: Optional[int] = None
    remaining_time_s: Optional[int] = None
    raw_hex: str = ""

def parse_treadmill_data(payload: bytes) -> TreadmillTelemetry:
    t = TreadmillTelemetry(flags=0, raw_hex=payload.hex())
    if len(payload) < 4:
        return t

    flags = u16(payload, 0)
    t.flags = flags
    idx = 2

    # Instantaneous Speed is part of the treadmill record; commonly 0.01 km/h (uint16).
    # (Spec: instantaneous speed exists as a treadmill data field; data records can be split using "More Data".)  [oai_citation:1‡OneLap](https://www.onelap.cn/pdf/FTMS_v1.0.pdf)
    t.speed_kmh = u16(payload, idx) / 100.0
    idx += 2

    # These bit positions match the common FTMS treadmill layout.
    # If your device disagrees, we’ll adjust based on observed payload lengths/changes.
    MORE_DATA = 0      # record split indicator
    AVG_SPEED_PRESENT = 1
    TOTAL_DISTANCE_PRESENT = 2
    INCLINE_RAMP_PRESENT = 3
    ELEVATION_GAIN_PRESENT = 4  # (not decoded below yet)
    INSTANT_PACE_PRESENT = 5     # (not decoded below yet)
    AVG_PACE_PRESENT = 6         # (not decoded below yet)
    ENERGY_PRESENT = 7
    HR_PRESENT = 8
    MET_PRESENT = 9              # (not decoded below yet)
    ELAPSED_TIME_PRESENT = 10
    REMAINING_TIME_PRESENT = 11

    if flags & (1 << AVG_SPEED_PRESENT):
        t.avg_speed_kmh = u16(payload, idx) / 100.0
        idx += 2

    if flags & (1 << TOTAL_DISTANCE_PRESENT):
        # FTMS commonly uses 24-bit total distance in meters
        t.total_distance_m = u24(payload, idx)
        idx += 3

    if flags & (1 << INCLINE_RAMP_PRESENT):
        # Inclination is typically SINT16 in 0.1% units; Ramp angle SINT16 in 0.1 degrees.
        inc_raw = s16(payload, idx); idx += 2
        ramp_raw = s16(payload, idx); idx += 2
        if inc_raw != 0x7FFF:
            t.incline_percent = inc_raw / 10.0
        if ramp_raw != 0x7FFF:
            t.ramp_angle_deg = ramp_raw / 10.0

    if flags & (1 << ENERGY_PRESENT):
        # Total Energy (uint16, kcal), Energy/hr (uint16), Energy/min (uint8)
        t.total_energy_kcal = u16(payload, idx); idx += 2
        t.energy_per_hour = u16(payload, idx); idx += 2
        t.energy_per_min = payload[idx]; idx += 1

    if flags & (1 << HR_PRESENT):
        t.heart_rate_bpm = payload[idx]
        idx += 1

    if flags & (1 << ELAPSED_TIME_PRESENT):
        t.elapsed_time_s = u16(payload, idx); idx += 2

    if flags & (1 << REMAINING_TIME_PRESENT):
        t.remaining_time_s = u16(payload, idx); idx += 2

    return t