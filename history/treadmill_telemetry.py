from dataclasses import dataclass

@dataclass
class TreadmillTelemetry:
    flags: int
    speed_kmh: float | None = None

def u16_le(b: bytes, i: int) -> int:
    return b[i] | (b[i+1] << 8)

def parse_treadmill_data(payload: bytes) -> TreadmillTelemetry:
    """
    FTMS Treadmill Data (0x2ACD) is flag-driven. Many devices follow the spec,
    some are 'mostly' spec. We'll start with the safe basics: flags + speed.
    """
    if len(payload) < 4:
        return TreadmillTelemetry(flags=0)

    flags = u16_le(payload, 0)
    idx = 2

    # In FTMS, instantaneous speed is typically present unless a flag says otherwise.
    # Many devices encode speed in 0.01 km/h units (uint16).
    speed_raw = u16_le(payload, idx)
    idx += 2
    speed_kmh = speed_raw / 100.0

    return TreadmillTelemetry(flags=flags, speed_kmh=speed_kmh)