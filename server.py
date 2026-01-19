import asyncio
import json
import time

from dataclasses import dataclass
from typing import Optional, Dict, Any, Set

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from bleak import BleakScanner, BleakClient

# ---- FTMS UUIDs ----
TREADMILL_DATA = "00002acd-0000-1000-8000-00805f9b34fb"   # Notify
CONTROL_POINT  = "00002ad9-0000-1000-8000-00805f9b34fb"   # Write + Indicate
SUPPORTED_INCLINE_RANGE = "00002ad5-0000-1000-8000-00805f9b34fb"

# ---- FTMS Control Point opcodes ----
OP_REQUEST_CONTROL   = 0x00
OP_SET_TARGET_SPEED  = 0x02
OP_START_OR_RESUME   = 0x07
OP_STOP_OR_PAUSE     = 0x08
OP_SET_TARGET_INCLINATION = 0x03

# ---- Configure this ----
TREADMILL_NAME = "TM06BK"



def u16(b: bytes, i: int) -> int:
    return b[i] | (b[i+1] << 8)

def s16(b: bytes, i: int) -> int:
    v = u16(b, i)
    return v - 0x10000 if v & 0x8000 else v

def u24(b: bytes, i: int) -> int:
    return b[i] | (b[i+1] << 8) | (b[i+2] << 16)

def u16_le(n: int) -> bytes:
    return bytes([n & 0xFF, (n >> 8) & 0xFF])

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


class BroadcastHub:
    """
    Very small broadcast hub:
    - each websocket gets an asyncio.Queue
    - when we publish(), we put_nowait() into each queue
    """
    def __init__(self):
        self._queues: Set[asyncio.Queue] = set()
        self._lock = asyncio.Lock()

    async def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        async with self._lock:
            self._queues.add(q)
        return q

    async def unsubscribe(self, q: asyncio.Queue):
        async with self._lock:
            self._queues.discard(q)

    async def publish(self, msg: Dict[str, Any]):
        # Fan-out (best-effort). Drop if a client falls behind.
        async with self._lock:
            queues = list(self._queues)

        for q in queues:
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                # Drop newest for that slow client
                pass


class FTMSController:
    def __init__(self, name: str, hub: BroadcastHub):
        self.name = name
        self.hub = hub

        self.client: Optional[BleakClient] = None
        self.device_address: Optional[str] = None

        self.last_cp: Optional[bytes] = None
        self.last_tm: Optional[Telemetry] = None
        self.last_tm_ts: Optional[float] = None
        self.last_cp_ts: Optional[float] = None

        self._lock = asyncio.Lock()
        self._stop_event = asyncio.Event()
        self._runner_task: Optional[asyncio.Task] = None

    # ---- BLE callbacks ----
    def _on_cp(self, _: int, data: bytearray):
        self.last_cp = bytes(data)
        self.last_cp_ts = time.time()

        # publish control point responses too (useful for debugging)
        asyncio.create_task(self.hub.publish({
            "type": "control_point",
            "ts": self.last_cp_ts,
            "hex": self.last_cp.hex(),
        }))

    def _on_tm(self, _: int, data: bytearray):
        self.last_tm = parse_treadmill_data(bytes(data))
        self.last_tm_ts = time.time()

        asyncio.create_task(self.hub.publish({
            "type": "telemetry",
            "ts": self.last_tm_ts,
            "flags": f"0x{self.last_tm.flags:04x}",
            "speed_kmh": self.last_tm.speed_kmh,
            "incline_percentage": self.last_tm.incline_percent,
            "raw_hex": self.last_tm.raw_hex,
        }))

    async def _write_cp(self, payload: bytes):
        if not self.client or not self.client.is_connected:
            raise RuntimeError("Not connected")
        await self.client.write_gatt_char(CONTROL_POINT, payload, response=True)

    async def _connect_once(self) -> bool:
        # Find treadmill (cache address once found)
        if not self.device_address:
            devices = await BleakScanner.discover(timeout=5.0)
            for d in devices:
                if d.name and self.name.lower() == d.name.lower():
                    self.device_address = d.address
                    break

        if not self.device_address:
            return False

        self.client = BleakClient(self.device_address)
        await self.client.connect(timeout=10.0)

        # Subscribe
        await self.client.start_notify(CONTROL_POINT, self._on_cp)
        await self.client.start_notify(TREADMILL_DATA, self._on_tm)

        # Request control
        await self._write_cp(bytes([OP_REQUEST_CONTROL]))

        await self.hub.publish({
            "type": "system",
            "ts": time.time(),
            "event": "connected",
            "name": self.name,
            "address": self.device_address,
        })

        return True

    async def _disconnect(self):
        if self.client:
            try:
                if self.client.is_connected:
                    try:
                        await self.client.stop_notify(TREADMILL_DATA)
                    except Exception:
                        pass
                    try:
                        await self.client.stop_notify(CONTROL_POINT)
                    except Exception:
                        pass
                    await self.client.disconnect()
            finally:
                self.client = None

        await self.hub.publish({
            "type": "system",
            "ts": time.time(),
            "event": "disconnected",
            "name": self.name,
            "address": self.device_address,
        })

    async def runner(self):
        backoff = 1.0
        while not self._stop_event.is_set():
            try:
                ok = await self._connect_once()
                if not ok:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 1.5, 10.0)
                    continue

                backoff = 1.0

                while not self._stop_event.is_set():
                    if not self.client or not self.client.is_connected:
                        break
                    await asyncio.sleep(1.0)

            except Exception:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 1.5, 10.0)
            finally:
                await self._disconnect()

    async def start_background(self):
        if self._runner_task and not self._runner_task.done():
            return
        self._stop_event.clear()
        self._runner_task = asyncio.create_task(self.runner())

    async def stop_background(self):
        self._stop_event.set()
        await asyncio.sleep(0)
        await self._disconnect()

    # ---- Public commands ----
    async def start(self):
        async with self._lock:
            await self._write_cp(bytes([OP_START_OR_RESUME]))
            
    async def set_incline_percent(self, percent: float):
        # Common FTMS encoding: 0.1% units as SINT16 little-endian.
        # Example from the field: 2.0% => 20 => 0x0014 => payload 03 14 00  [oai_citation:3‡NoblePro](https://www.noble-pro.com/third-party-apps/understanding-ftms-bluetooth-advanced/?srsltid=AfmBOoo0PBSl2aFiTmNs78G_IS3D-zWz_089fZUvYVEGUzSAIlsSLqxZ&utm_source=chatgpt.com)
        raw = int(round(percent * 10))
        if raw < -32768 or raw > 32767:
            raise ValueError("incline out of SINT16 range")

        b0 = raw & 0xFF
        b1 = (raw >> 8) & 0xFF
        await self._write_cp(bytes([OP_SET_TARGET_INCLINATION, b0, b1]))

    async def stop(self, mode: str = "stop"):
        param = 0x01 if mode == "stop" else 0x02
        async with self._lock:
            await self._write_cp(bytes([OP_STOP_OR_PAUSE, param]))

    async def set_speed_kmh(self, kmh: float):
        raw = int(round(max(0.0, kmh) * 100))
        async with self._lock:
            await self._write_cp(bytes([OP_SET_TARGET_SPEED]) + u16_le(raw))

    def status(self) -> Dict[str, Any]:
        connected = bool(self.client and self.client.is_connected)
        return {
            "name": self.name,
            "address": self.device_address,
            "connected": connected,
            "last_control_point_hex": self.last_cp.hex() if self.last_cp else None,
            "last_control_point_ts": self.last_cp_ts,
            "telemetry": {
                "flags": f"0x{self.last_tm.flags:04x}" if self.last_tm else None,
                "speed_kmh": self.last_tm.speed_kmh if self.last_tm else None,
                "raw_hex": self.last_tm.raw_hex if self.last_tm else None,
                "ts": self.last_tm_ts,
            },
        }


# ---- FastAPI ----
app = FastAPI(title="Walkpad FTMS API + WebSocket")
hub = BroadcastHub()
ctl = FTMSController(TREADMILL_NAME, hub)

class SpeedIn(BaseModel):
    kmh: float

class StopIn(BaseModel):
    mode: str = "stop"  # "stop" or "pause"

@app.on_event("startup")
async def on_startup():
    await ctl.start_background()

@app.on_event("shutdown")
async def on_shutdown():
    await ctl.stop_background()

@app.get("/status")
async def get_status():
    return ctl.status()

@app.post("/start")
async def post_start():
    try:
        await ctl.start()
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=409, detail=str(e))

@app.post("/stop")
async def post_stop(body: StopIn):
    if body.mode not in ("stop", "pause"):
        raise HTTPException(status_code=400, detail="mode must be 'stop' or 'pause'")
    try:
        await ctl.stop(mode=body.mode)
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=409, detail=str(e))

@app.post("/speed")
async def post_speed(body: SpeedIn):
    if body.kmh < 0 or body.kmh > 20:
        raise HTTPException(status_code=400, detail="kmh out of reasonable range")
    try:
        await ctl.set_speed_kmh(body.kmh)
        return {"ok": True, "kmh": body.kmh}
    except Exception as e:
        raise HTTPException(status_code=409, detail=str(e))

class InclineIn(BaseModel):
    percent: float

@app.post("/incline")
async def post_incline(body: InclineIn):
    # keep sane bounds; you can tighten after reading supported range
    if body.percent < -10 or body.percent > 30:
        raise HTTPException(status_code=400, detail="percent out of reasonable range")
    try:
        await ctl.set_incline_percent(body.percent)
        return {"ok": True, "percent": body.percent}
    except Exception as e:
        raise HTTPException(status_code=409, detail=str(e))

@app.websocket("/ws/telemetry")
async def ws_telemetry(ws: WebSocket):
    await ws.accept()
    q = await hub.subscribe()

    # send a snapshot immediately
    await ws.send_text(json.dumps({
        "type": "snapshot",
        "ts": time.time(),
        "status": ctl.status(),
    }))

    try:
        while True:
            msg = await q.get()
            await ws.send_text(json.dumps(msg))
    except WebSocketDisconnect:
        pass
    finally:
        await hub.unsubscribe(q)