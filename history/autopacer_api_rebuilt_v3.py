"""
autopacer_api.py (controller)

- Your distance provider posts to POST /distance (already working).
- This controller adjusts treadmill speed by calling the treadmill bridge server (server.py)
  at http://127.0.0.1:8765 via POST /speed {"kmh": <float>}.

IMPORTANT SAFETY/BEHAVIOR CHANGES (per your feedback):
- This controller does NOT call /start or /stop. It only sets speed.
- It does NOT spam the treadmill: it only changes speed after N consecutive
  "too close" or "too far" readings, and it rate-limits commands.

Core logic:
- Compute zone relative to SETPOINT_CM:
    * TOO_CLOSE: distance < (SETPOINT_CM - DEADBAND_CM)
    * TOO_FAR:   distance > (SETPOINT_CM + DEADBAND_CM)
    * NEUTRAL:   inside deadband
- Require REPEAT_COUNT consecutive samples in TOO_CLOSE/TOO_FAR before adjusting speed.
- Any NEUTRAL sample resets counters (so a single "move back" cancels the pending change).
- After a speed change, counters reset and we wait for another REPEAT_COUNT confirmations
  before changing again.
"""

from __future__ import annotations

import asyncio
from collections import deque
import logging
import time
from dataclasses import dataclass
from typing import Literal, Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ============================================================
# TUNING KNOBS (edit these)
# ============================================================

# Target distance from sensor (cm)
SETPOINT_CM: float = 25.0

# Neutral zone +/- around SETPOINT_CM (cm)
DEADBAND_CM: float = 10.0

# Speed bounds (km/h)
MIN_SPEED_KMH: float = 0.8
BASE_SPEED_KMH: float = 2.2
MAX_SPEED_KMH: float = 6.0

# Speed change per adjustment (km/h)
STEP_KMH: float = 0.2

# Require this many consecutive TOO_CLOSE or TOO_FAR samples before adjusting speed
REPEAT_COUNT: int = 5

# Distance samples arrive about every 500ms; this should be >= that.
CONTROL_LOOP_HZ: float = 4.0  # decisions per second (2.0–6.0 is reasonable)

# Hard rate limit: never send commands faster than this
MAX_COMMAND_HZ: float = 0.5   # 0.5 => at most 1 command every 2 seconds

# Consider sensor stale if we haven't received a sample in this long
SENSOR_STALE_MS: int = 2000

# On stale/invalid data: "hold" (do nothing) or "min_speed" (set to MIN_SPEED_KMH)
FAULT_ACTION: Literal["hold", "min_speed"] = "hold"

# Treadmill bridge base URL (server.py)
TREADMILL_BRIDGE_URL: str = "http://127.0.0.1:8765"

# Timeout for bridge calls
BRIDGE_TIMEOUT_S: float = 1.5



# Distance logging + debug buffer
# - Logs every N distance posts so you can troubleshoot sensor readings without spamming.
DISTANCE_LOG_EVERY: int = 1  # 1 = log every reading; 5 = log every 5th reading
DISTANCE_BUFFER_SIZE: int = 300  # how many recent readings to keep for /debug/distances

# ============================================================
# Logging
# ============================================================

logger = logging.getLogger("autopacer_controller")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ============================================================
# FastAPI models
# ============================================================


class DistanceIn(BaseModel):
    distance_cm: float = Field(..., ge=0, description="Distance from sensor in cm")
    valid: bool = Field(True, description="Whether the reading is valid")
    sequence: Optional[int] = Field(None, description="Optional monotonically increasing sequence number")


class EnableIn(BaseModel):
    enabled: bool = True


class ManualSpeedIn(BaseModel):
    target_speed_kmh: float = Field(..., ge=0, le=20)


class ControllerStatusOut(BaseModel):
    enabled: bool
    mode: Literal["DISABLED", "AUTO", "MANUAL", "FAULT"]
    last_distance_cm: Optional[float] = None
    last_distance_valid: Optional[bool] = None
    last_distance_age_ms: Optional[int] = None
    zone: Literal["NEUTRAL", "TOO_CLOSE", "TOO_FAR", "INVALID", "STALE"] = "STALE"
    close_streak: int
    far_streak: int
    repeat_count: int
    target_speed_kmh: Optional[float] = None
    last_sent_speed_kmh: Optional[float] = None
    last_decision: str
    updated_at_ms: int


# ============================================================
# Controller engine
# ============================================================


def now_ms() -> int:
    return int(time.time() * 1000)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def classify_zone(distance_cm: float) -> Literal["NEUTRAL", "TOO_CLOSE", "TOO_FAR"]:
    if distance_cm < (SETPOINT_CM - DEADBAND_CM):
        return "TOO_CLOSE"
    if distance_cm > (SETPOINT_CM + DEADBAND_CM):
        return "TOO_FAR"
    return "NEUTRAL"


@dataclass
class SensorState:
    last_sequence: Optional[int] = None
    last_distance_cm: Optional[float] = None
    last_valid: Optional[bool] = None
    last_rx_ms: Optional[int] = None


@dataclass
class EngineState:
    enabled: bool = False
    mode: Literal["DISABLED", "AUTO", "MANUAL", "FAULT"] = "DISABLED"

    # streak counters
    close_streak: int = 0
    far_streak: int = 0
    zone: Literal["NEUTRAL", "TOO_CLOSE", "TOO_FAR", "INVALID", "STALE"] = "STALE"

    # speed tracking
    target_speed_kmh: Optional[float] = None
    last_sent_speed_kmh: Optional[float] = None
    last_command_ms: int = 0

    last_decision: str = "init"


class ControlEngine:
    def __init__(self, bridge_url: str) -> None:
        self.bridge_url = bridge_url.rstrip("/")
        self.sensor = SensorState()
        self.state = EngineState()
        self._lock = asyncio.Lock()
        self._task: Optional[asyncio.Task] = None
        self._client = httpx.AsyncClient(timeout=BRIDGE_TIMEOUT_S)
        self._distance_buf = deque(maxlen=DISTANCE_BUFFER_SIZE)
        self._distance_log_counter = 0

    # --------------- public API -----------------

    async def start(self) -> None:
        async with self._lock:
            self.state.enabled = True
            self.state.mode = "AUTO"
            if self.state.target_speed_kmh is None:
                self.state.target_speed_kmh = BASE_SPEED_KMH
            self.state.last_decision = "enabled:auto"
            if self._task is None or self._task.done():
                self._task = asyncio.create_task(self.control_loop())

    async def stop(self) -> None:
        async with self._lock:
            self.state.enabled = False
            self.state.mode = "DISABLED"
            self.state.last_decision = "disabled"
            self.state.close_streak = 0
            self.state.far_streak = 0
            self.state.zone = "STALE"

    async def set_manual_speed(self, speed_kmh: float) -> None:
        async with self._lock:
            self.state.enabled = True
            self.state.mode = "MANUAL"
            self.state.target_speed_kmh = clamp(speed_kmh, MIN_SPEED_KMH, MAX_SPEED_KMH)
            self.state.last_decision = f"manual:set_target:{self.state.target_speed_kmh:.2f}"
        await self._maybe_send_speed(self.state.target_speed_kmh)

    async def ingest_distance(self, distance_cm: float, valid: bool, sequence: Optional[int]) -> None:
        rx = now_ms()
        async with self._lock:
            if sequence is not None and self.sensor.last_sequence is not None and sequence < self.sensor.last_sequence:
                # out-of-order; ignore
                return
            self.sensor.last_sequence = sequence
            self.sensor.last_distance_cm = distance_cm
            self.sensor.last_valid = valid
            self.sensor.last_rx_ms = rx


        # Keep recent samples for troubleshooting
        self._distance_buf.append(
            {
                "rx_ms": rx,
                "sequence": sequence,
                "distance_cm": distance_cm,
                "valid": valid,
            }
        )

        # Optional distance logging (every N reads)
        self._distance_log_counter += 1
        if DISTANCE_LOG_EVERY > 0 and (self._distance_log_counter % DISTANCE_LOG_EVERY == 0):
            logger.info(
                "distance rx: %.1f cm (valid=%s, seq=%s)",
                distance_cm,
                valid,
                sequence,
            )
    async def get_status(self) -> ControllerStatusOut:
        async with self._lock:
            age = None
            if self.sensor.last_rx_ms is not None:
                age = now_ms() - self.sensor.last_rx_ms
            return ControllerStatusOut(
                enabled=self.state.enabled,
                mode=self.state.mode,
                last_distance_cm=self.sensor.last_distance_cm,
                last_distance_valid=self.sensor.last_valid,
                last_distance_age_ms=age,
                zone=self.state.zone,
                close_streak=self.state.close_streak,
                far_streak=self.state.far_streak,
                repeat_count=REPEAT_COUNT,
                target_speed_kmh=self.state.target_speed_kmh,
                last_sent_speed_kmh=self.state.last_sent_speed_kmh,
                last_decision=self.state.last_decision,
                updated_at_ms=now_ms(),
            )



    async def get_recent_distances(self, limit: int = 50):
        async with self._lock:
            if limit <= 0:
                return []
            items = list(self._distance_buf)[-limit:]
            return items

    # --------------- control loop -----------------

    async def control_loop(self) -> None:
        sleep_s = max(0.05, 1.0 / max(0.1, CONTROL_LOOP_HZ))
        while True:
            await asyncio.sleep(sleep_s)
            await self._tick()

    async def _tick(self) -> None:
        async with self._lock:
            if not self.state.enabled or self.state.mode == "DISABLED":
                return

            # manual mode: do nothing in the loop (manual endpoint sends speed)
            if self.state.mode == "MANUAL":
                return

            # AUTO mode
            zone = self._compute_zone_locked()
            self.state.zone = zone

            if zone in ("INVALID", "STALE"):
                self._handle_fault_locked(zone)
                return

            # neutral cancels pending changes
            if zone == "NEUTRAL":
                self.state.close_streak = 0
                self.state.far_streak = 0
                self.state.last_decision = "neutral:hold"
                return

            # update streak counters
            if zone == "TOO_CLOSE":
                self.state.close_streak += 1
                self.state.far_streak = 0
                self.state.last_decision = f"too_close:streak:{self.state.close_streak}/{REPEAT_COUNT}"
                if self.state.close_streak < REPEAT_COUNT:
                    return
                # confirmed too close => speed up
                self._apply_speed_step_locked(delta=+STEP_KMH, reason="confirmed_too_close")
                return

            if zone == "TOO_FAR":
                self.state.far_streak += 1
                self.state.close_streak = 0
                self.state.last_decision = f"too_far:streak:{self.state.far_streak}/{REPEAT_COUNT}"
                if self.state.far_streak < REPEAT_COUNT:
                    return
                # confirmed too far => slow down
                self._apply_speed_step_locked(delta=-STEP_KMH, reason="confirmed_too_far")
                return

    def _compute_zone_locked(self) -> Literal["NEUTRAL", "TOO_CLOSE", "TOO_FAR", "INVALID", "STALE"]:
        if self.sensor.last_rx_ms is None:
            return "STALE"
        age = now_ms() - self.sensor.last_rx_ms
        if age > SENSOR_STALE_MS:
            return "STALE"
        if not self.sensor.last_valid:
            return "INVALID"
        if self.sensor.last_distance_cm is None:
            return "STALE"
        return classify_zone(self.sensor.last_distance_cm)

    def _handle_fault_locked(self, kind: Literal["INVALID", "STALE"]) -> None:
        # No /stop calls allowed. We either hold speed, or gently drop to MIN.
        if kind == "STALE":
            self.state.last_decision = f"fault:stale>{SENSOR_STALE_MS}ms"
        else:
            self.state.last_decision = "fault:invalid_reading"

        self.state.close_streak = 0
        self.state.far_streak = 0

        if FAULT_ACTION == "min_speed":
            target = MIN_SPEED_KMH
            self.state.target_speed_kmh = target
            # Send outside the lock via task
            asyncio.create_task(self._maybe_send_speed(target))

    def _apply_speed_step_locked(self, delta: float, reason: str) -> None:
        # Reset counters so we require another REPEAT_COUNT confirmations
        self.state.close_streak = 0
        self.state.far_streak = 0

        cur = self.state.target_speed_kmh if self.state.target_speed_kmh is not None else BASE_SPEED_KMH
        nxt = clamp(cur + delta, MIN_SPEED_KMH, MAX_SPEED_KMH)
        self.state.target_speed_kmh = nxt
        self.state.last_decision = f"{reason}:target:{nxt:.2f}"

        asyncio.create_task(self._maybe_send_speed(nxt))

    async def _maybe_send_speed(self, speed_kmh: float) -> None:
        # rate limit + only send when it actually changes
        async with self._lock:
            min_interval_ms = int(1000 / max(0.01, MAX_COMMAND_HZ))
            since_last = now_ms() - self.state.last_command_ms
            if since_last < min_interval_ms:
                return

            last_sent = self.state.last_sent_speed_kmh
            if last_sent is not None and abs(speed_kmh - last_sent) < 1e-6:
                return

            self.state.last_command_ms = now_ms()

        try:
            url = f"{self.bridge_url}/speed"
            payload = {"kmh": float(speed_kmh)}
            async with self._client as client:
                r = await client.post(url, json=payload)
                if r.status_code >= 400:
                    raise HTTPException(status_code=r.status_code, detail=r.text)

            async with self._lock:
                self.state.last_sent_speed_kmh = float(speed_kmh)
        except Exception as e:
            async with self._lock:
                self.state.last_decision = f"bridge_error:{type(e).__name__}"
            logger.warning("Failed to send speed to treadmill bridge: %s", e)


# ============================================================
# FastAPI app wiring
# ============================================================

app = FastAPI(title="Autopacer Controller", version="2.0")

engine = ControlEngine(TREADMILL_BRIDGE_URL)


@app.on_event("startup")
async def _startup() -> None:
    # Do not auto-enable; you explicitly enable via /enable.
    logger.info("Controller started. Bridge=%s", TREADMILL_BRIDGE_URL)


@app.on_event("shutdown")
async def _shutdown() -> None:
    try:
        await engine.stop()
    finally:
        logger.info("Controller shutting down.")


@app.post("/distance")
async def post_distance(d: DistanceIn):
    await engine.ingest_distance(distance_cm=d.distance_cm, valid=d.valid, sequence=d.sequence)
    return {"ok": True}


@app.post("/enable")
async def post_enable(body: EnableIn = EnableIn()):
    if body.enabled:
        await engine.start()
    else:
        await engine.stop()
    return {"ok": True, "enabled": body.enabled}


@app.post("/manual_speed")
async def post_manual_speed(body: ManualSpeedIn):
    await engine.set_manual_speed(body.target_speed_kmh)
    return {"ok": True, "mode": "MANUAL", "target_speed_kmh": body.target_speed_kmh}


@app.get("/status", response_model=ControllerStatusOut)
async def get_status():
    return await engine.get_status()



@app.get("/status", response_model=ControllerStatusOut)
async def get_status():
    return await engine.get_status()


@app.get("/debug/distances")
async def get_debug_distances(limit: int = 50):
    # Guardrails to avoid huge payloads
    if limit > 500:
        limit = 500
    if limit < 1:
        limit = 1
    samples = await engine.get_recent_distances(limit=limit)
    return {"count": len(samples), "samples": samples}

