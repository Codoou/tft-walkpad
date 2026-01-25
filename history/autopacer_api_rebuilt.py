"""
autopacer_api.py

FastAPI "controller" (brains) API for your walk-pad distance → treadmill speed system.

- ESP32 (or your distance provider) posts distance samples to POST /distance
- This controller runs a background control loop that:
    * reads the latest distance sample
    * decides a target treadmill speed
    * sends commands to the treadmill bridge server (server.py) at 127.0.0.1:8765

You said:
- Do NOT modify server.py (treadmill bridge)
- /distance ingestion is already configured and working
- Update THIS file to interact with server.py and update speeds based on distance
- Make distance threshold variables static at the top so you can tweak them easily
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ============================================================
# TUNING KNOBS (edit these)
# ============================================================

# Where you want to "hover" relative to the sensor, in cm.
SETPOINT_CM: float = 100.0

# No speed change inside this +/- band around SETPOINT_CM.
DEADBAND_CM: float = 8.0

# Hard safety stops (in cm). If distance is outside this range -> STOP.
DANGER_CLOSE_CM: float = 45.0
DANGER_FAR_CM: float = 220.0

# Speed limits (km/h)
MIN_SPEED_KMH: float = 0.8
BASE_SPEED_KMH: float = 2.5
MAX_SPEED_KMH: float = 6.0

# How much to change speed per adjustment step (km/h)
STEP_KMH: float = 0.2

# Control loop frequency
LOOP_HZ: float = 6.0  # decisions per second

# Don't spam the treadmill: maximum command rate
MAX_COMMAND_HZ: float = 2.0

# If we haven't received a distance sample in this long, we treat it as stale and fault
SENSOR_STALE_MS: int = 1000

# What to do on fault/stale/invalid data: "stop" (recommended) or "hold"
FAULT_ACTION: Literal["stop", "hold"] = "stop"

# If True: enabling the controller will POST /start to treadmill bridge automatically
AUTO_START_ON_ENABLE: bool = True

# Treadmill bridge base URL (server.py)
TREADMILL_BRIDGE_URL: str = "http://127.0.0.1:8765"

# ============================================================
# Logging
# ============================================================

logger = logging.getLogger("autopacer_controller")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ============================================================
# API Models
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
    state: Literal["DISABLED", "ARMED", "RUNNING", "MANUAL_OVERRIDE", "FAULT"]
    last_distance_cm: Optional[float] = None
    last_distance_valid: Optional[bool] = None
    last_distance_age_ms: Optional[int] = None
    last_decision: str
    last_target_speed_kmh: Optional[float] = None
    last_sent_speed_kmh: Optional[float] = None
    treadmill_connected: Optional[bool] = None
    updated_at_ms: int


# ============================================================
# Controller State + Engine
# ============================================================


@dataclass
class ControllerState:
    enabled: bool = False
    state: Literal["DISABLED", "ARMED", "RUNNING", "MANUAL_OVERRIDE", "FAULT"] = "DISABLED"

    last_sequence: Optional[int] = None
    last_distance_cm: Optional[float] = None
    last_distance_valid: Optional[bool] = None
    last_distance_rx_ms: Optional[int] = None

    last_decision: str = "boot"
    last_target_speed_kmh: Optional[float] = None
    last_sent_speed_kmh: Optional[float] = None

    treadmill_connected: Optional[bool] = None

    # task lifecycle
    loop_task: Optional[asyncio.Task] = None


class ControlEngine:
    def __init__(self) -> None:
        self._s = ControllerState()
        self._client: Optional[httpx.AsyncClient] = None
        self._shutdown = asyncio.Event()

        # command pacing
        self._last_command_ts: float = 0.0

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

    async def start(self) -> None:
        # Async http client for treadmill bridge
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=2.0)

        # Start background loop once
        if self._s.loop_task is None or self._s.loop_task.done():
            self._shutdown.clear()
            self._s.loop_task = asyncio.create_task(self.control_loop())

    async def stop(self) -> None:
        self._shutdown.set()
        if self._s.loop_task is not None:
            try:
                await asyncio.wait_for(self._s.loop_task, timeout=2.0)
            except asyncio.TimeoutError:
                self._s.loop_task.cancel()
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ----------------------------
    # Public controls
    # ----------------------------

    async def set_enabled(self, enabled: bool) -> None:
        self._s.enabled = enabled
        if not enabled:
            self._s.state = "DISABLED"
            self._s.last_decision = "disabled"
            if FAULT_ACTION == "stop":
                await self._treadmill_stop()
            return

        # enabled
        self._s.state = "ARMED"
        self._s.last_decision = "armed"
        if AUTO_START_ON_ENABLE:
            await self._treadmill_start()
            await self._treadmill_speed(BASE_SPEED_KMH)

    def ingest_distance(self, d: DistanceIn) -> None:
        # Optional sequence enforcement (ignore out-of-order)
        if d.sequence is not None and self._s.last_sequence is not None:
            if d.sequence <= self._s.last_sequence:
                return

        self._s.last_sequence = d.sequence
        self._s.last_distance_cm = d.distance_cm
        self._s.last_distance_valid = d.valid
        self._s.last_distance_rx_ms = self._now_ms()

    async def set_manual_speed(self, target_speed_kmh: float) -> None:
        self._s.state = "MANUAL_OVERRIDE"
        self._s.last_decision = f"manual_speed_set:{target_speed_kmh:.2f}"
        await self._treadmill_start()
        await self._treadmill_speed(target_speed_kmh)

    # ----------------------------
    # Loop
    # ----------------------------

    async def control_loop(self) -> None:
        """Runs forever until shutdown. Reads last distance and commands treadmill bridge."""
        assert self._client is not None, "engine.start() must be called before control_loop()"

        sleep_s = 1.0 / max(1.0, LOOP_HZ)

        while not self._shutdown.is_set():
            try:
                await self._tick()
            except Exception as e:
                # never crash the loop
                self._s.state = "FAULT"
                self._s.last_decision = f"loop_error:{type(e).__name__}"
                logger.exception("Control loop error: %s", e)
                if FAULT_ACTION == "stop":
                    await self._treadmill_stop()
            await asyncio.sleep(sleep_s)

    async def _tick(self) -> None:
        # If not enabled, do nothing
        if not self._s.enabled:
            return

        now_ms = self._now_ms()

        # Validate distance freshness + validity
        if self._s.last_distance_rx_ms is None:
            await self._fault("no_distance_yet")
            return

        age_ms = now_ms - self._s.last_distance_rx_ms
        if age_ms > SENSOR_STALE_MS:
            await self._fault(f"distance_stale:{age_ms}ms")
            return

        if not self._s.last_distance_valid:
            await self._fault("distance_invalid")
            return

        d = float(self._s.last_distance_cm or 0.0)

        # Hard safety window
        if d <= DANGER_CLOSE_CM:
            await self._fault(f"danger_close:{d:.1f}cm")
            return
        if d >= DANGER_FAR_CM:
            await self._fault(f"danger_far:{d:.1f}cm")
            return

        # Normal running
        if self._s.state in ("ARMED", "FAULT"):
            self._s.state = "RUNNING"

        target = self._compute_target_speed_kmh(d)
        self._s.last_target_speed_kmh = target
        self._s.last_decision = f"target:{target:.2f}kmh d={d:.1f}cm"

        await self._maybe_send_speed(target)

    def _compute_target_speed_kmh(self, distance_cm: float) -> float:
        """
        Simple rule:
        - If you're closer than setpoint: speed up
        - If you're farther than setpoint: slow down
        - Deadband around setpoint: hold speed
        """
        last = self._s.last_sent_speed_kmh
        current = last if last is not None else BASE_SPEED_KMH

        error = SETPOINT_CM - distance_cm  # positive means too close (need to go faster)
        if abs(error) <= DEADBAND_CM:
            return self._clamp(current)

        if error > 0:
            # too close -> speed up
            return self._clamp(current + STEP_KMH)

        # too far -> slow down
        return self._clamp(current - STEP_KMH)

    @staticmethod
    def _clamp(v: float) -> float:
        return max(MIN_SPEED_KMH, min(MAX_SPEED_KMH, v))

    async def _fault(self, reason: str) -> None:
        self._s.state = "FAULT"
        self._s.last_decision = f"fault:{reason}"
        if FAULT_ACTION == "stop":
            await self._treadmill_stop()

    # ----------------------------
    # Treadmill bridge calls
    # ----------------------------

    async def _maybe_send_speed(self, kmh: float) -> None:
        # command pacing (MAX_COMMAND_HZ)
        now = time.time()
        min_dt = 1.0 / max(0.1, MAX_COMMAND_HZ)
        if (now - self._last_command_ts) < min_dt:
            return

        # Avoid re-sending nearly identical speeds
        if self._s.last_sent_speed_kmh is not None:
            if abs(kmh - self._s.last_sent_speed_kmh) < (STEP_KMH * 0.5):
                return

        await self._treadmill_start()
        ok = await self._treadmill_speed(kmh)
        if ok:
            self._s.last_sent_speed_kmh = kmh
            self._last_command_ts = now

    async def _treadmill_start(self) -> bool:
        assert self._client is not None
        try:
            r = await self._client.post(f"{TREADMILL_BRIDGE_URL}/start")
            r.raise_for_status()
            self._s.treadmill_connected = True
            return True
        except Exception:
            self._s.treadmill_connected = False
            return False

    async def _treadmill_stop(self) -> bool:
        assert self._client is not None
        try:
            r = await self._client.post(f"{TREADMILL_BRIDGE_URL}/stop", json={"mode": "stop"})
            r.raise_for_status()
            self._s.treadmill_connected = True
            return True
        except Exception:
            self._s.treadmill_connected = False
            return False

    async def _treadmill_speed(self, kmh: float) -> bool:
        assert self._client is not None
        try:
            r = await self._client.post(f"{TREADMILL_BRIDGE_URL}/speed", json={"kmh": float(kmh)})
            r.raise_for_status()
            self._s.treadmill_connected = True
            return True
        except Exception:
            self._s.treadmill_connected = False
            return False

    # ----------------------------
    # Status
    # ----------------------------

    def status(self) -> ControllerStatusOut:
        now = self._now_ms()
        age = None
        if self._s.last_distance_rx_ms is not None:
            age = now - self._s.last_distance_rx_ms

        return ControllerStatusOut(
            enabled=self._s.enabled,
            state=self._s.state,
            last_distance_cm=self._s.last_distance_cm,
            last_distance_valid=self._s.last_distance_valid,
            last_distance_age_ms=age,
            last_decision=self._s.last_decision,
            last_target_speed_kmh=self._s.last_target_speed_kmh,
            last_sent_speed_kmh=self._s.last_sent_speed_kmh,
            treadmill_connected=self._s.treadmill_connected,
            updated_at_ms=now,
        )


# ============================================================
# FastAPI app
# ============================================================

app = FastAPI(title="Autopacer Controller API", version="0.2.0")
engine = ControlEngine()


@app.on_event("startup")
async def _startup() -> None:
    await engine.start()


@app.on_event("shutdown")
async def _shutdown() -> None:
    await engine.stop()


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/status", response_model=ControllerStatusOut)
def get_status() -> ControllerStatusOut:
    return engine.status()


@app.post("/enable", response_model=ControllerStatusOut)
async def enable() -> ControllerStatusOut:
    await engine.set_enabled(True)
    return engine.status()


@app.post("/disable", response_model=ControllerStatusOut)
async def disable() -> ControllerStatusOut:
    await engine.set_enabled(False)
    return engine.status()


@app.post("/enabled", response_model=ControllerStatusOut)
async def set_enabled(body: EnableIn) -> ControllerStatusOut:
    await engine.set_enabled(body.enabled)
    return engine.status()


@app.post("/distance", response_model=ControllerStatusOut)
def post_distance(body: DistanceIn) -> ControllerStatusOut:
    engine.ingest_distance(body)

    st = engine.status()
    logger.info(
        "DISTANCE rx: %.1fcm valid=%s seq=%s | state=%s target=%s sent=%s",
        body.distance_cm,
        body.valid,
        body.sequence,
        st.state,
        st.last_target_speed_kmh,
        st.last_sent_speed_kmh,
    )
    return st


@app.post("/manual/speed", response_model=ControllerStatusOut)
async def manual_speed(body: ManualSpeedIn) -> ControllerStatusOut:
    await engine.set_manual_speed(body.target_speed_kmh)
    return engine.status()
