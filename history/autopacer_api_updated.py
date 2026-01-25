"""
autopacer_api.py

FastAPI "controller" (brains) API for your walk-pad distance → treadmill speed system.

This file ONLY creates the Controller API layer:
- ESP32 posts distance samples to /distance
- You (or UI) can enable/disable autopacing and set tuning parameters
- It exposes controller status for debugging

It does NOT implement the actual control algorithm or talk to your treadmill bridge yet.
Those will live behind a ControlEngine interface stub, so you can plug in logic later.

Run:
  pip install fastapi uvicorn pydantic
  uvicorn autopacer_api:app --host 0.0.0.0 --port 8001

Example:
  curl -X POST http://localhost:8001/distance -H "Content-Type: application/json" \
    -d '{"distance_cm": 102, "valid": true, "sequence": 1}'

  curl -X POST http://localhost:8001/enable
  curl http://localhost:8001/status
"""

from __future__ import annotations

import time
import asyncio
import json
from urllib import request as urllib_request
from urllib.error import URLError, HTTPError
from dataclasses import dataclass, field
from typing import Optional, Literal, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import logging

logger = logging.getLogger("controller")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# -----------------------------
# Static thresholds / knobs (edit these)
# -----------------------------
# Treadmill bridge (DO NOT change server.py; we talk to it here)
TREADMILL_BRIDGE_BASE_URL = "http://127.0.0.1:8765"

# Distance control (sensor measures distance in centimeters)
SETPOINT_CM = 95          # "ideal" distance where speed holds steady
DEADBAND_CM = 10          # +/- band around setpoint where we do nothing
DANGER_CLOSE_CM = 40      # too close -> immediate stop (safety)
DANGER_FAR_CM = 180       # too far   -> immediate stop (safety)

# Speed control (km/h)
BASE_SPEED_KMH = 4.0      # starting speed when enabling autopacer
MIN_SPEED_KMH = 2.0       # clamp lower bound
MAX_SPEED_KMH = 8.0       # clamp upper bound
STEP_KMH = 0.2            # speed step per decision

# Safety / rate limiting
MAX_COMMAND_HZ = 3.0      # don't spam the treadmill with commands
SENSOR_STALE_MS = 1500    # if no valid distance update within this -> fault
FAULT_ACTION = "stop"     # "stop" | "min_speed" | "hold"

# Optional: set to True if you want controller to hit /start when enabling.
AUTO_START_ON_ENABLE = False


# -----------------------------
# Models
# -----------------------------
class DistanceIn(BaseModel):
    distance_cm: int = Field(..., ge=0, le=10000, description="Measured distance in centimeters")
    valid: bool = Field(True, description="Sensor validity flag")
    sequence: Optional[int] = Field(None, ge=0, description="Monotonic sequence number from ESP32")
    ts_ms: Optional[int] = Field(None, ge=0, description="Timestamp in ms from ESP32 (optional)")
    meta: Optional[Dict[str, Any]] = Field(None, description="Optional sensor metadata (signal strength, etc.)")


class EnableIn(BaseModel):
    enabled: bool = Field(..., description="Enable/disable autopacing")


class TuningIn(BaseModel):
    # Minimal but practical knobs; tune to your environment later
    setpoint_cm: int = Field(SETPOINT_CM, ge=10, le=500, description="Ideal distance where speed holds steady")
    deadband_cm: int = Field(DEADBAND_CM, ge=0, le=200, description="No-change band around setpoint")
    danger_close_cm: int = Field(DANGER_CLOSE_CM, ge=0, le=500, description="Too close: emergency slowdown/stop threshold")
    danger_far_cm: int = Field(DANGER_FAR_CM, ge=0, le=2000, description="Too far: emergency stop threshold")

    base_speed_kmh: float = Field(BASE_SPEED_KMH, ge=0.0, le=25.0, description="Base speed around which control operates")
    min_speed_kmh: float = Field(MIN_SPEED_KMH, ge=0.0, le=25.0, description="Minimum allowed speed while running")
    max_speed_kmh: float = Field(MAX_SPEED_KMH, ge=0.0, le=25.0, description="Maximum allowed speed while running")
    step_kmh: float = Field(STEP_KMH, ge=0.0, le=5.0, description="Speed adjustment step per decision")

    max_command_hz: float = Field(MAX_COMMAND_HZ, ge=0.1, le=20.0, description="Max rate the controller should send speed commands")
    sensor_stale_ms: int = Field(SENSOR_STALE_MS, ge=50, le=60000, description="If no valid distance for this long -> fault")
    fault_action: Literal["stop", "min_speed", "hold"] = Field(
        FAULT_ACTION,
        description="What to do if sensor/telemetry is stale or invalid",
    )


class ManualSpeedIn(BaseModel):
    # Useful for testing / future UI; not required for ESP32
    target_speed_kmh: float = Field(..., ge=0.0, le=25.0)


class ControllerStatusOut(BaseModel):
    enabled: bool
    state: Literal["DISABLED", "ARMED", "RUNNING", "FAULT", "MANUAL_OVERRIDE"]

    last_distance_cm: Optional[int]
    last_distance_valid: bool
    last_distance_age_ms: Optional[int]

    tuning: TuningIn

    # Useful debugging fields
    last_decision: Optional[str]
    last_target_speed_kmh: Optional[float]
    updated_at_ms: int


# -----------------------------
# Control "engine" stub
# -----------------------------
@dataclass
class ControllerState:
    enabled: bool = False
    state: str = "DISABLED"  # DISABLED/ARMED/RUNNING/FAULT/MANUAL_OVERRIDE

    tuning: TuningIn = field(default_factory=TuningIn)

    last_distance_cm: Optional[int] = None
    last_distance_valid: bool = False
    last_distance_rx_ms: Optional[int] = None
    last_sequence: Optional[int] = None

    last_decision: Optional[str] = None
    last_target_speed_kmh: Optional[float] = None


class ControlEngine:
    """
    A tiny interface where you'll later plug in:
      - WS subscription to treadmill telemetry (/ws/telemetry on the treadmill-bridge)
      - command emission to treadmill-bridge (/start, /speed, /stop)
      - distance filtering + zone logic + safety rules

    For now it just stores inputs.
    """

    def __init__(self) -> None:
        self._s = ControllerState()

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

    @staticmethod
    def _clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

async def _bridge_post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """POST JSON to treadmill bridge (server.py) using stdlib urllib."""
    url = f"{TREADMILL_BRIDGE_BASE_URL}{path}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib_request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    def _do():
        with urllib_request.urlopen(req, timeout=2.0) as resp:
            return json.loads(resp.read().decode("utf-8") or "{}")

    try:
        return await asyncio.to_thread(_do)
    except HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"bridge HTTP {e.code} on {path}: {body}") from e
    except URLError as e:
        raise RuntimeError(f"bridge unreachable at {TREADMILL_BRIDGE_BASE_URL}: {e}") from e
    except Exception as e:
        raise RuntimeError(f"bridge error on {path}: {e}") from e

async def _bridge_start(self) -> None:
    await self._bridge_post("/start", {})

async def _bridge_stop(self, mode: str = "stop") -> None:
    await self._bridge_post("/stop", {"mode": mode})

async def _bridge_set_speed(self, kmh: float) -> None:
    await self._bridge_post("/speed", {"kmh": kmh})

def _compute_next_speed(self, distance_cm: int, current_kmh: float) -> tuple[float, str]:
    """Distance -> speed rule set.

    Assumption: the sensor is *in front of you*.
      - If you drift closer (smaller distance), treadmill is too slow -> speed up
      - If you drift farther (bigger distance), treadmill is too fast -> slow down
    """
    t = self._s.tuning

    # Safety stops
    if distance_cm <= t.danger_close_cm:
        return 0.0, f"STOP:too_close({distance_cm}cm)"
    if distance_cm >= t.danger_far_cm:
        return 0.0, f"STOP:too_far({distance_cm}cm)"

    # Deadband around setpoint
    low = t.setpoint_cm - t.deadband_cm
    high = t.setpoint_cm + t.deadband_cm

    if distance_cm < low:
        # too close -> speed up
        nxt = current_kmh + t.step_kmh
        nxt = self._clamp(nxt, t.min_speed_kmh, t.max_speed_kmh)
        return nxt, f"UP:{distance_cm}cm<{low}cm -> {nxt:.2f}kmh"
    if distance_cm > high:
        # too far -> slow down
        nxt = current_kmh - t.step_kmh
        nxt = self._clamp(nxt, t.min_speed_kmh, t.max_speed_kmh)
        return nxt, f"DOWN:{distance_cm}cm>{high}cm -> {nxt:.2f}kmh"

    # in band -> hold
    return current_kmh, f"HOLD:{low}cm<= {distance_cm}cm <= {high}cm"

async def control_loop(self) -> None:
    """Background loop that reads the latest distance and emits speed commands."""
    # Controller-local notion of what we last commanded (start at base speed)
    current_speed = float(self._s.tuning.base_speed_kmh)
    last_cmd_ms = 0

    while True:
        await asyncio.sleep(0.05)  # 20 Hz evaluation; output is rate-limited
        if not self._s.enabled:
            continue

        now = self._now_ms()
        t = self._s.tuning

        # No distance yet
        if self._s.last_distance_cm is None:
            self._s.state = "ARMED"
            self._s.last_decision = "armed_waiting_for_distance"
            continue

        age = (now - self._s.last_distance_rx_ms) if self._s.last_distance_rx_ms else None
        if (not self._s.last_distance_valid) or (age is None) or (age > t.sensor_stale_ms):
            self._s.state = "FAULT"
            self._s.last_decision = f"fault:stale_or_invalid(age_ms={age}, valid={self._s.last_distance_valid})"

            # Fault behavior
            if t.fault_action == "stop":
                # Rate limit stop too
                min_interval_ms = int(1000 / max(0.1, t.max_command_hz))
                if now - last_cmd_ms >= min_interval_ms:
                    try:
                        await self._bridge_stop("stop")
                        current_speed = 0.0
                        self._s.last_target_speed_kmh = 0.0
                        last_cmd_ms = now
                    except Exception as e:
                        logger.error("FAULT stop failed: %s", e)
            elif t.fault_action == "min_speed":
                target = float(t.min_speed_kmh)
                min_interval_ms = int(1000 / max(0.1, t.max_command_hz))
                if now - last_cmd_ms >= min_interval_ms and abs(target - current_speed) > 1e-6:
                    try:
                        await self._bridge_set_speed(target)
                        current_speed = target
                        self._s.last_target_speed_kmh = target
                        last_cmd_ms = now
                    except Exception as e:
                        logger.error("FAULT min_speed failed: %s", e)
            # "hold" does nothing
            continue

        # Normal control
        self._s.state = "RUNNING"
        distance_cm = int(self._s.last_distance_cm)

        target_speed, decision = self._compute_next_speed(distance_cm, current_speed)

        # Rate limit commands
        min_interval_ms = int(1000 / max(0.1, t.max_command_hz))
        if now - last_cmd_ms < min_interval_ms:
            self._s.last_decision = f"rate_limited:{decision}"
            continue

        # Safety stop vs speed set
        try:
            if target_speed <= 0.0:
                await self._bridge_stop("stop")
                current_speed = 0.0
                self._s.last_target_speed_kmh = 0.0
            else:
                # If we were previously stopped, optionally start
                if AUTO_START_ON_ENABLE and current_speed <= 0.0:
                    await self._bridge_start()
                # Only send if changed
                if abs(target_speed - current_speed) >= 0.01:
                    await self._bridge_set_speed(float(target_speed))
                    current_speed = float(target_speed)
                    self._s.last_target_speed_kmh = float(target_speed)
                else:
                    # no-op / hold
                    self._s.last_target_speed_kmh = float(current_speed)

            last_cmd_ms = now
            self._s.last_decision = decision
        except Exception as e:
            self._s.state = "FAULT"
            self._s.last_decision = f"fault:bridge_error:{e}"
            logger.error("Bridge command failed: %s", e)

    def set_enabled(self, enabled: bool) -> None:
        self._s.enabled = enabled
        if not enabled:
            self._s.state = "DISABLED"
            self._s.last_decision = "disabled"
        else:
            # When enabled, start in ARMED (waiting for stable presence)
            self._s.state = "ARMED"
            self._s.last_target_speed_kmh = float(self._s.tuning.base_speed_kmh)
            self._s.last_decision = "armed"

    def update_tuning(self, tuning: TuningIn) -> None:
        # Basic sanity checks beyond Pydantic constraints
        if tuning.min_speed_kmh > tuning.max_speed_kmh:
            raise ValueError("min_speed_kmh cannot be > max_speed_kmh")
        if tuning.danger_close_cm >= tuning.danger_far_cm:
            raise ValueError("danger_close_cm must be < danger_far_cm")
        self._s.tuning = tuning
        self._s.last_decision = "tuning_updated"

    def ingest_distance(self, d: DistanceIn) -> None:
        # Optional sequence enforcement (helps with dropped packets)
        if d.sequence is not None and self._s.last_sequence is not None:
            if d.sequence <= self._s.last_sequence:
                # Ignore out-of-order updates
                return

        self._s.last_sequence = d.sequence
        self._s.last_distance_cm = d.distance_cm
        self._s.last_distance_valid = d.valid
        self._s.last_distance_rx_ms = self._now_ms()
        self._s.last_decision = "distance_ingested"

        # NOTE: real control loop would run here or in a background task,
        # computing a target speed and calling treadmill-bridge endpoints.
        # We'll keep it out for now as requested.

    def set_manual_speed(self, target_speed_kmh: float) -> None:
        self._s.state = "MANUAL_OVERRIDE"
        self._s.last_target_speed_kmh = target_speed_kmh
        self._s.last_decision = f"manual_speed_set:{target_speed_kmh}"

    def status(self) -> ControllerStatusOut:
        now = self._now_ms()
        age = None
        if self._s.last_distance_rx_ms is not None:
            age = now - self._s.last_distance_rx_ms

        return ControllerStatusOut(
            enabled=self._s.enabled,
            state=self._s.state,  # type: ignore
            last_distance_cm=self._s.last_distance_cm,
            last_distance_valid=self._s.last_distance_valid,
            last_distance_age_ms=age,
            tuning=self._s.tuning,
            last_decision=self._s.last_decision,
            last_target_speed_kmh=self._s.last_target_speed_kmh,
            updated_at_ms=now,
        )


# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Autopacer Controller API", version="0.1.0")
engine = ControlEngine()


@app.on_event("startup")
async def _startup():
    # Start background control loop
    asyncio.create_task(engine.control_loop())

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/status", response_model=ControllerStatusOut)
def get_status():
    return engine.status()


@app.post("/enable", response_model=ControllerStatusOut)
async def enable():
    engine.set_enabled(True)
    # On enable, optionally start treadmill and set base speed immediately.
    if AUTO_START_ON_ENABLE:
        try:
            await engine._bridge_start()
        except Exception as e:
            raise HTTPException(status_code=409, detail=str(e))
    try:
        await engine._bridge_set_speed(float(engine._s.tuning.base_speed_kmh))
    except Exception as e:
        raise HTTPException(status_code=409, detail=str(e))
    return engine.status()



@app.post("/disable", response_model=ControllerStatusOut)
async def disable():
    engine.set_enabled(False)
    try:
        await engine._bridge_stop("stop")
    except Exception as e:
        # If bridge is down, we still disable locally.
        logger.error("Disable: stop failed: %s", e)
    return engine.status()



@app.post("/enabled", response_model=ControllerStatusOut)
async def set_enabled(body: EnableIn):
    engine.set_enabled(body.enabled)
    if body.enabled:
        if AUTO_START_ON_ENABLE:
            try:
                await engine._bridge_start()
            except Exception as e:
                raise HTTPException(status_code=409, detail=str(e))
        try:
            await engine._bridge_set_speed(float(engine._s.tuning.base_speed_kmh))
        except Exception as e:
            raise HTTPException(status_code=409, detail=str(e))
    else:
        try:
            await engine._bridge_stop("stop")
        except Exception as e:
            logger.error("Disable via /enabled: stop failed: %s", e)
    return engine.status()



@app.post("/tuning", response_model=ControllerStatusOut)
def set_tuning(body: TuningIn):
    try:
        engine.update_tuning(body)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return engine.status()


@app.post("/distance", response_model=ControllerStatusOut)
def post_distance(body: DistanceIn):
    # If controller is disabled, we still accept distance for debugging,
    # but we don't transition to RUNNING or send commands (future behavior).
    engine.ingest_distance(body)
    
    state = engine.status()
    logger.info(
        "DISTANCE rx: %scm valid=%s seq=%s | state=%s target=%s",
        body.distance_cm,
        body.valid,
        body.sequence,
        state.state,
        state.last_target_speed_kmh,
    )
    return engine.status()


@app.post("/manual/speed", response_model=ControllerStatusOut)
def manual_speed(body: ManualSpeedIn):
    # Optional helper endpoint for testing and future UI.
    # Real implementation would forward to treadmill-bridge /speed.
    engine.set_manual_speed(body.target_speed_kmh)
    return engine.status()