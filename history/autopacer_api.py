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
    setpoint_cm: int = Field(95, ge=10, le=500, description="Ideal distance where speed holds steady")
    deadband_cm: int = Field(10, ge=0, le=200, description="No-change band around setpoint")
    danger_close_cm: int = Field(40, ge=0, le=500, description="Too close: emergency slowdown/stop threshold")
    danger_far_cm: int = Field(180, ge=0, le=2000, description="Too far: emergency stop threshold")

    base_speed_kmh: float = Field(4.0, ge=0.0, le=25.0, description="Base speed around which control operates")
    min_speed_kmh: float = Field(2.0, ge=0.0, le=25.0, description="Minimum allowed speed while running")
    max_speed_kmh: float = Field(8.0, ge=0.0, le=25.0, description="Maximum allowed speed while running")
    step_kmh: float = Field(0.2, ge=0.0, le=5.0, description="Speed adjustment step per decision")

    max_command_hz: float = Field(3.0, ge=0.1, le=20.0, description="Max rate the controller should send speed commands")
    sensor_stale_ms: int = Field(1500, ge=50, le=60000, description="If no valid distance for this long -> fault")
    fault_action: Literal["stop", "min_speed", "hold"] = Field(
        "stop",
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

    def set_enabled(self, enabled: bool) -> None:
        self._s.enabled = enabled
        if not enabled:
            self._s.state = "DISABLED"
            self._s.last_decision = "disabled"
        else:
            # When enabled, start in ARMED (waiting for stable presence)
            self._s.state = "ARMED"
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


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/status", response_model=ControllerStatusOut)
def get_status():
    return engine.status()


@app.post("/enable", response_model=ControllerStatusOut)
def enable():
    engine.set_enabled(True)
    return engine.status()


@app.post("/disable", response_model=ControllerStatusOut)
def disable():
    engine.set_enabled(False)
    return engine.status()


@app.post("/enabled", response_model=ControllerStatusOut)
def set_enabled(body: EnableIn):
    engine.set_enabled(body.enabled)
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