"""
Robot-free calibration.

Uses ONLY the IMU's own data. No robot joint data is read, fit against, or
used to choose any number in this file.

Two corrections per device:

1. OFFSET (tare): the average reading during the first genuine rest period,
   found automatically with motion-state detection (gyro magnitude). After
   this, the device's home pose reads 0.

2. GAIN (gyro consistency): during the first motion between two rest
   periods, compare how far the fused angle moved against how far the
   gyro's own integrated rotation moved. The gyro is an independent
   measurement from the same chip, so if the fused angle overshoots or
   undershoots the gyro, that ratio corrects it.

   Safety checks (a gain that fails either is rejected, gain stays 1.0):
     - the gyro axis must track the fused angle almost perfectly
       (|correlation| >= 0.999). If the sensor is tilted relative to the
       joint, the rotation splits across gyro axes and one axis no longer
       represents it.
     - the gain must be within 10% of 1.0. A real sensor scale error that
       large is not plausible; a ratio that far off means the axis match is
       wrong, not that the sensor is wrong.

Only physical mounting information is configured by hand: which fused angle
tracks the device's joint, and whether the IMU is mounted upside down.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class MountingConfig:
    axis: str   # "yaw", "pitch" or "roll"
    sign: int   # -1 if the IMU is mounted upside down, else 1


MOUNTING = {
    279: MountingConfig(axis="pitch", sign=1),
    280: MountingConfig(axis="yaw", sign=1),
    281: MountingConfig(axis="yaw", sign=1),
    282: MountingConfig(axis="yaw", sign=-1),
}


@dataclass
class DeviceCalibration:
    device_id: int
    axis: str
    sign: int
    offset: float
    gain: float
    gain_source: str


def gyro_magnitude(gyro_x: np.ndarray, gyro_y: np.ndarray, gyro_z: np.ndarray) -> np.ndarray:
    return np.sqrt(gyro_x ** 2 + gyro_y ** 2 + gyro_z ** 2)


def detect_motion(gyro_mag: np.ndarray, threshold: float = 0.02, min_state_run: int = 5) -> np.ndarray:
    """Same logic as motion_state.py, applied to a whole recording at once."""
    raw_moving = gyro_mag > threshold
    state = raw_moving.copy()
    run_start = 0
    for i in range(1, len(state)):
        if state[i] != state[i - 1]:
            if i - run_start < min_state_run:
                if run_start > 0:
                    state[run_start:i] = state[run_start - 1]
                else:
                    state[run_start:i] = state[i]
            run_start = i
    return state


def find_rest_periods(moving: np.ndarray, min_length: int = 50) -> list:
    """Still periods at least min_length samples long, as (start, end) pairs."""
    periods = []
    start = None
    for i, is_moving in enumerate(moving):
        if not is_moving and start is None:
            start = i
        elif is_moving and start is not None:
            if i - start >= min_length:
                periods.append((start, i))
            start = None
    if start is not None:
        if len(moving) - start >= min_length:
            periods.append((start, len(moving)))
    return periods


def estimate_offset(angle: np.ndarray, rest_periods: list) -> Optional[float]:
    if not rest_periods:
        return None
    start, end = rest_periods[0]
    return float(np.mean(angle[start:end]))


def estimate_gyro_gain(
    angle: np.ndarray,
    gyro_xyz: dict,
    dt_seconds: np.ndarray,
    rest_periods: list,
    min_correlation: float = 0.999,
    max_deviation: float = 0.10,
) -> tuple:
    """Returns (gain, reason). gain is 1.0 whenever a safety check fails."""
    if len(rest_periods) < 2:
        return 1.0, "not enough rest periods to isolate a motion"

    motion_start = rest_periods[0][1]
    motion_end = rest_periods[1][0]
    fused = angle[motion_start:motion_end] - angle[motion_start]
    fused_peak = fused[np.argmax(np.abs(fused))]
    if abs(fused_peak) < 1.0:
        return 1.0, "motion too small to measure"

    best_axis = None
    best_corr = 0.0
    best_peak = 0.0
    for axis_name, rates in gyro_xyz.items():
        rotation = np.cumsum(np.degrees(rates[motion_start:motion_end]) * dt_seconds[motion_start:motion_end])
        corr = np.corrcoef(rotation, fused)[0, 1]
        if abs(corr) > abs(best_corr):
            best_axis = axis_name
            best_corr = corr
            best_peak = rotation[np.argmax(np.abs(rotation))]

    if abs(best_corr) < min_correlation:
        return 1.0, f"rejected: best gyro axis {best_axis} correlation {best_corr:.4f} too low"

    gain = abs(best_peak) / abs(fused_peak)
    if abs(gain - 1.0) > max_deviation:
        return 1.0, f"rejected: gain {gain:.3f} from {best_axis} outside plausible range"

    return float(gain), f"gyro {best_axis}, correlation {best_corr:.4f}"


def calibrate_device(
    device_id: int,
    angles: dict,
    gyro_xyz: dict,
    capture_time_ms: np.ndarray,
) -> DeviceCalibration:
    """
    angles: {"yaw": array, "pitch": array, "roll": array} from the IMU
    gyro_xyz: {"gyro_x": array, "gyro_y": array, "gyro_z": array} in rad/s
    capture_time_ms: IMU timestamps
    """
    mounting = MOUNTING[device_id]
    angle = mounting.sign * np.asarray(angles[mounting.axis], dtype=float)

    gyro_mag = gyro_magnitude(gyro_xyz["gyro_x"], gyro_xyz["gyro_y"], gyro_xyz["gyro_z"])
    moving = detect_motion(gyro_mag)
    rest_periods = find_rest_periods(moving)

    offset = estimate_offset(angle, rest_periods)
    if offset is None:
        offset = 0.0

    dt_seconds = np.diff(capture_time_ms, prepend=capture_time_ms[0]) / 1000.0
    gain, reason = estimate_gyro_gain(angle, gyro_xyz, dt_seconds, rest_periods)

    return DeviceCalibration(
        device_id=device_id,
        axis=mounting.axis,
        sign=mounting.sign,
        offset=offset,
        gain=gain,
        gain_source=reason,
    )


def apply_calibration(calibration: DeviceCalibration, raw_value: float) -> float:
    signed = calibration.sign * raw_value
    return calibration.gain * (signed - calibration.offset)