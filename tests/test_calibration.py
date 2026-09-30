"""
Tests for robot-free calibration. All test data is synthetic IMU data:
a rest period, one motion, then another rest period.
"""

import numpy as np
import pytest

from imu.calibration import (
    DeviceCalibration,
    MOUNTING,
    apply_calibration,
    calibrate_device,
    detect_motion,
    estimate_gyro_gain,
    find_rest_periods,
    gyro_magnitude,
)

SAMPLE_MS = 10.0


def make_session(peak_deg=30.0, fused_scale=1.0, start_offset=0.0, gyro_axis="gyro_y",
                 rest=300, move=400):
    """
    Synthetic recording: rest, rotate up to peak_deg and back down, rest.
    The gyro reports the true rotation rate on gyro_axis (rad/s). The fused
    angle is the true angle times fused_scale plus start_offset, to simulate
    a sensor that over- or under-reads and doesn't start at zero.
    """
    half = move // 2
    up = np.linspace(0.0, peak_deg, half)
    down = np.linspace(peak_deg, 0.0, move - half)
    true_angle = np.concatenate([np.zeros(rest), up, down, np.zeros(rest)])

    dt = SAMPLE_MS / 1000.0
    rate_deg_s = np.diff(true_angle, prepend=true_angle[0]) / dt
    rate_rad_s = np.radians(rate_deg_s)

    gyro = {"gyro_x": np.zeros_like(true_angle), "gyro_y": np.zeros_like(true_angle), "gyro_z": np.zeros_like(true_angle)}
    gyro[gyro_axis] = rate_rad_s

    fused = true_angle * fused_scale + start_offset
    times = np.arange(len(true_angle)) * SAMPLE_MS
    return true_angle, fused, gyro, times


def test_detects_rest_and_motion():
    _, _, gyro, _ = make_session()
    moving = detect_motion(gyro_magnitude(gyro["gyro_x"], gyro["gyro_y"], gyro["gyro_z"]))
    periods = find_rest_periods(moving)
    assert len(periods) == 2
    assert periods[0][0] == 0


def test_offset_removes_starting_bias():
    true_angle, fused, gyro, times = make_session(start_offset=4.0)
    cal = calibrate_device(281, {"yaw": fused, "pitch": fused, "roll": fused}, gyro, times)
    assert cal.offset == pytest.approx(4.0, abs=1e-6)
    assert apply_calibration(cal, 4.0) == pytest.approx(0.0, abs=1e-6)


def test_gain_corrects_overreading_sensor():
    true_angle, fused, gyro, times = make_session(fused_scale=1.05)
    cal = calibrate_device(281, {"yaw": fused, "pitch": fused, "roll": fused}, gyro, times)
    assert cal.gain == pytest.approx(1 / 1.05, rel=0.01)
    corrected = np.array([apply_calibration(cal, v) for v in fused])
    assert np.max(np.abs(corrected - true_angle)) < 0.5


def test_implausible_gain_is_rejected():
    _, fused, gyro, times = make_session(fused_scale=2.5)
    cal = calibrate_device(281, {"yaw": fused, "pitch": fused, "roll": fused}, gyro, times)
    assert cal.gain == 1.0
    assert "rejected" in cal.gain_source


def test_split_rotation_is_rejected():
    """Rotation spread across two gyro axes (tilted sensor): no single axis
    matches, so the gain must not be trusted."""
    true_angle, fused, gyro, times = make_session()
    noise = np.random.default_rng(0).normal(0.0, 0.05, len(true_angle))
    gyro["gyro_x"] = gyro["gyro_y"] * 0.5 + noise
    gyro["gyro_y"] = gyro["gyro_y"] * 0.5 + noise
    moving = detect_motion(gyro_magnitude(gyro["gyro_x"], gyro["gyro_y"], gyro["gyro_z"]))
    periods = find_rest_periods(moving)
    dt = np.diff(times, prepend=times[0]) / 1000.0
    gain, reason = estimate_gyro_gain(fused, gyro, dt, periods)
    assert gain == 1.0


def test_inverted_mounting_flips_sign():
    true_angle, fused, gyro, times = make_session()
    cal = calibrate_device(282, {"yaw": -fused, "pitch": fused, "roll": fused}, gyro, times)
    assert cal.sign == -1
    peak_index = int(np.argmax(true_angle))
    assert apply_calibration(cal, -fused[peak_index]) == pytest.approx(true_angle[peak_index], abs=0.5)


def test_no_rest_period_falls_back_safely():
    fused = np.linspace(0.0, 30.0, 200)
    gyro = {"gyro_x": np.zeros(200), "gyro_y": np.full(200, 0.5), "gyro_z": np.zeros(200)}
    times = np.arange(200) * SAMPLE_MS
    cal = calibrate_device(281, {"yaw": fused, "pitch": fused, "roll": fused}, gyro, times)
    assert cal.offset == 0.0
    assert cal.gain == 1.0


def test_mounting_covers_known_devices():
    assert set(MOUNTING) == {279, 280, 281, 282}


def test_calibration_module_does_not_use_robot_data():
    import imu.calibration as module
    source = open(module.__file__).read().lower()
    assert "joint_" not in source
    assert "robot session" not in source