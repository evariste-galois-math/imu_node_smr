from dataclasses import dataclass
from typing import Optional

@dataclass
class AxisCalibration:
    joint: Optional[str]
    gain: Optional[float]
    offset: Optional[float]
    stable: bool
    note: str = ""

@dataclass
class DeviceProfile:
    device_id: int
    yaw: Optional[AxisCalibration] = None
    pitch: Optional[AxisCalibration] = None
    roll: Optional[AxisCalibration] = None

DEVICE_PROFILES = {
    279: DeviceProfile(
        device_id=279,
        pitch=AxisCalibration("joint_4", 0.981, 0.253, stable=True,
                               note="validated against robot ground truth across sessions "
                                    "402-405 (independently re-derived, matches to 3 decimals)"),
        yaw=None,   # confirmed dead, motor-triggered magnetometer corruption, proven via gyro cross-check
        roll=None,  # unconfirmed, weak/confounded correlation (-0.715 vs joint_4), never isolated
    ),
    280: DeviceProfile(
        device_id=280,
        yaw=AxisCalibration("joint_1", 1.087, 5.077, stable=False,
                             note="unstable: gain/offset drift session to session, residual "
                                  "flips sign during motion. Also fails motion-state detection "
                                  "consistently (83-87% vs 98-99% for other devices, all 4 "
                                  "sessions) -- second independent signal, suspect hardware/"
                                  "motor interference, needs physical inspection"),
        pitch=None,  # no meaningful motion observed this session, not tested, not "dead"
        roll=None,   # same, static this session, untested
    ),
    281: DeviceProfile(
        device_id=281,
        yaw=AxisCalibration("joint_2", 0.965, 0.006, stable=True,
                             note="validated against robot ground truth across sessions "
                                  "402-405 (independently re-derived, matches to 3 decimals)"),
        pitch=None,  # showed a correlation hit in the classifier, but confounded by joint 1-4
                     # collinearity and not part of the known mounting map, don't trust this yet
        roll=None,   # static this session, untested
    ),
    282: DeviceProfile(
        device_id=282,
        yaw=AxisCalibration("joint_3", -1.021, 0.128, stable=True,
                             note="mounted inverted, expected; validated against robot ground "
                                  "truth across sessions 402-405 (independently re-derived)"),
        pitch=None,  # same caveat as 281, confounded correlation, not a confirmed mapping
        roll=None,   # static this session, untested
    ),
    283: DeviceProfile(
        device_id=283,
        yaw=None,    # device never in the known mounting map at all, every axis here is
        pitch=None,  # a confounded correlation hit, not a confirmed match. Needs isolated-
        roll=None,   # joint test data before any of these three get calibrated.
    ),
}

def apply_calibration(profile: DeviceProfile, axis: str, raw_value: float) -> float:
    calibration = getattr(profile, axis)

    if calibration is None:
        raise ValueError(
            f"No calibration available for device {profile.device_id}, axis '{axis}'."
        )

    return calibration.gain * raw_value + calibration.offset