from typing import Optional

import numpy as np


def estimate_resting_value(
    calibrated_history: list,
    is_moving_history: list,
    min_run_for_rest: int = 50,
) -> Optional[float]:
    """
    Finds the first genuinely long still run in the history and returns the
    calibrated signal's own average during it. Returns None if no rest period
    long enough has been seen yet. Uses IMU-derived values only.
    """
    run_start = None
    for i, moving in enumerate(is_moving_history):
        if not moving and run_start is None:
            run_start = i
        elif moving and run_start is not None:
            if i - run_start >= min_run_for_rest:
                return float(np.mean(calibrated_history[run_start:i]))
            run_start = None
    if run_start is not None:
        if len(is_moving_history) - run_start >= min_run_for_rest:
            return float(np.mean(calibrated_history[run_start:]))
    return None


class DriftCorrector:
    """
    Runs on top of calibration. During confirmed long still periods, nudges
    the calibrated value back toward the device's resting value.

    A still period only counts once it has lasted min_still_run samples. A
    brief still flag (for example the gyro passing through zero while the
    joint reverses direction) is not real rest, and correcting during it
    causes large error spikes. min_still_run=50 was chosen from the IMU data:
    a direction-reversal blip lasted about 30 samples, real rest periods run
    into the hundreds.

    Tested on sessions 402-405 on top of robot-free calibration (robot data
    used only to score results):
      - device 279: ~0% change
      - device 280: 14% better than raw on average (vs 11% from calibration
        alone). Known issue: 280's readings freeze in flat steps while the
        joint is still moving, and the gyro reads zero during those freezes,
        so a long freeze can be mistaken for rest and cause a short dip.
      - device 281: 39% better than raw
      - device 282: 21% better than raw (vs 18% from calibration alone)
    """

    ENABLED_DEVICES = {279, 280, 281, 282}

    def __init__(self, device_id: int, correction_rate: float = 0.1, min_still_run: int = 50):
        self.device_id = device_id
        self.correction_rate = correction_rate
        self.min_still_run = min_still_run
        self._still_run_length = 0
        self._last_value: Optional[float] = None
        self._resting_value: Optional[float] = None
        self._calibrated_history: list = []
        self._is_moving_history: list = []

    @property
    def enabled(self) -> bool:
        return self.device_id in self.ENABLED_DEVICES

    def step(self, calibrated_value: float, is_moving: bool) -> float:
        if not self.enabled:
            return calibrated_value

        self._calibrated_history.append(calibrated_value)
        self._is_moving_history.append(is_moving)

        if is_moving:
            self._still_run_length = 0
            self._last_value = calibrated_value
            return calibrated_value

        self._still_run_length += 1

        if self._resting_value is None:
            self._resting_value = estimate_resting_value(
                self._calibrated_history,
                self._is_moving_history,
                self.min_still_run,
            )

        if self._still_run_length < self.min_still_run:
            self._last_value = calibrated_value
            return calibrated_value

        if self._resting_value is None:
            self._last_value = calibrated_value
            return calibrated_value

        corrected = self._last_value + self.correction_rate * (self._resting_value - self._last_value)
        self._last_value = corrected
        return corrected