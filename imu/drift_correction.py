from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class DriftCorrectionResult:
    resting_value: float
    corrected_value: float


def estimate_resting_value(calibrated_history: list[float], is_moving_history: list[bool],
                            min_run_for_rest: int = 50) -> Optional[float]:
    """
    Finds the first genuinely long still run in recent history and returns
    the calibrated signal's own average during that run as the resting value.
    Returns None if no long-enough still run has been seen yet.
    """
    run_start = None
    for i, moving in enumerate(is_moving_history):
        if not moving and run_start is None:
            run_start = i
        elif moving and run_start is not None:
            if i - run_start >= min_run_for_rest:
                return float(np.mean(calibrated_history[run_start:i]))
            run_start = None
    if run_start is not None and len(is_moving_history) - run_start >= min_run_for_rest:
        return float(np.mean(calibrated_history[run_start:]))
    return None


class DriftCorrector:
    """
    Nudges a calibrated value back toward its resting value during CONFIRMED
    long still periods only.

    IMPORTANT: a brief "still" flag (e.g. gyro passing through zero during a
    direction reversal, mid-motion) is NOT the same as genuinely being at
    rest. Correcting during those brief blips causes large, damaging error
    spikes (confirmed: an early version without the min_still_run guard made
    RMSE ~30x worse). min_still_run=50 was chosen from real data -- a
    direction-reversal blip lasted ~30 samples, genuine rest periods ran into
    the hundreds.

    Validated against real robot ground truth (sessions 402-405):
      - device 281: ~9% RMSE improvement on average
      - devices 279, 282: ~no effect (calibration was already near-perfect)
      - device 280: makes things WORSE (-18% average) -- do not apply to
        this device until its hardware issue (see calibration.py notes) is
        resolved
    """

    # Devices confirmed safe to drift-correct. Device 280 is deliberately
    # excluded -- see class docstring.
    ENABLED_DEVICES = {279, 281, 282}

    def __init__(self, device_id: int, correction_rate: float = 0.1, min_still_run: int = 50):
        self.device_id = device_id
        self.correction_rate = correction_rate
        self.min_still_run = min_still_run
        self._still_run_length = 0
        self._last_value: Optional[float] = None
        self._resting_value: Optional[float] = None
        self._calibrated_history: list[float] = []
        self._is_moving_history: list[bool] = []

    @property
    def enabled(self) -> bool:
        return self.device_id in self.ENABLED_DEVICES

    def step(self, calibrated_value: float, is_moving: bool) -> float:
        if not self.enabled:
            return calibrated_value  # pass through untouched for excluded devices

        self._calibrated_history.append(calibrated_value)
        self._is_moving_history.append(is_moving)

        if is_moving:
            self._still_run_length = 0
            self._last_value = calibrated_value
            return calibrated_value

        self._still_run_length += 1

        if self._resting_value is None:
            self._resting_value = estimate_resting_value(
                self._calibrated_history, self._is_moving_history, self.min_still_run
            )

        if self._still_run_length < self.min_still_run or self._resting_value is None:
            self._last_value = calibrated_value
            return calibrated_value

        corrected = self._last_value + self.correction_rate * (self._resting_value - self._last_value)
        self._last_value = corrected
        return corrected