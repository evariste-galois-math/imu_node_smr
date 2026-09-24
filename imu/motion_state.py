from dataclasses import dataclass
import numpy as np

@dataclass
class MotionStateConfig:
    gyro_threshold: float = 0.02
    min_state_run: int = 5

class MotionStateDetector:

    def __init__(self, config: MotionStateConfig = MotionStateConfig()):
        self.config = config
        self._current_state = False
        self._pending_state = False
        self._pending_count = 0

    @staticmethod
    def gyro_magnitude(gyro_x: float, gyro_y: float, gyro_z: float) -> float:
        return float(np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2))

    def step(self, gyro_x: float, gyro_y: float, gyro_z: float) -> bool:
        raw_moving = self.gyro_magnitude(gyro_x, gyro_y, gyro_z) > self.config.gyro_threshold

        if raw_moving == self._current_state:
            self._pending_count = 0
            return self._current_state

        if raw_moving == self._pending_state:
            self._pending_count += 1
        else:
            self._pending_state = raw_moving
            self._pending_count = 1

        if self._pending_count >= self.config.min_state_run:
            self._current_state = raw_moving
            self._pending_count = 0

        return self._current_state