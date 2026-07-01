"""
adaptive_pid.utils.jitter
------------------------------------
Applies controlled smoothing, rate limiting and deadzone logic
to avoid unstable jumps in kp, ki, kd.
"""
from __future__ import annotations
from typing import Tuple


def apply_jitter(
    kp_old: float,
    ki_old: float,
    kd_old: float,
    kp_new: float,
    ki_new: float,
    kd_new: float,
    max_delta: float = 0.15,      # 15% max change
    alpha: float = 0.3,           # exponential smoothing factor
    deadzone: float = 1e-6        # ignore tiny changes
) -> Tuple[float, float, float]:

    def adjust(old: float, new: float) -> float:
        # Deadzone
        if abs(new - old) < deadzone:
            return old

        # Limit max percentual change
        limit = abs(old * max_delta)
        diff = new - old

        if abs(diff) > limit:
            # clamp diff to ±limit
            diff = limit if diff > 0 else -limit
            new = old + diff

        # Exponential smoothing
        return alpha * new + (1 - alpha) * old

    kp = adjust(kp_old, kp_new)
    ki = adjust(ki_old, ki_new)
    kd = adjust(kd_old, kd_new)

    return kp, ki, kd