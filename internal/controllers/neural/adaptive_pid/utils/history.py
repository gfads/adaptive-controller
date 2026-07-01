"""
adaptive_pid.utils.history
------------------------------------
Handles structured logging for PID tuning processes.
"""

from __future__ import annotations
import time
from typing import Dict, List


def log_history(
    history: List[Dict],
    mode_name: str,
    kp: float,
    ki: float,
    kd: float,
    raw_kp: float,
    raw_ki: float,
    raw_kd: float,
) -> None:

    history.append({
        "timestamp": time.time(),
        "mode": mode_name,
        "kp": kp,
        "ki": ki,
        "kd": kd,
        "raw_kp": raw_kp,
        "raw_ki": raw_ki,
        "raw_kd": raw_kd,
    })
