"""
adaptive_pid.utils.evaluation
------------------------------------
Defines a PID evaluation cost function used by tuners.
"""

from __future__ import annotations
import pandas as pd


def pid_cost(kp: float, ki: float, kd: float, df: pd.DataFrame) -> float:
    """
    Computes the cost of a PID configuration using absolute output error.
    """
    total = 0.0

    for _, row in df.iterrows():
        e = row["error"]
        ie = row["integral_error"]
        de = row["derivative_error"]

        output = kp * e + ki * ie + kd * de
        total += abs(output)

    return float(total)
