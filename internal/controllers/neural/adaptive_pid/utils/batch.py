"""
adaptive_pid.utils.batch
------------------------------------
Handles batch preparation: validation + trimming to last N samples.
"""

from __future__ import annotations
import pandas as pd


REQUIRED_COLUMNS = {"error", "integral_error", "derivative_error"}


def validate_batch(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def trim_batch(df: pd.DataFrame, n: int = 300) -> pd.DataFrame:
    """Returns the last n rows (default 300)."""
    if len(df) <= n:
        return df
    return df.tail(n)
