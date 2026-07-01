from __future__ import annotations
import time
import math
import json
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import os


class AdaptiveBase:
    """
    Base class that provides:
    - storage of current kp, ki, kd
    - jitter control (deadzone, rate-limit, EMA smoothing) with separate alphas
    - dataframe normalization & validation helpers
    - history logging and export
    - utilities to get/set parameters
    """

    def __init__(
        self,
        kp_init: float = 0.00098,
        ki_init: float = 0.00028,
        kd_init: float = 0.0,
        max_delta: float = 0.15,
        alpha_p: float = 0.2,
        alpha_i: float = 0.05,
        alpha_d: float = 0.4,
        deadzone: float = 1e-6,
        kp_bounds: Tuple[float, float] = (0.0, 1.0),
        ki_bounds: Tuple[float, float] = (0.0, 1.0),
        kd_bounds: Tuple[float, float] = (0.0, 1.0),
        enforce_last_n: int = 300,
    ):
        # current parameters
        self.kp = float(kp_init)
        self.ki = float(ki_init)
        self.kd = float(kd_init)

        # jitter control params
        self.max_delta = float(max_delta)
        self.alpha_p = float(alpha_p)
        self.alpha_i = float(alpha_i)
        self.alpha_d = float(alpha_d)
        self.deadzone = float(deadzone)

        # bounds / safety clamps
        self.kp_min, self.kp_max = kp_bounds
        self.ki_min, self.ki_max = ki_bounds
        self.kd_min, self.kd_max = kd_bounds

        # how many last rows to enforce when optimizing
        self.enforce_last_n = int(enforce_last_n)

        # history list
        self.history = []

        # optional storage for RL model or other artifacts
        self._artifact_store = {}

    # ---------------------------
    # DataFrame helpers
    # ---------------------------
    @staticmethod
    def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize incoming DataFrame columns: strip spaces, remove BOM, lower-case.
        Also fixes common misspelling 'intregal_error' -> 'integral_error'.
        """
        df = df.copy()
        # normalize column labels
        df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=False).str.lower()
        if "intregal_error" in df.columns:
            df = df.rename(columns={"intregal_error": "integral_error"})
        return df

    @staticmethod
    def validate_required_columns(df: pd.DataFrame, required=("error", "integral_error", "derivative_error")):
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"Missing required columns: {missing}")

    def prepare_lote(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize, validate and trim DataFrame to last `enforce_last_n` rows.
        Returns a clean DataFrame ready for optimization.
        """
        df2 = self.normalize_df(df)
        self.validate_required_columns(df2)
        if len(df2) > self.enforce_last_n:
            df2 = df2.tail(self.enforce_last_n).reset_index(drop=True)
        return df2

    # ---------------------------
    # Jitter control (rate limit + EMA + deadzone)
    # ---------------------------
    def _adjust_single(self, old: float, new: float, alpha: float) -> float:
        """
        Adjust a single coefficient using deadzone, rate limit and EMA smoothing.
        """
        # deadzone: ignore trivial changes
        if abs(new - old) < self.deadzone:
            return old

        # rate limit: maximum absolute change per update
        if old == 0:
            delta_max = self.max_delta
        else:
            delta_max = abs(old) * self.max_delta

        if abs(new - old) > delta_max:
            new = old + delta_max if new > old else old - delta_max

        # EMA smoothing
        final = alpha * new + (1.0 - alpha) * old

        # enforce non-negativity (and clamp to safety bounds handled by caller)
        return max(final, 0.0)

    def apply_jitter(self, raw: Dict[str, float]) -> Tuple[float, float, float]:
        """
        Apply jitter control to a raw suggestion {kp, ki, kd}.
        Updates internal kp/ki/kd and returns the applied triple.
        """
        kp_new = self._adjust_single(self.kp, float(raw["kp"]), self.alpha_p)
        ki_new = self._adjust_single(self.ki, float(raw["ki"]), self.alpha_i)
        kd_new = self._adjust_single(self.kd, float(raw["kd"]), self.alpha_d)

        # safety clamps
        kp_new = float(np.clip(kp_new, self.kp_min, self.kp_max))
        ki_new = float(np.clip(ki_new, self.ki_min, self.ki_max))
        kd_new = float(np.clip(kd_new, self.kd_min, self.kd_max))

        self.kp, self.ki, self.kd = kp_new, ki_new, kd_new
        return self.kp, self.ki, self.kd

    # ---------------------------
    # History & utilities
    # ---------------------------
    def log(self, raw: Dict[str, Any], mode_name: str = "unknown") -> Dict[str, Any]:
        """
        Append a history record and return it.
        """
        rec = {
            "timestamp": time.time(),
            "mode": mode_name,
            "kp": float(self.kp),
            "ki": float(self.ki),
            "kd": float(self.kd),
            "raw_kp": float(raw.get("kp", math.nan)),
            "raw_ki": float(raw.get("ki", math.nan)),
            "raw_kd": float(raw.get("kd", math.nan)),
        }
        self.history.append(rec)
        return rec

    def export_history(self, path: str = "pid_history.csv") -> None:
        """
        Export history list to CSV.
        """
        pd.DataFrame(self.history).to_csv(path, index=False)

    def get_history_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)

    # ---------------------------
    # Parameter helpers
    # ---------------------------
    def get_params(self) -> Dict[str, float]:
        return {"kp": float(self.kp), "ki": float(self.ki), "kd": float(self.kd)}

    def set_params(self, kp: Optional[float] = None, ki: Optional[float] = None, kd: Optional[float] = None) -> None:
        if kp is not None:
            self.kp = float(kp)
        if ki is not None:
            self.ki = float(ki)
        if kd is not None:
            self.kd = float(kd)

    def save_params(self, path: str = "pid_params.json") -> None:
        payload = self.get_params()
        with open(path, "w") as f:
            json.dump(payload, f)

    def load_params(self, path: str = "pid_params.json") -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, "r") as f:
            payload = json.load(f)
        self.set_params(payload.get("kp"), payload.get("ki"), payload.get("kd"))