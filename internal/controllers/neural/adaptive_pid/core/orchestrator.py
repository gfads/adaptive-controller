"""
adaptive_pid.core.orchestrator
---------------------------------------------------
Central controller responsible for coordinating:
  - batch preparation (300 amostras)
  - tuner execution (optuna, RL, SPSA, PSO)
  - jitter application
  - PID model evaluation
  - historical logging

This keeps the AdaptivePID class clean and high-level.
"""

from __future__ import annotations

import pandas as pd
from typing import Dict, Optional

from adaptive_pid.utils.batch import validate_batch, trim_batch
from adaptive_pid.utils.jitter import apply_jitter
from adaptive_pid.utils.history import log_history
from adaptive_pid.models.pid_model import PIDErrorModel


class PIDOrchestrator:
    """
    Orchestrates the entire PID adaptation cycle.
    """

    def __init__(
        self,
        tuner,
        jitter_cfg: Optional[Dict] = None,
        history_list: Optional[list] = None,
        initial_gains: Optional[Dict[str, float]] = None,
    ):
        self.tuner = tuner
        self.jitter_cfg = jitter_cfg or {
            "max_delta": 0.15,
            "alpha": 0.3,
            "deadzone": 1e-6
        }
        self.history = history_list if history_list is not None else []

        # Gains
        self.kp = initial_gains.get("kp", 0.001) if initial_gains else 0.001
        self.ki = initial_gains.get("ki", 0.0002) if initial_gains else 0.0002
        self.kd = initial_gains.get("kd", 0.0) if initial_gains else 0.0

        # PID model
        self.model = PIDErrorModel()

    # ------------------------------------------------------------------ #
    # CORE UPDATE
    # ------------------------------------------------------------------ #
    def update(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Receives raw data of any size, extracts the last 300 lines,
        tunes PID gains using the configured tuner,
        applies jitter smoothing, logs history and returns new gains.
        """

        # 1) Validate + trim batch
        validate_batch(df)
        df_batch = trim_batch(df, 300)

        # 2) Raw PID tuning (could be Optuna, RL, SPSA, PSO...)
        raw_params = self.tuner.tune(df_batch)

        raw_kp = raw_params["kp"]
        raw_ki = raw_params["ki"]
        raw_kd = raw_params["kd"]

        # 3) Jitter smoothing
        kp_new, ki_new, kd_new = apply_jitter(
            self.kp, self.ki, self.kd,
            raw_kp, raw_ki, raw_kd,
            max_delta=self.jitter_cfg["max_delta"],
            alpha=self.jitter_cfg["alpha"],
            deadzone=self.jitter_cfg["deadzone"]
        )

        # 4) Update internal state
        self.kp, self.ki, self.kd = kp_new, ki_new, kd_new

        # 5) History
        log_history(
            self.history,
            self.tuner.name,
            self.kp, self.ki, self.kd,
            raw_kp, raw_ki, raw_kd
        )

        # 6) Return gains
        return {
            "kp": self.kp,
            "ki": self.ki,
            "kd": self.kd,
            "raw_kp": raw_kp,
            "raw_ki": raw_ki,
            "raw_kd": raw_kd,
        }
