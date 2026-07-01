import numpy as np 
from time import time 
from .tunners.optuna_tuner import OptunaTuner 
from .tunners.rl_tuner import RLTuner 
from .tunners.spsa_tuner import SPSATuner 
from .tunners.bayesian_tuner import BayesianTuner 
from .tunners.evolution_timer import EvolutionTuner 
from .tunners.pso_tuner import PSOTuner 
from .models.pid_model import PIDModel

class AdaptivePID:
    def __init__(self, mode="optuna", jitter_cfg=None, initial_params=None, jitter_enabled=True):
        self.mode = mode.lower()
        self.jitter_enabled = bool(jitter_enabled)

        # Inicial PID
        if initial_params is None:
            self.model = PIDModel(kp=0.00098, ki=0.00028, kd=0.0)
        else:
            self.model = PIDModel(**initial_params)

        # jitter configuration
        self.jitter_cfg = jitter_cfg or {
            "max_delta": 0.15,
            "alpha": 0.3,
            "deadzone": 0.000001,
        }

        # Tuner map
        self.tuners = {
            "optuna": OptunaTuner(),
            "rl": RLTuner(),
            "spsa": SPSATuner(),
            "bayes": BayesianTuner(),
            "evolution": EvolutionTuner(),
            "pso": PSOTuner(),
        }

        if self.mode not in self.tuners:
            raise ValueError(f"Modo inválido '{self.mode}'")

        self.tuner = self.tuners[self.mode]

    # ------------------------------------------------------------
    # JITTER CONTROL
    # ------------------------------------------------------------
    def apply_jitter(self, old, new):
        if not self.jitter_enabled:
            return new  # <<<<<< JITTER OFF (retorna valor cru)

        max_delta = self.jitter_cfg["max_delta"]
        alpha = self.jitter_cfg["alpha"]
        deadzone = self.jitter_cfg["deadzone"]

        if abs(new - old) < deadzone:
            return old

        delta_max = abs(old) * max_delta
        diff = new - old

        if abs(diff) > delta_max:
            new = old + np.sign(diff) * delta_max

        return alpha * new + (1 - alpha) * old

    # ------------------------------------------------------------
    # CICLO COMPLETO
    # ------------------------------------------------------------
    def update(self, df):
        if len(df) < 10:
            raise ValueError("DataFrame muito pequeno (mín: 10 linhas).")

        df_batch = df.tail(300).reset_index(drop=True)

        # tuner calcula novos KP/KI/KD
        new_params = self.tuner.tune(df_batch)

        raw_kp = new_params['kp']
        raw_ki = new_params['ki']
        raw_kd = new_params['kd']

        # aplicar jitter ou não
        kp = self.apply_jitter(self.model.kp, raw_kp)
        ki = self.apply_jitter(self.model.ki, raw_ki)
        kd = self.apply_jitter(self.model.kd, raw_kd)

        self.model.update(kp, ki, kd)

        return {
            "kp": float(kp),
            "ki": float(ki),
            "kd": float(kd),
        }