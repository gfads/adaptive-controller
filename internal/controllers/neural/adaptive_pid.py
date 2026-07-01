import time
import math
import numpy as np
import pandas as pd
import optuna
from time import sleep

# RL imports (optional, only needed for mode 'rl' or 'hybrid')
try:
    import gym
    from gym import spaces
    from stable_baselines3 import PPO
    SB3_AVAILABLE = True
except Exception:
    SB3_AVAILABLE = False


class PIDEnv(gym.Env):
    """
    Ambiente Gym simples que usa um DataFrame como sequência de observações.
    A ação é um delta em (kp, ki, kd). Observação é (error, integral_error, derivative_error).
    """
    metadata = {"render.modes": []}

    def __init__(self, df, kp_init=0.00098, ki_init=0.00028, kd_init=0.0,
                 action_scale=0.0005, clamp_min=0.0, clamp_max=1.0):
        super().__init__()
        # Expect dataframe with columns: 'error', 'integral_error', 'derivative_error'
        self.df = df.reset_index(drop=True)
        self.kp = float(kp_init)
        self.ki = float(ki_init)
        self.kd = float(kd_init)
        self.index = 0

        # action: continuous change in kp, ki, kd
        self.action_space = spaces.Box(
            low=np.array([-action_scale, -action_scale, -action_scale], dtype=np.float32),
            high=np.array([action_scale, action_scale, action_scale], dtype=np.float32),
            dtype=np.float32
        )
        # observation: error, integral_error, derivative_error
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
        )

        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def reset(self):
        self.index = 0
        # keep initial kp/ki/kd as is (useful to start from previous state)
        row = self.df.iloc[self.index]
        return np.array([row["error"], row["integral_error"], row["derivative_error"]], dtype=np.float32)

    def step(self, action):
        # apply action (delta)
        self.kp = float(np.clip(self.kp + float(action[0]), self.clamp_min, self.clamp_max))
        self.ki = float(np.clip(self.ki + float(action[1]), self.clamp_min, self.clamp_max))
        self.kd = float(np.clip(self.kd + float(action[2]), self.clamp_min, self.clamp_max))

        row = self.df.iloc[self.index]
        output = (self.kp * row["error"] + self.ki * row["integral_error"] + self.kd * row["derivative_error"])

        # reward: negative absolute output (minimize absolute output)
        reward = -abs(output)

        self.index += 1
        done = self.index >= len(self.df)

        if not done:
            row = self.df.iloc[self.index]
            obs = np.array([row["error"], row["integral_error"], row["derivative_error"]], dtype=np.float32)
        else:
            obs = np.zeros(3, dtype=np.float32)

        return obs, reward, done, {}

    def render(self, mode="human"):
        pass

    def close(self):
        pass


class AdaptivePID:
    """
    AdaptivePID final: Optuna + RL (PPO) + jitter control + history + utilities.
    mode: 'optuna', 'rl', or 'hybrid' (run optuna then RL and combine)
    """

    def __init__(
            self,
            mode="optuna",
            kp_init=0.00098,
            ki_init=0.00028,
            kd_init=0.0,
            # jitter params
            max_delta=0.15,
            alpha_p=0.2,
            alpha_i=0.05,
            alpha_d=0.4,
            deadzone=1e-6,
            # optuna params
            optuna_trials=100,
            # rl params
            rl_timesteps=5000,
            rl_device="cpu",  # use "cpu" to avoid AcceleratorError
            rl_action_scale=0.0005,
            clamp_min=0.0,
            clamp_max=1.0,
            # clamps for final KP/KI/KD (safety)
            kp_min=0.0, kp_max=1.0,
            ki_min=0.0, ki_max=1.0,
            kd_min=0.0, kd_max=1.0,
    ):
        self.mode = mode
        self.kp = float(kp_init)
        self.ki = float(ki_init)
        self.kd = float(kd_init)

        # jitter config
        self.max_delta = float(max_delta)
        self.alpha_p = float(alpha_p)
        self.alpha_i = float(alpha_i)
        self.alpha_d = float(alpha_d)
        self.deadzone = float(deadzone)

        # optimization config
        self.optuna_trials = int(optuna_trials)

        # RL config
        self.rl_timesteps = int(rl_timesteps)
        self.rl_device = rl_device
        self.rl_action_scale = float(rl_action_scale)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)

        # final clamps
        self.kp_min, self.kp_max = kp_min, kp_max
        self.ki_min, self.ki_max = ki_min, ki_max
        self.kd_min, self.kd_max = kd_min, kd_max

        # history list of dicts
        self.history = []

        # store RL model path if saved
        self._rl_model = None

        # validate SB3 if RL mode requested
        if self.mode in ("rl", "hybrid") and not SB3_AVAILABLE:
            raise RuntimeError("stable-baselines3 / gym not available. Install SB3 to use RL mode.")

    # --------------------------
    # Utilities: normalize dataframe columns
    # --------------------------
    @staticmethod
    def _normalize_df(df):
        # remove BOM/spaces and lower-case columns
        df = df.copy()
        df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=False).str.lower()
        # try to harmonize known typos
        if "intregal_error" in df.columns:
            df = df.rename(columns={"intregal_error": "integral_error"})
        return df

    # --------------------------
    # jitter control helper
    # --------------------------
    def _adjust_with_jitter(self, old, new, alpha):
        # deadzone
        if abs(new - old) < self.deadzone:
            return old

        # rate limit (fallback if old == 0)
        if old == 0:
            delta_max = self.max_delta
        else:
            delta_max = abs(old) * self.max_delta

        if abs(new - old) > delta_max:
            new = old + delta_max if new > old else old - delta_max

        # EMA smoothing
        final = alpha * new + (1 - alpha) * old

        return final

    # --------------------------
    # cost evaluator used by Optuna
    # --------------------------
    def _evaluate_pid_cost(self, kp, ki, kd, df):
        err = 0.0
        for _, row in df.iterrows():
            e = float(row["error"])
            ie = float(row["integral_error"])
            de = float(row["derivative_error"])
            output = kp * e + ki * ie + kd * de
            err += abs(output)
        return err

    # --------------------------
    # Optuna optimizer
    # --------------------------
    def _optimize_optuna(self, df):
        def objective(trial):
            kp = trial.suggest_float("kp", 1e-8, 1e-1, log=True)
            ki = trial.suggest_float("ki", 1e-8, 1e-1, log=True)
            kd = trial.suggest_float("kd", 0.0, 1e-1)
            return self._evaluate_pid_cost(kp, ki, kd, df)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.optuna_trials, show_progress_bar=False)
        best = study.best_params
        return {"kp": float(best["kp"]), "ki": float(best["ki"]), "kd": float(best["kd"])}

    # --------------------------
    # RL optimizer using PPO
    # --------------------------
    def _optimize_rl(self, df):
        if not SB3_AVAILABLE:
            raise RuntimeError("stable-baselines3 not installed; cannot run RL.")

        env = PIDEnv(
            df=df,
            kp_init=self.kp,
            ki_init=self.ki,
            kd_init=self.kd,
            action_scale=self.rl_action_scale,
            clamp_min=self.clamp_min,
            clamp_max=self.clamp_max
        )

        # Force device setting to avoid AcceleratorError on machines without GPU
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            device=self.rl_device,
            learning_rate=3e-4,
            n_steps=256,
            batch_size=64
        )

        # train
        model.learn(total_timesteps=self.rl_timesteps)

        # play one episode to get final parameters
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = env.step(action)

        # save model to memory (optional)
        self._rl_model = model
        return {"kp": float(env.kp), "ki": float(env.ki), "kd": float(env.kd)}

    # --------------------------
    # Public update API: pass dataframe of ~300 rows
    # --------------------------
    def update(self, df_lote):
        """
        df_lote: pandas DataFrame with columns ['error','integral_error','derivative_error'] (case-insensitive)
        Returns: dict with updated kp, ki, kd
        """
        # normalize
        df = self._normalize_df(df_lote)

        required = {"error", "integral_error", "derivative_error"}
        if not required.issubset(set(df.columns)):
            raise KeyError(f"DataFrame must contain columns: {required}. Got: {df.columns.tolist()}")

        # Option: run optuna / rl / hybrid
        raw = {"kp": self.kp, "ki": self.ki, "kd": self.kd}
        try:
            if self.mode == "optuna":
                raw = self._optimize_optuna(df)
            elif self.mode == "rl":
                raw = self._optimize_rl(df)
            elif self.mode == "hybrid":
                # first optuna (coarse), then RL (fine), combine by averaging raw suggestions
                raw_o = self._optimize_optuna(df)
                raw_r = self._optimize_rl(df)
                raw = {
                    "kp": float((raw_o["kp"] + raw_r["kp"]) / 2.0),
                    "ki": float((raw_o["ki"] + raw_r["ki"]) / 2.0),
                    "kd": float((raw_o["kd"] + raw_r["kd"]) / 2.0),
                }
            else:
                raise ValueError("mode must be 'optuna', 'rl' or 'hybrid'")

        except Exception as e:
            # If optimizer fails, keep current parameters (log the error)
            print(f"[AdaptivePID] optimizer error: {e}; keeping current kp/ki/kd.")
            raw = {"kp": self.kp, "ki": self.ki, "kd": self.kd}

        # Apply jitter control (rate limit + EMA + deadzone) using separate alphas
        kp_new = self._adjust_with_jitter(self.kp, raw["kp"], self.alpha_p)
        ki_new = self._adjust_with_jitter(self.ki, raw["ki"], self.alpha_i)
        kd_new = self._adjust_with_jitter(self.kd, raw["kd"], self.alpha_d)

        # Final clamps for safety
        kp_new = float(np.clip(kp_new, self.kp_min, self.kp_max))
        ki_new = float(np.clip(ki_new, self.ki_min, self.ki_max))
        kd_new = float(np.clip(kd_new, self.kd_min, self.kd_max))

        # update internal state
        self.kp, self.ki, self.kd = kp_new, ki_new, kd_new

        # log history
        record = {
            "timestamp": time.time(),
            "kp": self.kp,
            "ki": self.ki,
            "kd": self.kd,
            "raw_kp": float(raw.get("kp", math.nan)),
            "raw_ki": float(raw.get("ki", math.nan)),
            "raw_kd": float(raw.get("kd", math.nan)),
            "mode": self.mode
        }
        self.history.append(record)

        return {"kp": self.kp, "ki": self.ki, "kd": self.kd}

    # --------------------------
    # Save / load RL model (stable-baselines)
    # --------------------------
    def save_rl_model(self, path):
        if self._rl_model is None:
            raise RuntimeError("No RL model trained to save.")
        self._rl_model.save(path)

    def load_rl_model(self, path, env=None):
        if not SB3_AVAILABLE:
            raise RuntimeError("stable-baselines3 not installed.")
        model = PPO.load(path, env=env, device=self.rl_device)
        self._rl_model = model
        return model

    # --------------------------
    # export history to csv
    # --------------------------
    def export_history(self, path="pid_history.csv"):
        pd.DataFrame(self.history).to_csv(path, index=False)

    # pretty print last
    def last(self):
        if not self.history:
            return None
        return self.history[-1]
