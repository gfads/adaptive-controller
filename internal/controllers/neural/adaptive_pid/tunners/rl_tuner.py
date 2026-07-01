# RL tuner (PPO). stable-baselines3 must be installed to use this tuner.
import numpy as np
from ..base import AdaptiveBase

try:
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO
    SB3 = True
except Exception:
    SB3 = False


class PIDEnv(gym.Env):
    """
    Ambiente Gym que itera sobre um DataFrame (seq. de observações).
    A ação é delta em (kp, ki, kd). Observação: (error, integral_error, derivative_error).
    """
    metadata = {"render.modes": []}

    def __init__(self, df, kp_init, ki_init, kd_init, action_scale=0.0005, clamp_min=0.0, clamp_max=1.0):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.kp = float(kp_init)
        self.ki = float(ki_init)
        self.kd = float(kd_init)
        self.index = 0

        self.action_space = spaces.Box(
            low=np.array([-action_scale]*3, dtype=np.float32),
            high=np.array([action_scale]*3, dtype=np.float32),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3,),
            dtype=np.float32
        )

        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.index = 0
        row = self.df.iloc[self.index]
        obs = np.array(
            [row['error'], row['integral_error'], row['derivative_error']],
            dtype=np.float32
        )
        return obs, {}

    def step(self, action):
        # aplica delta e clampa
        self.kp = float(np.clip(self.kp + float(action[0]), self.clamp_min, self.clamp_max))
        self.ki = float(np.clip(self.ki + float(action[1]), self.clamp_min, self.clamp_max))
        self.kd = float(np.clip(self.kd + float(action[2]), self.clamp_min, self.clamp_max))

        row = self.df.iloc[self.index]
        output = (
            self.kp * row['error'] +
            self.ki * row['integral_error'] +
            self.kd * row['derivative_error']
        )

        reward = -abs(output)  # minimizar erro

        self.index += 1
        terminated = self.index >= len(self.df)
        truncated = False  # não usamos limite de tempo artificial

        if not terminated:
            row = self.df.iloc[self.index]
            obs = np.array(
                [row['error'], row['integral_error'], row['derivative_error']],
                dtype=np.float32
            )
        else:
            obs = np.zeros(3, dtype=np.float32)

        return obs, reward, terminated, truncated, {}


class RLTuner(AdaptiveBase):

    def __init__(self, *args, rl_timesteps=5000, rl_device='cpu',
                 action_scale=0.0005, clamp_min=0.0, clamp_max=1.0, **kwargs):

        super().__init__(*args, **kwargs)

        self.rl_timesteps = int(rl_timesteps)
        self.rl_device = rl_device
        self.action_scale = float(action_scale)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)

        if not SB3:
            raise RuntimeError("stable-baselines3 (and gym) are required for RLTuner")

    # 🔥 IMPORTANTE — método padronizado
    def tune(self, df):
        return self.optimize(df)

    def optimize(self, df):
        df = self.normalize_df(df)

        if len(df) >= self.enforce_last_n:
            df = df.tail(self.enforce_last_n)

        env = PIDEnv(
            df, self.kp, self.ki, self.kd,
            action_scale=self.action_scale,
            clamp_min=self.clamp_min,
            clamp_max=self.clamp_max
        )

        model = PPO(
            'MlpPolicy',
            env,
            verbose=0,
            device=self.rl_device,
            learning_rate=3e-4
        )

        model.learn(total_timesteps=self.rl_timesteps)

        # Roda um episódio para capturar KP/KI/KD finais
        obs, _ = env.reset()
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)

        self._rl_model = model  # opcional

        return {
            "kp": float(env.kp),
            "ki": float(env.ki),
            "kd": float(env.kd)
        }
