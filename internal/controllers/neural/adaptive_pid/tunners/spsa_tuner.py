import numpy as np
from ..models.pid_model import PIDModel


class SPSATuner:
    """
    SPSA — Simultaneous Perturbation Stochastic Approximation.
    Indicado para problemas ruidosos.
    """

    def __init__(
        self,
        a=0.01,
        c=0.005,
        iterations=60,
        kp_range=(1e-6, 0.01),
        ki_range=(1e-6, 0.01),
        kd_range=(0.0, 0.01),
    ):
        self.a = a
        self.c = c
        self.iterations = iterations
        self.kp_range = kp_range
        self.ki_range = ki_range
        self.kd_range = kd_range

    # ----------------------------------------------------------------------
    def clip(self, kp, ki, kd):
        kp = np.clip(kp, *self.kp_range)
        ki = np.clip(ki, *self.ki_range)
        kd = np.clip(kd, *self.kd_range)
        return kp, ki, kd

    # ----------------------------------------------------------------------
    def tune(self, df):

        pid = PIDModel()

        # start from mid-point of ranges
        kp = np.mean(self.kp_range)
        ki = np.mean(self.ki_range)
        kd = np.mean(self.kd_range)

        for k in range(1, self.iterations + 1):

            ak = self.a / k
            ck = self.c / np.sqrt(k)

            # Rademacher random variables
            delta = np.random.choice([-1, 1], size=3)

            params_plus = (kp + ck * delta[0], ki + ck * delta[1], kd + ck * delta[2])
            params_minus = (kp - ck * delta[0], ki - ck * delta[1], kd - ck * delta[2])

            params_plus = self.clip(*params_plus)
            params_minus = self.clip(*params_minus)

            # evaluate
            pid.update(*params_plus)
            loss_plus = pid.evaluate_batch(df)

            pid.update(*params_minus)
            loss_minus = pid.evaluate_batch(df)

            gk = (loss_plus - loss_minus) / (2 * ck * delta)

            # gradient step
            kp -= ak * gk[0]
            ki -= ak * gk[1]
            kd -= ak * gk[2]

            kp, ki, kd = self.clip(kp, ki, kd)

        return {
            "kp": float(kp),
            "ki": float(ki),
            "kd": float(kd)
        }
