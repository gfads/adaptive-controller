import numpy as np
import warnings
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from ..models.pid_model import PIDModel


warnings.filterwarnings("ignore")

class BayesianTuner:
    """
    Implementação simples de Bayesian Optimization manual usando
    Gaussian Process Regressor.
    """

    def __init__(
        self,
        kp_range=(1e-6, 0.01),
        ki_range=(1e-6, 0.01),
        kd_range=(0.0, 0.01),
        iterations=40,
        samples_init=10,
    ):
        self.kp_range = kp_range
        self.ki_range = ki_range
        self.kd_range = kd_range
        self.iterations = iterations
        self.samples_init = samples_init

    # ----------------------------------------------------------------------
    def random_params(self):
        return (
            np.random.uniform(*self.kp_range),
            np.random.uniform(*self.ki_range),
            np.random.uniform(*self.kd_range),
        )

    # ----------------------------------------------------------------------
    def tune(self, df):

        pid = PIDModel()

        # Kernel for Bayesian Optimization
        kernel = ConstantKernel(1.0, (1e-3, 10)) * Matern(nu=2.5) + WhiteKernel()
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)

        X = []
        y = []

        # Initial random exploration
        for _ in range(self.samples_init):
            kp, ki, kd = self.random_params()
            pid.update(kp, ki, kd)
            score = pid.evaluate_batch(df)

            X.append([kp, ki, kd])
            y.append(score)

        X = np.array(X)
        y = np.array(y)

        # Bayesian Optimization loop
        for _ in range(self.iterations):
            gp.fit(X, y)

            # sample many candidates
            candidates = np.array([
                self.random_params() for _ in range(500)
            ])

            # evaluate acquisition function (Lower Confidence Bound)
            mu, sigma = gp.predict(candidates, return_std=True)
            lcb = mu - 1.2 * sigma
            best_idx = np.argmin(lcb)

            kp, ki, kd = candidates[best_idx]

            pid.update(kp, ki, kd)
            score = pid.evaluate_batch(df)

            X = np.vstack([X, [kp, ki, kd]])
            y = np.append(y, score)

        best_idx = np.argmin(y)
        best_params = X[best_idx]


        return {
            "kp": float(best_params[0]),
            "ki": float(best_params[1]),
            "kd": float(best_params[2])
        }
