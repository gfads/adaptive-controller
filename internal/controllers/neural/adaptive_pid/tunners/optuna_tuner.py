import optuna
from ..models.pid_model import PIDModel


optuna.logging.set_verbosity(optuna.logging.WARNING)

class OptunaTuner:
    """
    Tuner baseado em Optuna (Bayesian Optimization).
    Ele gera kp, ki e kd ótimos analisando o lote de dados.
    """

    def __init__(
        self,
        kp_range=(1e-8, 1e-1),
        ki_range=(1e-8, 1e-1),
        kd_range=(0.0, 1e-1),
        n_trials=100,
    ):
        self.kp_range = kp_range
        self.ki_range = ki_range
        self.kd_range = kd_range
        self.n_trials = n_trials

    # ----------------------------------------------------------------------
    def tune(self, df):
        """
        Recebe um dataframe com exatamente os erros necessários:
        - error
        - integral_error
        - derivative_error

        Retorna kp, ki, kd otimizados.
        """

        pid = PIDModel()   # modelo temporário para avaliação

        def objective(trial):
            kp = trial.suggest_float("kp", self.kp_range[0], self.kp_range[1])
            ki = trial.suggest_float("ki", self.ki_range[0], self.ki_range[1])
            kd = trial.suggest_float("kd", self.kd_range[0], self.kd_range[1])

            pid.update(kp, ki, kd)
            return pid.evaluate_batch(df)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials)

        best = study.best_params

        return {
            "kp": float(best["kp"]),
            "ki": float(best["ki"]),
            "kd": float(best["kd"])
        }
