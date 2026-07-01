import numpy as np

class PIDModel:
    """
    Representa um controlador PID simples usado apenas para avaliação estatística.
    Não controla um processo real — apenas calcula a saída PID
    baseado nos erros fornecidos no lote.
    """

    def __init__(self, kp=0.00098, ki=0.00028, kd=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd

    # ----------------------------------------------------------------------
    def update(self, kp, ki, kd):
        """Atualiza os ganhos atuais."""
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)

    # ----------------------------------------------------------------------
    def compute_output(self, error, i_error, d_error):
        """Calcula o output PID para uma única amostra."""
        return (
            self.kp * error +
            self.ki * i_error +
            self.kd * d_error
        )

    # ----------------------------------------------------------------------
    def evaluate_batch(self, df):
        """
        Avalia o erro acumulado de todo o batch (usado pelos tuners).
        Quanto menor o erro, melhor o PID.
        """

        total = 0.0

        # usamos valores da própria instância (kp/ki/kd)
        kp = self.kp
        ki = self.ki
        kd = self.kd

        for _, row in df.iterrows():
            e = row["error"]
            ie = row["integral_error"]
            de = row["derivative_error"]

            output = kp * e + ki * ie + kd * de

            # objetivo → minimizar |output|
            total += abs(output)

        return total
