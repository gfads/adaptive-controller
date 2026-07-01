import sys
import argparse
import json
import pandas as pd
from adaptive_pid import AdaptivePID

def run_adaptive_pid_optimization(
        input_file: str,
        mode: str,
        kp_init: float,
        ki_init: float,
        kd_init: float,
        max_delta: float,
        alpha_p: float,
        alpha_i: float,
        alpha_d: float,
        deadzone: float,
        optuna_trials: int,
        rl_timesteps: int,
        rl_device: str,
) -> tuple[float, float, float]:
    """
    Executa a otimização do AdaptivePID carregando o arquivo de dados
    e retornando os coeficientes Kp, Ki e Kd otimizados.
    """
    try:
        # 1. Carrega os dados e garante que as colunas essenciais existem
        df = pd.read_csv(input_file)

    except Exception as e:
        raise RuntimeError(f"Erro ao carregar, encontrar colunas ou preparar dados do arquivo {input_file}: {e}")

    # 3. Inicializa e executa o PID
    pid = AdaptivePID(
        mode=mode, kp_init=kp_init, ki_init=ki_init, kd_init=kd_init,
        max_delta=max_delta, alpha_p=alpha_p, alpha_i=alpha_i, alpha_d=alpha_d,
        deadzone=deadzone, optuna_trials=optuna_trials, rl_timesteps=rl_timesteps,
        rl_device=rl_device
    )

    # Assumimos que o .update(df) dispara todos os trials/timesteps de otimização
    pid.update(df)

    # 4. Retorna os coeficientes otimizados
    return pid.kp, pid.ki, pid.kd

def main():
    # ---------------------------------------------------------------
    # 1. Definição dos parâmetros padrão (como no seu código original)
    # ---------------------------------------------------------------
    DEFAULT_KP = 0.00098
    DEFAULT_KI = 0.00028
    DEFAULT_KD = 0.0
    DEFAULT_MAX_DELTA = 0.15
    DEFAULT_ALPHA_P = 0.2
    DEFAULT_ALPHA_I = 0.05
    DEFAULT_ALPHA_D = 0.4
    DEFAULT_DEADZONE = 1e-6
    DEFAULT_TRIALS = 100
    DEFAULT_TIMESTEPS = 4000

    parser = argparse.ArgumentParser(description="Executa otimização AdaptivePID via CLI.")

    # Argumento obrigatório para o nome do arquivo de dados (o 'novo_estado.csv')
    parser.add_argument('input_file', type=str, help='Caminho para o arquivo CSV de entrada (ex: novo_estado.csv).')

    # 2. Adiciona todos os parâmetros de configuração do AdaptivePID
    parser.add_argument('--mode', type=str, default='optuna', choices=['optuna', 'rl', 'hybrid'])
    parser.add_argument('--kp_init', type=float, default=DEFAULT_KP)
    parser.add_argument('--ki_init', type=float, default=DEFAULT_KI)
    parser.add_argument('--kd_init', type=float, default=DEFAULT_KD)
    parser.add_argument('--max_delta', type=float, default=DEFAULT_MAX_DELTA)
    parser.add_argument('--alpha_p', type=float, default=DEFAULT_ALPHA_P)
    parser.add_argument('--alpha_i', type=float, default=DEFAULT_ALPHA_I)
    parser.add_argument('--alpha_d', type=float, default=DEFAULT_ALPHA_D)
    parser.add_argument('--deadzone', type=float, default=DEFAULT_DEADZONE)
    parser.add_argument('--optuna_trials', type=int, default=DEFAULT_TRIALS)
    parser.add_argument('--rl_timesteps', type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument('--rl_device', type=str, default="cpu")

    args = parser.parse_args()

    try:
        # 3. Chama a função de otimização com TODOS os argumentos
        kp, ki, kd = run_adaptive_pid_optimization(
            input_file=args.input_file,
            mode=args.mode,
            kp_init=args.kp_init,
            ki_init=args.ki_init,
            kd_init=args.kd_init,
            max_delta=args.max_delta,
            alpha_p=args.alpha_p,
            alpha_i=args.alpha_i,
            alpha_d=args.alpha_d,
            deadzone=args.deadzone,
            optuna_trials=args.optuna_trials,
            rl_timesteps=args.rl_timesteps,
            rl_device=args.rl_device
        )

        # 4. Imprime o resultado final para STDOUT (única coisa que o Go lê)
        result = {"status": "success", "kp": kp, "ki": ki, "kd": kd}
        print(json.dumps(result))

    except Exception as e:
        # 5. Imprime o erro para STDERR (onde o Go deve procurar logs de erro)
        error_result = {"status": "error", "message": f"Falha na execução: {e}"}
        print(json.dumps(error_result), file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()