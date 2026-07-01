import argparse
import pandas as pd
import sys
import os

# Adiciona o diretório pai (raiz do projeto) ao sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from adaptive_pid.adaptive_pid import AdaptivePID

#MODE = "optuna", "rl", "spsa", "bayes", "evolution", "pso"

parser = argparse.ArgumentParser()

parser.add_argument("--mode", type=str, required=True)
parser.add_argument("--csv", type=str, default="novo_estado.csv")

parser.add_argument("--kp", type=float, required=True)
parser.add_argument("--ki", type=float, required=True)
parser.add_argument("--kd", type=float, required=True)

parser.add_argument("--jitter_max", type=float, default=0.15)
parser.add_argument("--jitter_alpha", type=float, default=0.3)
parser.add_argument("--deadzone", type=float, default=1e-6)
parser.add_argument("--jitter_on", type=int, default=1, help="1 = jitter ON, 0 = jitter OFF")

args = parser.parse_args()

pid = AdaptivePID(
    mode=args.mode,
    jitter_cfg={
        "max_delta": args.jitter_max,
        "alpha": args.jitter_alpha,
        "deadzone": args.deadzone,
    },
    initial_params={"kp": args.kp, "ki": args.ki, "kd": args.kd},
    jitter_enabled=bool(args.jitter_on)
)

#print("AdaptivePID loop started. Mode:", args.mode)

try:
    df = pd.read_csv(args.csv)
except Exception as e:
    print("Erro lendo CSV:", e)
    time.sleep(5)
    #continue

#if len(df) < 300:
#    print(f"Aguardando 300 linhas (atualmente {len(df)}).")
#    time.sleep(5)
#    continue

# pegar últimas 300 linhas
#df_batch = df.tail(300).reset_index(drop=True)
df_batch = df.reset_index(drop=True)

result = pid.update(df_batch)

#print(time.strftime("%Y-%m-%d %H:%M:%S"), "Novo PID aplicado:", result)
print(result)

# opcional: salvar histórico local
try:
    pid_history = pid.get_history()
    import json
    with open("pid_history.json", "w") as fh:
        json.dump(pid_history, fh, indent=2)
except Exception:
    pass
