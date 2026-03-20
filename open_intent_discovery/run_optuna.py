import optuna
import subprocess
import os


def objective(trial):

    # hiperparâmetros
    lr_pre = trial.suggest_float("lr_pre", 1e-6, 1e-4, log=True)
    lr = trial.suggest_float("lr", 1e-6, 5e-5, log=True)

    temperature = trial.suggest_float("temperature", 0.01, 0.2)

    batch_size = trial.suggest_categorical(
        "batch_size", [8, 16, 32]
    )

    output_dir = f"optuna_runs/trial_{trial.number}"
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "python",
        "run.py",

        "--lr_pre", str(lr_pre),
        "--lr", str(lr),

        "--train_temperature", str(temperature),

        "--train_batch_size", str(batch_size),

        "--output_dir", output_dir
    ]

    subprocess.run(cmd, check=True)

    score_file = os.path.join(output_dir, "optuna_score.txt")

    with open(score_file) as f:
        score = float(f.read().strip())

    return score


study = optuna.create_study(direction="maximize")

study.optimize(objective, n_trials=50)

print("Best parameters:", study.best_params)
print("Best score:", study.best_value)