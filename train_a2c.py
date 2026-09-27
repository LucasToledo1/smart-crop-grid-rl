"""Treina um agente A2C no Smart Crop Irrigation Grid.

Uso:
    python train_a2c.py --run-name a2c_baseline --timesteps 200000
"""

import argparse
import os

from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from smart_crop_irrigation_V2 import SmartCropIrrigationEnv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--learning-rate", type=float, default=7e-4)  # Padrão do A2C
    parser.add_argument("--n-steps", type=int, default=5)  # Padrão do A2C
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--ent-coef", type=float, default=0.0)  # Ajuda na exploração
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--run-name", type=str, default="a2c_baseline")
    return parser.parse_args()


def main():
    args = parse_args()

    log_dir = os.path.join("logs", args.run_name)
    model_dir = os.path.join("models", args.run_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    train_env = Monitor(SmartCropIrrigationEnv())
    eval_env = Monitor(SmartCropIrrigationEnv())

    # Inicializa o A2C com os hiperparâmetros fornecidos
    model = A2C(
        "MultiInputPolicy",
        train_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        verbose=1,
        tensorboard_log="tensorboard_logs",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=args.eval_freq,
        n_eval_episodes=10,
        deterministic=True,
    )

    model.learn(
        total_timesteps=args.timesteps,
        tb_log_name=args.run_name,
        callback=eval_callback,
    )

    model.save(os.path.join(model_dir, "final_model"))
    print(f"Treino concluído. Modelo final em {model_dir}/final_model.zip")
    print(f"Melhor modelo (avaliação periódica) em {model_dir}/best_model.zip")
    print(f"Ver curvas: tensorboard --logdir tensorboard_logs")


if __name__ == "__main__":
    main()
