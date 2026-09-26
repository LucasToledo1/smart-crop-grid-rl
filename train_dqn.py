"""DQN no mesmo estilo do train_ppo.py.
Uso: python train_dqn.py --run-name dqn_padrao_s42 --timesteps 200000
Ambiente original, Dict com MultiInputPolicy, Monitor e EvalCallback.
Resultados: logs/RUN/evaluations.npz; modelos/config.json em models/RUN.
Depende de TensorBoard, assim como train_ppo.py.
"""
import argparse
import os
import json
import hashlib
from pathlib import Path

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor


import smart_crop_irrigation_V2 as crop
from smart_crop_irrigation_V2 import SmartCropIrrigationEnv


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--buffer-size", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--target-update-interval", type=int, default=10_000)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--run-name", type=str, default="dqn_baseline")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--dry-rate", type=float, default=0.08)
    parser.add_argument("--exploration-fraction", type=float, default=0.3)
    parser.add_argument("--exploration-final-eps", type=float, default=0.1)
    parser.add_argument("--eval-episodes", type=int, default=10)
    args = parser.parse_args()
    if Path(args.run_name).name != args.run_name or args.run_name in ("", ".", ".."):
        parser.error("run-name deve ser apenas um nome de pasta")
    if not 0 <= args.dry_rate <= 1 or args.grid_size < 2 or args.max_steps < 1:
        parser.error("Parametros do ambiente invalidos")
    if args.timesteps <= args.learning_starts or args.learning_starts < 0 or args.learning_rate <= 0:
        parser.error("Verifique timesteps, learning-starts e learning-rate")
    for name in ("buffer_size", "batch_size", "train_freq", "target_update_interval", "eval_freq", "eval_episodes"):
        if getattr(args, name) < 1:
            parser.error(f"{name} deve ser positivo")
    if not 0 < args.gamma <= 1 or not 0 < args.exploration_fraction <= 1 or not 0 <= args.exploration_final_eps <= 1:
        parser.error("Gamma ou exploracao invalida")
    return args


def make_env(args, filename=None):
    env = SmartCropIrrigationEnv(
        dry_rate=args.dry_rate,
        grid_size=args.grid_size,
        max_steps=args.max_steps,
        render_mode=None,
    )
    return Monitor(env, filename=filename)


def main():
    args = parse_args()

    log_dir = os.path.join("logs", args.run_name)
    model_dir = os.path.join("models", args.run_name)

    if os.path.exists(log_dir) or os.path.exists(model_dir):
        raise FileExistsError("Execucao existente. Escolha outro --run-name.")
    os.makedirs(log_dir, exist_ok=False)
    os.makedirs(model_dir, exist_ok=False)
    config = vars(args).copy()
    config["policy"] = "MultiInputPolicy"
    config["evaluation_protocol"] = "EvalCallback, RNG continuo, seed inicial 2000"
    config["environment_sha256"] = hashlib.sha256(Path(crop.__file__).read_bytes()).hexdigest()
    Path(model_dir, "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    train_env = make_env(args, os.path.join(log_dir, "treino"))
    eval_env = make_env(args)
    eval_env.reset(seed=2000)
    eval_env.action_space.seed(2000)

    model = DQN(
        "MultiInputPolicy",
        train_env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        gamma=args.gamma,
        learning_starts=args.learning_starts,
        train_freq=args.train_freq,
        target_update_interval=args.target_update_interval,
        verbose=1,
        tensorboard_log="tensorboard_logs",
        exploration_fraction=args.exploration_fraction,
        exploration_final_eps=args.exploration_final_eps,
        seed=args.seed,
        device="cpu",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
    )

    try:
        model.learn(
            total_timesteps=args.timesteps,
            tb_log_name=args.run_name,
            callback=eval_callback,
        )
        model.save(os.path.join(model_dir, "final_model"))
    except KeyboardInterrupt:
        model.save(os.path.join(model_dir, "interrompido"))
        print("Interrompido: pesos salvos, sem replay buffer para retomada exata.")
        return
    finally:
        train_env.close()
        eval_env.close()

    print(
        f"Treino concluido. Modelo final em "
        f"{model_dir}/final_model.zip"
    )

    print(
        f"Melhor modelo em "
        f"{model_dir}/best_model.zip"
    )

    print(
        "Ver curvas: tensorboard --logdir tensorboard_logs"
    )


if __name__ == "__main__":
    main()