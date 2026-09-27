"""Roda o ambiente com uma política aleatória ou um modelo treinado, só pra visualizar.

Uso:
    python demo.py                                            # politica aleatoria
    python demo.py --model models/ppo_best_1M/best_model.zip  # modelo treinado (PPO)
    python demo.py --model models/dqn_x/best_model.zip --algo dqn

Se o modelo foi treinado com parametros de ambiente diferentes do default
(grid_size, dry_rate, max_steps, inventory_capacity), passe os mesmos aqui -
o espaco de observacao precisa bater com o que o modelo aprendeu:

    python demo.py --model models/ppo_grid3_best/final_model.zip \\
        --grid-size 3 --dry-rate 0.01 --max-steps 200 --inventory-capacity 8
"""

import argparse

import pygame

from smart_crop_irrigation import SmartCropIrrigationEnv

ALGOS = {}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default=None, help="Caminho pro .zip de um modelo do SB3. Sem isso, usa politica aleatoria."
    )
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "dqn", "a2c"])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--dry-rate", type=float, default=0.08)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--inventory-capacity", type=int, default=3)
    parser.add_argument("--border-penalty", type=float, default=0.0)
    parser.add_argument("--idle-penalty", type=float, default=0.0)
    return parser.parse_args()


def load_model(path, algo):
    from stable_baselines3 import A2C, DQN, PPO

    ALGOS.update({"ppo": PPO, "dqn": DQN, "a2c": A2C})
    return ALGOS[algo].load(path)


def main():
    args = parse_args()
    model = load_model(args.model, args.algo) if args.model else None

    env = SmartCropIrrigationEnv(
        render_mode="human",
        grid_size=args.grid_size,
        dry_rate=args.dry_rate,
        max_steps=args.max_steps,
        inventory_capacity=args.inventory_capacity,
        border_penalty=args.border_penalty,
        idle_penalty=args.idle_penalty,
    )

    for ep in range(args.episodes):
        obs, info = env.reset()
        terminated = truncated = False
        total_reward = 0.0

        while not (terminated or truncated):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return

            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        print(
            f"Episodio {ep + 1}: recompensa total = {total_reward:.1f}, passos = {info['current_step']}, "
            f"entregues = {env.plants_delivered}/{env.total_plants}"
        )

    env.close()


if __name__ == "__main__":
    main()
