"""Roda o ambiente com uma política aleatória, só pra visualizar o grid funcionando."""

import pygame

from smart_crop_irrigation import SmartCropIrrigationEnv


def main(n_episodes=5):
    env = SmartCropIrrigationEnv(render_mode="human")

    for ep in range(n_episodes):
        obs, info = env.reset()
        terminated = truncated = False
        total_reward = 0.0

        while not (terminated or truncated):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return

            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        print(f"Episodio {ep + 1}: recompensa total = {total_reward:.1f}, passos = {info['current_step']}")

    env.close()


if __name__ == "__main__":
    main()
