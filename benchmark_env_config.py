"""Benchmark de configuracoes do AMBIENTE (nao dos hiperparametros do PPO).

Ataca o problema de o agente nunca completar o grid: testa combinacoes de
grid_size, dry_rate, max_steps e inventory_capacity, usando os hiperparametros
de PPO ja validados no benchmark anterior (n_steps=4096, gamma=0.995).

Uso: python benchmark_env_config.py
"""

import time

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from smart_crop_irrigation import SmartCropIrrigationEnv

TIMESTEPS = 200_000
N_EVAL_EPISODES = 30

PPO_KWARGS = dict(learning_rate=3e-4, n_steps=4096, batch_size=64, gamma=0.995)

ENV_CONFIGS = {
    "baseline": dict(grid_size=5, dry_rate=0.08, max_steps=250, inventory_capacity=3),
    "capacidade_maior": dict(grid_size=5, dry_rate=0.08, max_steps=250, inventory_capacity=6),
    "secagem_lenta": dict(grid_size=5, dry_rate=0.03, max_steps=250, inventory_capacity=3),
    "grid_menor": dict(grid_size=3, dry_rate=0.08, max_steps=250, inventory_capacity=3),
    "combo_generoso": dict(grid_size=4, dry_rate=0.04, max_steps=300, inventory_capacity=6),
    "grid_minimo_trip_unico": dict(grid_size=3, dry_rate=0.05, max_steps=150, inventory_capacity=8),
}


def evaluate(model, env_kwargs, n_episodes):
    env = SmartCropIrrigationEnv(**env_kwargs)
    rewards, delivered, successes = [], [], 0
    for ep in range(n_episodes):
        obs, info = env.reset(seed=1000 + ep)
        terminated = truncated = False
        total_reward = 0.0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(action)
            total_reward += r
        rewards.append(total_reward)
        delivered.append(env.plants_delivered / env.total_plants)
        if terminated:
            successes += 1
    env.close()
    return rewards, delivered, successes


def main():
    results = {}
    for name, env_kwargs in ENV_CONFIGS.items():
        print(f"\n=== {name}: {env_kwargs} ===")
        train_env = Monitor(SmartCropIrrigationEnv(**env_kwargs))
        model = PPO("MultiInputPolicy", train_env, verbose=0, **PPO_KWARGS)

        t0 = time.time()
        model.learn(total_timesteps=TIMESTEPS)
        elapsed = time.time() - t0

        rewards, delivered, successes = evaluate(model, env_kwargs, N_EVAL_EPISODES)
        mean_reward = sum(rewards) / len(rewards)
        mean_delivered_pct = 100 * sum(delivered) / len(delivered)
        success_rate = 100 * successes / N_EVAL_EPISODES

        results[name] = (mean_reward, mean_delivered_pct, success_rate, elapsed)
        print(
            f"{name}: reward={mean_reward:.1f}  entregue={mean_delivered_pct:.1f}%  "
            f"sucesso={success_rate:.1f}%  (treino: {elapsed:.0f}s)"
        )

    print("\n=== RESUMO (ordenado por taxa de sucesso, depois % entregue) ===")
    for name, (mr, pct, sr, el) in sorted(
        results.items(), key=lambda kv: (-kv[1][2], -kv[1][1])
    ):
        print(f"{name:28s} sucesso={sr:5.1f}%  entregue={pct:5.1f}%  reward={mr:7.1f}  treino={el:4.0f}s")


if __name__ == "__main__":
    main()
