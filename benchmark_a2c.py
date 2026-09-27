"""Benchmark rapido comparando configurações de hiperparametros do A2C.

Uso: python benchmark_a2c.py
"""

import time
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from smart_crop_irrigation_V2 import SmartCropIrrigationEnv

TIMESTEPS = 150_000
N_EVAL_EPISODES = 30

# O A2C não usa 'batch_size'. Ajustado os n_steps e o coeficiente de entropia,
# criado duas novas configurações exploracao_extrema e exploracao_muito_alta para testar o efeito da exploração no desempenho do agente.
CONFIGS = {
    "baseline": dict(learning_rate=7e-4, n_steps=5, gamma=0.99, ent_coef=0.0),
    "lr_baixo": dict(learning_rate=1e-4, n_steps=5, gamma=0.99, ent_coef=0.0),
    "n_steps_longo": dict(learning_rate=7e-4, n_steps=20, gamma=0.99, ent_coef=0.0),
    "exploracao_alta": dict(learning_rate=7e-4, n_steps=5, gamma=0.99, ent_coef=0.01),
    "exploracao_muito_alta": dict(
        learning_rate=7e-4, n_steps=5, gamma=0.99, ent_coef=0.05
    ),
    "exploracao_extrema": dict(learning_rate=7e-4, n_steps=5, gamma=0.99, ent_coef=0.1),
}


def main():
    results = {}
    for name, params in CONFIGS.items():
        print(f"\n=== {name}: {params} ===")
        train_env = Monitor(SmartCropIrrigationEnv())
        model = A2C("MultiInputPolicy", train_env, verbose=0, **params)

        t0 = time.time()
        model.learn(total_timesteps=TIMESTEPS)
        elapsed = time.time() - t0

        eval_env = Monitor(SmartCropIrrigationEnv())
        mean_reward, std_reward = evaluate_policy(
            model, eval_env, n_eval_episodes=N_EVAL_EPISODES, deterministic=True
        )
        results[name] = (mean_reward, std_reward, elapsed)
        print(
            f"{name}: reward={mean_reward:.1f} +/- {std_reward:.1f} (treino: {elapsed:.0f}s)"
        )

    print("\n=== RESUMO (ordenado por recompensa média) ===")
    for name, (mr, sr, el) in sorted(results.items(), key=lambda kv: -kv[1][0]):
        print(f"{name:30s} reward={mr:8.1f} +/- {sr:6.1f}   treino={el:5.0f}s")


if __name__ == "__main__":
    main()
