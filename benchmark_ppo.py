"""Benchmark rapido comparando configuracoes de hiperparametros do PPO.

Treina cada configuracao pelo mesmo numero de passos e avalia no final com
mais episodios (reduz ruido na comparacao). Serve como ponto de partida pra
documentar no relatorio as configuracoes testadas, nao so a melhor.

Uso: python benchmark_ppo.py
"""

import time

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from smart_crop_irrigation import SmartCropIrrigationEnv

TIMESTEPS = 150_000
N_EVAL_EPISODES = 30

CONFIGS = {
    "baseline": dict(learning_rate=3e-4, n_steps=2048, batch_size=64, gamma=0.99),
    "lr_baixo": dict(learning_rate=1e-4, n_steps=2048, batch_size=64, gamma=0.99),
    "rollout_longo_gamma_alto": dict(learning_rate=3e-4, n_steps=4096, batch_size=64, gamma=0.995),
    "batch_grande": dict(learning_rate=3e-4, n_steps=2048, batch_size=256, gamma=0.99),
}


def main():
    results = {}
    for name, params in CONFIGS.items():
        print(f"\n=== {name}: {params} ===")
        train_env = Monitor(SmartCropIrrigationEnv())
        model = PPO("MultiInputPolicy", train_env, verbose=0, **params)

        t0 = time.time()
        model.learn(total_timesteps=TIMESTEPS)
        elapsed = time.time() - t0

        eval_env = Monitor(SmartCropIrrigationEnv())
        mean_reward, std_reward = evaluate_policy(
            model, eval_env, n_eval_episodes=N_EVAL_EPISODES, deterministic=True
        )
        results[name] = (mean_reward, std_reward, elapsed)
        print(f"{name}: reward={mean_reward:.1f} +/- {std_reward:.1f} (treino: {elapsed:.0f}s)")

    print("\n=== RESUMO (ordenado por recompensa media) ===")
    for name, (mr, sr, el) in sorted(results.items(), key=lambda kv: -kv[1][0]):
        print(f"{name:30s} reward={mr:8.1f} +/- {sr:6.1f}   treino={el:5.0f}s")


if __name__ == "__main__":
    main()
