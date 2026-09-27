"""Benchmark DQN no padrão do benchmark PPO, com relatório Markdown automático.

Uso: python benchmark_dqn.py
Requer smart_crop_irrigation_V2.py na mesma pasta.
Treina modelos novos e gera benchmark_results_dqn.md ao concluir.
"""
import hashlib
from pathlib import Path
import time

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

import smart_crop_irrigation_V2 as crop
from smart_crop_irrigation_V2 import SmartCropIrrigationEnv

TIMESTEPS = 150_000
N_EVAL_EPISODES = 30
SEED = 42
EVAL_SEED = 2000
ENV_PARAMS = dict(grid_size=5, max_steps=250, dry_rate=0.005, render_mode=None)
REPORT_PATH = Path('benchmark_results_dqn.md')

CONFIGS = {
    'baseline': dict(learning_rate=5e-4, exploration_fraction=0.3, batch_size=64, gamma=0.99),
    'lr_baixo': dict(learning_rate=1e-4, exploration_fraction=0.3, batch_size=64, gamma=0.99),
    'exploracao_longa': dict(learning_rate=5e-4, exploration_fraction=0.6, batch_size=64, gamma=0.99),
    'batch_grande': dict(learning_rate=5e-4, exploration_fraction=0.3, batch_size=256, gamma=0.99),
}
COMMON_PARAMS = dict(buffer_size=50_000, learning_starts=1_000, train_freq=4,
                     gradient_steps=1, target_update_interval=10_000,
                     exploration_initial_eps=1.0, exploration_final_eps=0.1)


def save_report(results, environment_hash):
    ranked = sorted(results.items(), key=lambda item: -item[1][0])
    best_name = ranked[0][0]
    lines = [
        '# Benchmark de hiperparâmetros - DQN', '',
        f'Gerado por `benchmark_dqn.py`: {len(CONFIGS)} configurações, '
        f'{TIMESTEPS:,} passos de treino cada, avaliação final com '
        f'{N_EVAL_EPISODES} episódios (`deterministic=True`).', '',
        'Ambiente: `smart_crop_irrigation_V2`. '
        f"Grid: {ENV_PARAMS['grid_size']}x{ENV_PARAMS['grid_size']}; "
        f"secagem: {ENV_PARAMS['dry_rate']}; limite: {ENV_PARAMS['max_steps']} passos.",
        f'SHA-256 do ambiente: `{environment_hash}`.',
        f'Seed de treino: {SEED}; seed inicial da avaliação: {EVAL_SEED}. '
        'A avaliação reinicia o gerador nessa seed para cada configuração e depois usa seu fluxo contínuo.',
        'Política: `MultiInputPolicy`, CPU, sem renderização. Avaliação do modelo final, não do melhor checkpoint.', '',
        '| Config | learning_rate | exploration_fraction | batch_size | gamma | Recompensa média | Tempo de treino |',
        '|---|---|---|---|---|---|---|',
    ]
    for name, (mean, std, elapsed) in ranked:
        params = CONFIGS[name]
        reward = f'{mean:.1f} ± {std:.1f}'
        if name == best_name:
            reward = f'**{reward}**'
        lines.append(f"| {name} | {params['learning_rate']:g} | {params['exploration_fraction']} | "
                     f"{params['batch_size']} | {params['gamma']} | {reward} | {elapsed:.0f}s |")
    lines += ['', '## Leitura', '',
              f'- `{best_name}` apresentou a maior recompensa média nesta execução.',
              '- Cada alternativa modifica um hiperparâmetro em relação ao baseline.',
              '- O desvio padrão descreve a variação entre episódios. Não mede variação entre treinamentos.',
              '- Uma única seed de treino não estabelece superioridade estatística. Diferença menor que um desvio padrão não comprova empate.',
              '- Recompensa sozinha não demonstra que todas as plantas foram entregues nem explica a causa de um resultado inferior.',
              '- A comparação direta com PPO/A2C requer a mesma versão e parâmetros do ambiente. Este relatório usa a V2.', '',
              '## Configuração candidata para o treino longo', '',
              f'`{best_name}`: `{CONFIGS[best_name]}`.', '',
              'Selecionada pela maior média observada neste benchmark. Confirmar em novas seeds antes de concluir que é superior.', '',
              '## Parâmetros comuns do DQN', '', f'`{COMMON_PARAMS}`', '',
              'O tempo informado mede apenas `model.learn()` e depende do hardware. '
              'O benchmark treina modelos novos; não carrega os modelos de execuções anteriores.', '']
    REPORT_PATH.write_text('\n'.join(lines), encoding='utf-8')


def main():
    if REPORT_PATH.exists():
        raise FileExistsError(f'{REPORT_PATH} já existe. Renomeie o relatório anterior antes de repetir.')
    environment_hash = hashlib.sha256(Path(crop.__file__).read_bytes()).hexdigest()
    results = {}
    for index, (name, params) in enumerate(CONFIGS.items(), 1):
        print(f'\n=== [{index}/{len(CONFIGS)}] {name}: {params} ===', flush=True)
        train_env = Monitor(SmartCropIrrigationEnv(**ENV_PARAMS))
        eval_env = Monitor(SmartCropIrrigationEnv(**ENV_PARAMS))
        try:
            model = DQN('MultiInputPolicy', train_env, verbose=0, seed=SEED,
                        device='cpu', **COMMON_PARAMS, **params)
            t0 = time.perf_counter()
            model.learn(total_timesteps=TIMESTEPS)
            elapsed = time.perf_counter() - t0
            eval_env.reset(seed=EVAL_SEED)
            eval_env.action_space.seed(EVAL_SEED)
            mean_reward, std_reward = evaluate_policy(
                model, eval_env, n_eval_episodes=N_EVAL_EPISODES, deterministic=True)
            results[name] = (float(mean_reward), float(std_reward), elapsed)
            print(f'{name}: reward={mean_reward:.1f} +/- {std_reward:.1f} '
                  f'(treino: {elapsed:.0f}s)', flush=True)
        finally:
            train_env.close()
            eval_env.close()
    print('\n=== RESUMO (ordenado por recompensa média) ===')
    for name, (mean, std, elapsed) in sorted(results.items(), key=lambda item: -item[1][0]):
        print(f'{name:30s} reward={mean:8.1f} +/- {std:6.1f}   treino={elapsed:5.0f}s')
    save_report(results, environment_hash)
    print(f'\nRelatório gerado: {REPORT_PATH.resolve()}')


if __name__ == '__main__':
    main()
