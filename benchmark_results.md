# Benchmark de hiperparâmetros - PPO

Gerado por `benchmark_ppo.py`: 4 configurações, 150.000 passos de treino cada,
avaliação final com 30 episódios (`deterministic=True`).

| Config | learning_rate | n_steps | batch_size | gamma | Recompensa média | Tempo de treino |
|---|---|---|---|---|---|---|
| rollout_longo_gamma_alto | 3e-4 | 4096 | 64 | 0.995 | **-31.4 ± 24.4** | 37s |
| baseline | 3e-4 | 2048 | 64 | 0.99 | -35.1 ± 25.5 | 38s |
| batch_grande | 3e-4 | 2048 | 256 | 0.99 | -47.1 ± 19.4 | 24s |
| lr_baixo | 1e-4 | 2048 | 64 | 0.99 | -75.5 ± 12.0 | 37s |

## Leitura

- `rollout_longo_gamma_alto` e `baseline` ficaram estatisticamente empatados
  (diferença menor que o desvio padrão de cada um). Rollout maior e gamma mais
  alto pareceram ajudar um pouco, mas não é conclusivo nessa amostra.
- `batch_grande` piorou: menos atualizações de gradiente pro mesmo volume de
  dados prejudicou o aprendizado nesse orçamento de passos.
- `lr_baixo` foi claramente o pior: 150k passos não foi suficiente pra essa
  taxa de aprendizado convergir.
- Nenhuma configuração resolveu a tarefa (recompensa ainda bem negativa em
  todas) — 150k passos é pouco pro problema. Serve pra comparar direção
  relativa entre hiperparâmetros, não como resultado final.

## Configuração escolhida para o treino longo

`rollout_longo_gamma_alto` (learning_rate=3e-4, n_steps=4096, batch_size=64,
gamma=0.995), rodado por mais tempo com `train_ppo.py` pra ver se de fato
converge pra uma política capaz de colher e entregar plantas.
