# Benchmark de hiperparâmetros - DQN

Gerado por `benchmark_dqn.py`: 4 configurações, 150,000 passos de treino cada, avaliação final com 30 episódios (`deterministic=True`).

Ambiente: `smart_crop_irrigation_V2`. Grid: 5x5; secagem: 0.005; limite: 250 passos.
SHA-256 do ambiente: `165c54b4b313cd34068058f72dbce3659f9333b5296e9353066288e4aa9855a3`.
Seed de treino: 42; seed inicial da avaliação: 2000. A avaliação reinicia o gerador nessa seed para cada configuração e depois usa seu fluxo contínuo.
Política: `MultiInputPolicy`, CPU, sem renderização. Avaliação do modelo final, não do melhor checkpoint.

| Config | learning_rate | exploration_fraction | batch_size | gamma | Recompensa média | Tempo de treino |
|---|---|---|---|---|---|---|
| exploracao_longa | 0.0005 | 0.6 | 64 | 0.99 | **443.6 ± 120.1** | 154s |
| batch_grande | 0.0005 | 0.3 | 256 | 0.99 | 358.8 ± 137.6 | 174s |
| lr_baixo | 0.0001 | 0.3 | 64 | 0.99 | 260.7 ± 71.2 | 163s |
| baseline | 0.0005 | 0.3 | 64 | 0.99 | 255.4 ± 138.2 | 166s |

## Leitura

- `exploracao_longa` apresentou a maior recompensa média nesta execução.
- Cada alternativa modifica um hiperparâmetro em relação ao baseline.
- O desvio padrão descreve a variação entre episódios. Não mede variação entre treinamentos.
- Uma única seed de treino não estabelece superioridade estatística. Diferença menor que um desvio padrão não comprova empate.
- Recompensa sozinha não demonstra que todas as plantas foram entregues nem explica a causa de um resultado inferior.
- A comparação direta com PPO/A2C requer a mesma versão e parâmetros do ambiente. Este relatório usa a V2.

## Configuração candidata para o treino longo

`exploracao_longa`: `{'learning_rate': 0.0005, 'exploration_fraction': 0.6, 'batch_size': 64, 'gamma': 0.99}`.

Selecionada pela maior média observada neste benchmark. Confirmar em novas seeds antes de concluir que é superior.

## Parâmetros comuns do DQN

`{'buffer_size': 50000, 'learning_starts': 1000, 'train_freq': 4, 'gradient_steps': 1, 'target_update_interval': 10000, 'exploration_initial_eps': 1.0, 'exploration_final_eps': 0.1}`

O tempo informado mede apenas `model.learn()` e depende do hardware. O benchmark treina modelos novos; não carrega os modelos de execuções anteriores.
