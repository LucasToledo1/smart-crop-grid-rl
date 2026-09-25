# Smart Crop Irrigation Grid

Ambiente customizado do Gymnasium: um robô navega um grid, rega e colhe
plantas antes que sequem, e entrega na base.

## Instalação

```bash
./install.sh
```

Cria o `.venv`, instala tudo do `requirements.txt` e valida a instalação
(ambiente + os 3 algoritmos do SB3).

## Rodar o ambiente visualmente

```bash
source .venv/bin/activate

python demo.py                                            # politica aleatoria
python demo.py --model models/ppo_best_1M/best_model.zip  # modelo treinado
```

## Treinar

```bash
python train_ppo.py --run-name minha_run --timesteps 200000
```

Hiperparâmetros ajustáveis: `--learning-rate`, `--n-steps`, `--batch-size`,
`--gamma`, `--eval-freq`. Cada `--run-name` gera sua própria pasta em
`logs/` e `models/`.

## Comparar hiperparâmetros

```bash
python benchmark_ppo.py
```

Roda várias configurações e compara. Resultados já testados em
`benchmark_results.md`.

## Ver as curvas de treino

```bash
tensorboard --logdir tensorboard_logs
```
