# Benchmark de hiperparâmetros - A2C

Gerado por `benchmark_a2c.py`: 6 configurações, 150.000 passos de treino cada, com dry_rate de 0.08
avaliação final com 30 episódios (`deterministic=True`). Para o teste do A2C Resolvi utilizar mais 2 configurações de hiperparâmetros a exploracao_muito_alta e a exploracao_extrema envolvendo o campo **ent_coef** coeficiente de entropia.

| Config | learning_rate | n_steps | ent_coef | gamma | Recompensa média | Tempo de treino |
|---|---|---|---|---|---|---|
| baseline | 7e-4 | 5 | 0.0 | 0.99 | **-73.0 ± 0.0** | 310s |
| lr_baixo | 1e-4 | 5 | 0.0 | 0.99 | -73.0 ± 0.0 | 293s |
| n_steps_longo | 7e-4 | 20 | 0.0 | 0.99 | -73.0 ± 0.0 | 202s |
| exploracao_alta | 7e-4 | 5 | 0.01 | 0.99 | -73.0 ± 0.0 | 314s |
| exploracao_muito_alta | 7e-4 | 5 | 0.05 | 0.99 | -73.0 ± 0.0 | 298s |
| exploracao_extrema | 7e-4 | 5 | 0.1 | 0.99 | -73.0 ± 0.0 | 305s |

## Leitura

- Todas as 6 configurações atingiram -73 com variância de 0, indicando um **Colapso de Política** para um ótimo local.
- O agente descobriu que o limite do episódio é 250 passos (-0.1 de penalidade por passo = -25.0) e que se ele não fizer nada, as 24 plantas morrem (-2.0 por morte = -48.0). Totalizando -73.0 pontos de penalidade.
- Como o ambiente possui *recompensas esparsas* e difíceis de alcançar, como o +20 da entrega na base, a rede neural do A2C avaliou que explorar gerava punições piores (regar planta morta ou colher na hora errada). A estratégia mais segura aprendida pelo robô foi fugir para um canto e não fazer absolutamente nada.
- **Teste de Entropia Extrema:** Para confirmar a falha na exploração, configurações agressivas de ruído (entropia a 5% e 10%) foram adicionadas porém mesmo com 150k passos o resultado não foi satisfatório.

## Configuração escolhida para o treino longo

`exploracao_alta` (learning_rate=7e-4, n_steps=5, ent_coef=0.01, gamma=0.99).
Apesar do empate geral, o parâmetro `ent_coef=0.01` injeta ruído na tomada de decisão do ator, sendo uma tentativa de forçar o agente a sair da inércia. Foi rodada por um tempo muito maior (500.000 passos) com o script `train_a2c.py` para avaliar se o modelo conseguiria encontrar as recompensas positivas.

## Conclusão do Treino Longo (500.000 passos)

Mesmo após 500.000 steps, **o A2C não conseguiu convergir para uma política funcional**, mantendo a recompensa de avaliação travada exatamente em **-73.0**.

**Veredito Técnico:**
1. **Falha do Crítico:** As métricas de treino revelaram uma *explained_variance*  negativa de -90.2, indicando que a rede neural perdeu a capacidade de prever o valor dos estados do tabuleiro.
2. **Barreira das Recompensas Esparsas:** A injeção de entropia e o aumento da quantidade de steps do treino não foram suficientes para fazer o agente descobrir a sequência exata de ações exigida para a vitória (andar -> regar -> colher -> base -> descarregar). 
3. **Trauma de Exploração:** A quantidade de punições passivas e ativas 'desmotivou' o agente a explorar novos métodos. O colapso de política se manteve: o A2C concluiu que ficando parado no canto da tela era menos punitivo que tentar aprender o jogo.

Este resultado documenta que a arquitetura clássica do A2C possui uma limitação com a formulação atual deste ambiente, um método que pode motivar o agente seria o **Reward Shaping**, fornecer pequenas recompensas ou forçar ele a se mover.

# Teste de Mitigação: Atraso na Degradação do Ambiente

Para conferir se a o motivo do agente ficar inerto seria um problema do ambiente em vez da arquitetura do agente, foi alterado a probabilidade de as plantas secarem (`dry_rate`) de **0.08 (8%)** para **0.01 (1%)**. Aumentando o tempo de vida útil das plantas, motivando o robô a explorar mais.

Gerado por `benchmark_a2c.py`: 6 configurações, 150.000 passos de treino cada, com dry_rate de 0.01.

| Config | learning_rate | n_steps | ent_coef | gamma | Recompensa média | Tempo de treino |
|---|---|---|---|---|---|---|
| baseline | 7e-4 | 5 | 0.0 | 0.99 | -59.1 ± 3.7 | 307s |
| lr_baixo | 1e-4 | 5 | 0.0 | 0.99 | -59.7 ± 4.3 | 305s |
| n_steps_longo | 7e-4 | 20 | 0.0 | 0.99 | -59.3 ± 3.9 | 186s |
| exploracao_alta | 7e-4 | 5 | 0.01 | 0.99 | -77.1 ± 37.0 | 276s |
| **exploracao_muito_alta** | 7e-4 | 5 | 0.05 | 0.99 | **-35.5 ± 9.3** | 275s |
| exploracao_extrema | 7e-4 | 5 | 0.1 | 0.99 | -59.5 ± 5.2 | 273s |

## Leitura (dry_rate = 0.01)

- **Rompimento da Inércia:** Todas as configurações saíram da recompensa travada e sem variância de `-73.0`, confirmando que o colapso de política foi mitigado apenas aliviando a punição por inação.
- **O Fator Entropia:** A configuração `exploracao_muito_alta` destacou-se com o melhor resultado no benchmark rápido (-35.5). A combinação do maior tempo de sobrevivência (1%) com um coeficiente de entropia alto (5%) permitiu ao A2C realizar explorações bem-sucedidas o suficiente para começar a otimizar a rota.

## Conclusão do Treino Longo com Mitigação (500k passos)

O modelo foi treinado por 500.000 passos na configuração `exploracao_muito_alta` com o `dry_rate=0.01`.

O agente reverteu o colapso, atingindo recompensas de avaliação positivas consistentes (alcançando uma média superior a **81.8** durante o treinamento). Na renderização prática, o modelo conseguiu realizar a sequência certa de passos para a maior recompensa: navegou pelo grid, encontrou o momento ideal de colheita e encheu o inventário antes de retornar à base, efetuando entregas múltiplas (até 6 plantas e 99.0 pontos em um único episódio limitador de 250 passos).

**Análise Parcial:** O A2C consegue ter um desempenho Mediano no ambiente sem *Reward Shaping*, desde que o atraso na troca de estados (`dry_rate` baixo) permita tempo de aprendizado suficiente antes que o ambiente colapse naturalmente.

## Conclusão do Treino Longo com Mitigação (1M passos)

O modelo foi treinado por 1.000.000 passos na configuração `exploracao_muito_alta` com o `dry_rate=0.01`.

Apesar de ter dobrado o tempo em relação ao teste anterior, o limite de desempenho do A2C não é a ausência de interação ,mas sim o problema da *recompensa esparsa*, onde o A2C enfrenta dificuldades por não armazenar cenários passados em um Replay Buffer como o algoritmo DQN.

**Resultados analisados:**
- **Limite de entregas:** O robô travou em no máximo 6 plantas entregues por episódio (de 24 possíveis).
- **Falta de tempo:** Todos os episódios estouraram o tempo máximo do jogo (250 passos). Ele demora muito nas rotas.
- **Pontuação instável:** O modelo variou bastante. Fez +109.0 pontos em um teste, mas logo em seguida caiu para -303.0 e -99.0.

**Resumo:**
O A2C entendeu a mecânica do jogo (regar, colher e entregar na base) e conseguiu sair da inércia. O problema é que ele não sabe planejar os melhores caminhos, dando voltas desnecessárias e não colhendo 3 plantas antes de entregar a base, as plantas começam a secar e as punições se acumulam, o que explica as pontuações muito negativas de vez em quando.

## Conclusão Definitiva (Treino de 500.000 com Secagem a cada 10 passos)

Após balancear o ambiente para atrasar que ocorra a secagem a cada 10 passos (mantendo a taxa de mortalidade em 8%), o A2C foi treinado por 500.000 passos com o hiperparâmetro de `exploracao_muito_alta`. 

O modelo finalizou o treinamento com uma recompensa média de 119. O modelo conseguiu sair da inércia inicial e entregar de 4 a 5 plantas por episódio de forma consistente (de 5 episódios, nos 5 ele entregou de 5 a 4 plantas), mas atingiu um teto de aprendizado e caiu em um **Ótimo Local (Local Optimum)**, apresentando duas falhas estruturais características do algoritmo:

**Ineficiência de Inventário:** O agente ignora a capacidade de 3 espaços do inventário. Devido às recompensas esparsas, ele foca apenas na rota mais curta para o reforço: colhe uma única planta e volta correndo à base para garantir o +20.

**Loop de Movimentação:** A rede neural hipervalorizou a base como o único local seguro. Após algumas entregas, o agente prefere entrar em um loop infinito ao redor da zona de descarregamento (ou ficar parado) a assumir o risco matemático de explorar o restante do mapa.

**Análise Parcial:** O A2C aprende as mecânicas do jogo sendo mais consistente no início até a entrega numero 5, porém decai, mesmo sem a implementação de Reward Shaping (punições/recompensas de navegação).

## Teste com Reward Shaping Agressivo (Ambiente V2)

Para testar se a limitação do A2C era estrutural ou apenas falta de incentivo, o ambiente foi substituído pela versão smart_crop_irrigation_V2(`smart_crop_irrigation_v2`). Foram adicionadas penalidades de *Reward Shaping*: **-1.0 por colisão nas bordas** e **-0.5 por inatividade**. A mecânica de secagem retornou ao padrão acelerado (`dry_rate=0.08` aplicado a cada passo).

No benchmark rápido de 150k passos, a configuração `exploracao_muito_alta` subiu para **131.6 pontos** médios. Foi feito o treino longo final de 500.000 passos.

**Resultados observados:**
- **Ineficiência de Inventário:** Pressionado pelas punições de borda e de ficar parado, o robô abandonou a zona de conforto da base e passou a encher a capacidade máxima do inventário (3 plantas) antes de retornar para descarregar.
- **Salto de produtividade:** O teto de entregas subiu de 6 para 8 a 9 plantas consistentes por episódio, gerando recompensas totais altas que variaram de 139.0 a 194.4 pontos no teste prático.
- **Fim do Episódio devido a dry_rate alta:** Nenhum episódio chegou ao limite de 250 passos; todos encerraram entre 36 e 79 passos[cite: 7]. Devido ao `dry_rate` alto, as plantas não colhidas secaram rapidamente, causando um "Game Over" antecipado antes que o agente tivesse tempo físico para limpar o restante do mapa.

**Resumo:**
A aplicação de *Reward Shaping* punitivo do A2C, forçou o agente a explorar e usar o inventário de forma inteligente. O modelo provou ser capaz de operar o ambiente com eficiência superior aos testes anteriores. porém para a solução da tarefa, 6 a 9 plantas não é o cenário ideal.

## Otimização Final: Reward Shaping V2 + Sobrevivência Estendida (dry_rate=0.02)

Gerado por `benchmark_a2c.py`: 6 configurações, 150.000 passos de treino cada, utilizando o ambiente V2 (punição nas bordas e inatividade) e `dry_rate` mitigado para 0.02.

| Config | learning_rate | n_steps | ent_coef | gamma | Recompensa média | Tempo de treino |
|---|---|---|---|---|---|---|
| baseline | 7e-4 | 5 | 0.0 | 0.99 | -69.8 ± 2.2 | 440s |
| lr_baixo | 1e-4 | 5 | 0.0 | 0.99 | -70.2 ± 2.2 | 425s |
| n_steps_longo | 7e-4 | 20 | 0.0 | 0.99 | 312.7 ± 59.9 | 265s |
| exploracao_alta | 7e-4 | 5 | 0.01 | 0.99 | 136.1 ± 119.1 | 428s |
| exploracao_muito_alta | 7e-4 | 5 | 0.05 | 0.99 | 284.0 ± 53.7 | 437s |
| **exploracao_extrema** | 7e-4 | 5 | 0.1 | 0.99 | **314.2 ± 46.1** | 421s |

Para o teste definitivo, foi reduzido a taxa de secagem (`dry_rate=0.02`) com as punições do *Reward Shaping* (`smart_crop_irrigation_v2`), que aplica -1.0 por colisão nas bordas e -0.5 por inatividade. O benchmark inicial de 150k passos revelou que a configuração `exploracao_extrema` (`ent_coef=0.1`) foi a mais eficiente para este cenário, marcando 314.2 pontos de média.

Com base nisso, o modelo final foi treinado por 300.000 passos ao invés de 500.000, visto que nos penúltimo teste o modelo conseguiu compreender a dinamica e objetivo.

**Resultados do Teste Prático :**
- **Desempenho:** O modelo atingiu recompensas melhores em comparação com testes passados, com a menor pontuação sendo 338.0 e a maior chegando a 487.5 pontos por episódio.
- **Eficiência Logística Máxima:** O robô aprendeu o uso eficiente do inventário de forma consistente entregando entre 15 e 20 plantas (de 24 possíveis).
- **Gerenciamento de Tempo:** A redução do `dry_rate` foi eficaz. A maioria dos episódios encerrou entre 96 e 129 passos (quando as poucas plantas restantes secaram), dando tempo mais do que suficiente para o agente varrer quase todo o grid.

**Análise Final:**
Por mais que o A2C tenha apresentado dificuldades em atuar em ambientes com *recompensas esparsas* no início, quando submetido a um *Reward Shaping* punitivo (que destrói a zona de conforto do "ótimo local" na base) junto de uma taxa de secagem menor das plantas, o modelo prova ser eficiente para a tarefa. O agente aprendeu as tarefas de navegar, otimizar inventário e descarregar, consolidando o A2C como uma arquitetura viável para o problema proposto.
