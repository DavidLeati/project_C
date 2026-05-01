# Changelog

---

## Abril de 2026 - Semana 5 (26/04 a 30/04)

*Foco: Correção Causal Multi-Timeframe, Previsão Probabilística de Volatilidade e Recalibração do Agente de Trading*

### Avanços:

- **Reformulação Probabilística dos Previsores Multi-Timeframe (30/04):**

  - Alteração dos quatro T0s de previsão para emitir dois canais por horizonte (`mu`, `raw_sigma`), substituindo o contrato anterior de retorno único por uma formulação explícita de retorno esperado e volatilidade esperada.
  - Implementação da loss probabilística `_predictor_probabilistic_loss`, usando erro padronizado por `sigma` e penalidade de calibração positiva, permitindo que o previsor informe não apenas a direção esperada do retorno, mas também o grau de incerteza local.
  - Criação do empacotamento de sinais `mu`, `sigma` e `mu/sigma` por timeframe, expandindo a entrada do `T0_trade` de 4 sinais crus para 12 sinais ajustados por risco.
- **Recalibração Arquitetural do Logit de Política do Trader (30/04):**

  - Introdução de escala vetorial opcional por saída no `HSAMA` (`output_scale_init`, `learnable_output_scale`), preservando a escala dos previsores e permitindo inflar seletivamente apenas o logit de posição do agente trader.
  - Configuração do `T0_trade` com `output_scale_init=(1.0, 8.0)`, mantendo o canal de retorno esperado em escala neutra e dando ao canal de posição amplitude inicial suficiente para escapar da região linear fraca do `tanh`.
  - Aplicação da escala nos dois caminhos de execução (`use_local_greedy` e execução final), garantindo consistência entre predições intermediárias e saída global do grafo.
- **Controle de Giro e Zona Morta de Exposição (30/04):**

  - Criação de `position_from_logits()` em `trade/loss.py`, mapeando logits para posições contínuas com uma zona morta configurável (`position_deadzone=0.05`) para zerar exposições pequenas e reduzir rebalanceamentos motivados por ruído.
  - Integração da mesma função no treino, na inferência OOS e no cálculo de métricas, eliminando divergência entre a posição otimizada pela `TriplexTradingLoss` e a posição usada no relatório financeiro.
  - Reajuste do `TriplexTradingLoss` para operar com custo transacional mais conservador, maior peso de PnL e menor penalidade de viés direcional, buscando reduzir a destruição de retorno por atrito sem forçar colapso em posição fixa.
- **Correção do Alinhamento Causal de Features Multi-Timeframe (30/04):**

  - Refatoração de `CryptoFeatureBuilder` para expor `build_features_df()`, separando a geração de features do empacotamento final em tensores e permitindo que cada timeframe construa seus indicadores no próprio ritmo nativo.
  - Alteração de `load_multi_timeframe_sol()` para processar SOL/BTC/ETH em 15m, 1h, 4h e 1d antes da projeção ao grid de 15m, evitando que indicadores de horizontes maiores sejam calculados artificialmente sobre uma série já expandida por forward-fill.
  - Alinhamento final das matrizes prontas via `.reindex(anchor_index).ffill().bfill()`, mantendo a causalidade temporal e preservando targets/momentums corretos para cada granularidade.
- **Métricas e Artefatos de Diagnóstico do Novo Regime (30/04):**

  - Expansão do CSV `SOL_trades_oos.csv` para registrar `pred_*`, `vol_*` e `edge_*` por timeframe, tornando auditável a decomposição entre retorno esperado, volatilidade prevista e sinal ajustado por risco.
  - Atualização do painel inferior do gráfico OOS para plotar `edge = retorno / volatilidade`, deslocando a leitura visual do retorno bruto para o sinal efetivamente consumido pelo trader.
  - Tradução e padronização de comentários internos em `objectives.py` e `online.py`, mantendo a documentação inline coerente com a linguagem principal do projeto.

### Atrasos:

- **Persistência de Baixa Confiança e Atrito no Backtest OOS (30/04):**
  - O primeiro diagnóstico após a escala do logit indicou que a confiança média do agente ainda permanecia baixa (`|tanh| ~ 0.075`) e que os custos consumiam mais que o PnL bruto, tornando claro que o gargalo não era apenas amplitude final, mas ausência de estimativa explícita de incerteza e de filtro de não-operação.
  - Essa constatação atrasou a validação de performance do agente e forçou a mudança de contrato dos previsores para um regime probabilístico, com volatilidade prevista e sinal `mu/sigma` antes de nova rodada de backtest.

---

## Abril de 2026 - Semana 4 (19/04 a 25/04)

*Foco: Backtest em Dados Reais, Arquitetura Multi-Agente por Timeframe e Suporte a GPU*

### Avanços:

- **Construção do Motor de Backtest Multi-Timeframe para Solana (21/04):**

  - Implementação do `SolanaMultiTFEngine` (`trade/engine.py`), substituindo o motor de agente único por uma arquitetura de **5 T0s independentes**: quatro previsores de log-retorno (SOL/USDT 15m, 1h, 4h, 1d) e um agente de trading que recebe as 4 previsões como features adicionais e otimiza posição via `TriplexTradingLoss`.
  - Cada previsor possui seu próprio `MultiScaleT0Config` com escalas *fast/slow* calibradas para o horizonte do seu timeframe (ex.: T0_1d com `stride=1, history_length=7` captura 7 dias de regime).
  - O agente trader opera com `in_features = features_15m + 4` (sinais dos previsores concatenados), conectando os quatro horizontes temporais em uma única decisão de posição long/short contínua.
- **Pipeline de Alinhamento Multi-Timeframe sem Look-ahead Bias (21/04):**

  - Criação do método `load_multi_timeframe_sol()` em `trade/dataset.py`, que carrega os 4 arquivos Parquet da SOL, re-indexa os timeframes maiores (1h, 4h, 1d) ao grid de 15m via `.reindex() + .ffill()` e processa features com `window_size` proporcional a cada granularidade.
  - O alinhamento garante que o dado de uma vela diária só se torna visível nos passos de 15m *após* o fechamento daquela vela, eliminando qualquer contaminação futura nos tensores de treino e teste.
  - Split cronológico sincronizado: todos os timeframes são truncados ao comprimento mínimo comum antes da divisão train/test, mantendo o alinhamento temporal perfeito entre os 5 agentes.
- **Suporte a CUDA e Detecção Automática de Dispositivo (21/04):**

  - O engine detecta automaticamente CUDA na inicialização (`torch.cuda.is_available()`) e reporta o nome da GPU. Todos os modelos HSAMA e os `MultiScaleT0Builder` (GRUs + projeções) são movidos para o dispositivo via `_move_runtime_to_device()`.
  - Os tensores X/Y de todos os timeframes são transferidos para o device em bloco ao início do backtest, eliminando transferências host↔device por batch durante o treino.
  - Tensores de resultado (`posicoes_teste`, `Y_test_15m`) retornam a `.cpu()` apenas para métricas finais e geração de gráficos.
- **Aquisição de Dados Históricos via Binance (`fetch_binance.py`) (21/04):**

  - Script de coleta que baixa candles históricos da Binance para múltiplos pares e intervalos (15m, 1h, 4h, 1d), replicando a estrutura de colunas (`open_time`, `open`, `high`, `low`, `close`, `volume`) e salvando em Parquet no diretório `trade/data/`.
- **Instalação e Configuração do Ambiente de Trade (21/04):**

  - Instalação do pacote `pyarrow` para suporte a leitura de Parquet e do projeto em modo editable (`pip install -e .`) para resolver resolução de módulos ao executar scripts diretamente.
  - Adição de guard de `sys.path` em `engine.py` e `dataset.py` com fallback de imports absolutos, permitindo execução tanto como script direto (`python trade\engine.py`) quanto como módulo de pacote.

### Atrasos:

- **Custo Computacional do Treino Online em CPU (21/04):**
  - O volume de dados (175k passos de 15m × 5 agentes × 2 épocas) tornava o treino inviável em CPU (~horas). A adição de suporte a CUDA foi necessária para viabilizar iterações rápidas, o que atrasou o primeiro resultado de backtest OOS completo.

---

## Abril de 2026 - Semana 3 (12/04 a 18/04)

*Foco: Padronização, Benchmarks Competitivos e Refatoração de Diretórios Globais*

### Avanços:

- **Diagnóstico e Correção de Viés Estrutural no Benchmark do Runtime (18/04):**
  - Identificação e correção de bug crítico onde `observe()` decompunha cada batch em passos individuais, gerando 16× mais atualizações de gradiente que os modelos de referência e causando divergência (~1.4 MSE vs ~0.98). Corrigidos defeitos secundários de paridade de hiperparâmetros e shape espúrio no `PrioritizedSurprisalBuffer`.
- **Flexibilização da Arquitetura de Observação do Runtime (18/04):**
  - Introdução do campo `observe_mode` em `HSAMARuntimeConfig` com dois regimes: `"online"` (padrão, sample-a-sample) e `"batch"` (um único passo cobre o batch completo + replay). Implementados `_prepare_context_batch()` e `_observe_batch()`, reduzindo o tempo de benchmark em ~5×.
- **Reorganização Estrutural de Repositórios Globais (17/04):**
  - Refatoração do sub-projeto `c_src`, isolando-o de um ambiente "sandbox" para formar uma arquitetura limpa, autônoma e modular (base do atual *Project C*).
- **Benchmarking de Ablações Estruturais (Padé-KAN) (15/04):**
  - Execução estruturada de benchmarks avaliando 4 variantes construtivas (Linear Fixa, Linear Dinâmica, "MLP Edge", e Padé-KAN) usando conjuntos reais da paridade SOLUSDT.
  - Implementação de barreiras estritas contra vazamento de dados ("data leakage") usando pipelines cronológicos e expansão da engenharia de métricas (retornos multi-horizonte e indicadores de momento).
- **Padronização Código-fonte e Paths (13/04):**
  - Renomeação padronizada do módulo central de importações em Python, alterando a menção `hsama_model` nativamente para `src`.
  - Atualização dos metadados de configuração no `pyproject.toml` e adaptação extensa de árvores de diretórios.

### Atrasos:

- **Gargalos Empíricos e Dificuldade no Benchmarking contra XGBoost (14/04):**
  - O projeto sofreu um atraso de validação significativa ao constatar extrema dificuldade da arquitetura meta-otimizadora em superar de forma pura o agrupamento de árvores de decisão estruturadas (XGBoost).
  - Foi necessário atrasar as evoluções para focar estritamente em um profundo "profiling" dos gargalos de convergência perante a regressão sintética. O problema forçou a criar um mecanismo mecânico paliativo de *early stopping* no pipeline de treinamento da arquitetura até sua superação.

---

## Abril de 2026 - Semana 2 (05/04 a 11/04)

*Foco: Estabilização Matemática, Descarte de Rota L2O e Criação do Design Arquitetural Base*

### Avanços:

- **Estabilização Numérica do Meta-Otimizador (10/04):**
  - Implementação crítica da normalização invariante à escala no módulo `FeatureBuilder` visando resgatar sinais minúsculos de gradiente durante o treinamento e sobrevivência do modelo aos ruídos normais em mercado.
- **Documentação de Paradigmas "Schema-v3" (08/04):**
  - Cristalização conceitual convertendo um projeto que era apenas linha de "script" em uma abstração orientada a *plugins* isolados: injeções puras para "datasets", modelo e meta-otimizadores.
- **Modularização de Funções Meta-Objetivas e Desconstrução do Monólito (06/04):**
  - Descarte do *Mean Squared Error (MSE)* estático. Desenvolvimento de API por componentes suportando precisão de penalidade direcional compensada e regulação ativa baseada nos regimes de volatilidade do lote ("per-batch sample weighting").

### Atrasos:

- **O Descarte da Rota L2O (*Learning to Optimize*) (11/04):**
  - **Atraso Maior da Semana:** Uma barreira total de pesquisa engavetou o caminho estrito de L2O. Foi detectado experimentalmente que assumir uma meta-arquitetura na forma clássica de formulação de *Learning to Optimize* não conseguia fugir das instabilidades extremas em regimes exóticos e de altíssima cauda de volatilidade.
  - Como atraso secundário desse sintoma, verificou-se o crônico "problema de partida a frio" (*cold start zone* - 10/04). O algoritmo se encontrava num platô de transição e perdia as dinâmicas dos gradientes.
  - **Consequência do Abandono:** Essa falha motivou as 12 correções severas mapeadas no sub-diretório `notes/` e forçou o total redirecionamento para o sistema componentizado e configurável via grafos da versão HSAMA definitiva.
