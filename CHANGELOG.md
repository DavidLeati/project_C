# Changelog

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
