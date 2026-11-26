# StanDeriv: Arquitetura Atual

## Visão geral

O repositório mantém módulos Python executáveis na raiz. Essa organização
preserva os entrypoints CLI, os imports usados pelos scripts e os caminhos de
persistência derivados de `__file__`. `dashboard/` contém o servidor Flask e a
interface; `tests/` contém a suíte; `docs/historico/` guarda relatórios datados,
evidências e documentação arquitetural anterior.

```text
Deriv WebSocket
      |
collector.py -> ticks_v2.csv -> dataset_builder.py
                                  |
                         indicators.py + feature_engine.py
                                  |
                    train_model.py + model_contract.py
                                  |
pipeline.py -> strategy.py <- ai_predictor.py
                    |             |
                executor.py <- risk_manager.py
                    |
               operacoes_v2.csv

dashboard/server.py -> dashboard/app.js + index.html + style.css
```

O diagrama representa responsabilidades, não uma ordem estrita de chamadas.
`pipeline.py` coordena coleta, avaliação/treino e o início condicionado da
execução. `bot.py` é um entrypoint alternativo para execução; `collector.py`
também pode ser executado separadamente.

O Modelo de Treino isolado usa candles de 5 minutos para contratos de 15 minutos,
com 50 candles por amostra e janela de 28 dias (até 8.064 posições). Os modos
combinados mantêm candles de 15 minutos (até 2.688 posições). `model_contract.py`
separa o timeframe da IA da base operacional homologada. `training_data.py`
isola snapshots por ativo, registra quarentena e preserva vetos temporais.
`training_history.py` complementa OHLC e ticks por conexão pública separada,
com cache incremental exclusivo da IA em `.training_inputs/history/`.
Páginas parciais e falhas não autorizam preencher preços ou liberar contratos.
Esse limite da IA não altera o orçamento da coleta de sete dias/672 posições.
`training_candles.py` mantém o buffer privado de 5 minutos, reaproveita ticks
verificados e completa somente dados ausentes. O executor usa esse buffer
apenas na IA isolada; Score e demais modos preservam os buffers de 15 minutos
e seu aquecimento de 150 candles. Recuperação pública roda em worker separado
e não fabrica candles menores a partir de OHLC de 15 minutos.
O scanner consulta o
catálogo `active_symbols`, exige ADX atualizado >= 25 e prioriza ativos não
sintéticos; só tenta sintéticos abertos quando nenhum não sintético é elegível.
`Histórico Ativos/` mantém candles OHLC independentes por símbolo, com merge
incremental e limite local de 336 candles (`ASSET_HISTORY_CANDLES`); o treino ainda exige ticks identificados. Um
bundle aprovado offline ainda
precisa de validação forward sem compras. Na IA isolada de 5 minutos,
`training_canary.py` admite o canary `isolated_ai_canary_v1` (8 resultados
não sobrepostos, pelo menos 7 acertos e saldo unitário positivo) quando os
metadados comprovam holdout elegível com ao menos 30 contratos; a reprovação é
antecipada quando a meta fica impossível. Sem essa prova, ou nos modos
combinados, vale a validação original de `shadow_monitor.py`: 30 resultados
não sobrepostos e acerto >= 55%. Ambos exigem payout líquido observado >= 70%.
O fluxo de modelos treinados pode abrir contratos somente em conta demo.

## Módulos homologados — 15/10/2026

Score Ponderado e coleta de dados foram homologados pelo usuário e estão
congelados. Consulte o [registro de homologação](docs/HOMOLOGACAO_SCORE_COLETA_2026-10-15.md)
e a [especificação do Score](docs/SCORE_PONDERADO_BOLLINGER.md). Não alterar seus
resultados por manutenção de módulos compartilhados sem autorização explícita.

Nos caminhos técnicos, o executor fornece fechamentos de 15 minutos para a estratégia,
que usa os últimos 50, com Bollinger nos últimos 20. `MIN_CANDLES=150` define
o aquecimento. O tick aciona a avaliação, mas não substitui o último fechamento.
A coleta do pipeline grava ticks ao vivo; o backfill pesado de ticks está
desativado. O histórico OHLC é atualizado por lacunas e não convertido em ticks.

`USE_AI_MODEL=False` permite o caminho técnico sem bundle. Com IA ativa, o
voto do modelo exige compatibilidade e sua execução mantém shadow/demo. O
consenso exige maioria estrita dos módulos habilitados. Risco é uma camada
separada: `ENFORCE_RISK_GATE=False` registra pausas/stops sem bloquear ordens.
Essa política não foi alterada nem homologada nesta revisão.

Keepalive, calendário, recuperação de lacunas e diagnóstico pertencem ao fluxo
de coleta/continuidade: `api_keepalive.py`, `market_calendar.py`,
`request_throttle.py` e `operational_diagnostics.py`. `runtime_status.py`
coordena o status por sessão. Recuperação mantém os candles válidos; novas
avaliações de entrada permanecem bloqueadas enquanto o histórico está incompleto.

O Modelo de Treino acompanha identidades de rótulos por símbolo em
`training_policy.py`, inclusive quando o total da janela móvel fica constante.
Somente na IA isolada (300/900), o pipeline prioriza candidatos com capacidade
temporal verificada e pode ceder um bootstrap sem modelo a outro candidato
pronto; os modos combinados de 900 s mantêm a seleção homologada do scanner. `training_registry.py`
preserva bundles compatíveis por símbolo/hash em `.training_inputs/models/`;
`shadow_monitor.py` retoma estados completos da identidade exata a partir de
`shadow_state.json.states/`. A seleção por dados não interrompe um bundle ativo
em shadow. A política de avaliação 4 exige reavaliar artefatos anteriores.

## Responsabilidades

| Área | Módulos e responsabilidade |
|---|---|
| Configuração e sessão | `config.py` concentra parâmetros; `deriv_session.py` gerencia autenticação Deriv legada e Bearer/PAT. |
| Dados e features | `collector.py` coleta; `market_data.py` valida e persiste; `asset_history.py` mantém candles compartilhados por ativo; `indicators.py` calcula indicadores e Price Action; `feature_engine.py` calcula o mapa canônico de 27 features; `dataset_builder.py` constrói datasets temporais. |
| Modelos | `train_model.py`, `training_evaluation.py` e `training_experiments.py` treinam/avaliam candidatos; `model_contract.py` valida compatibilidade e persistência de bundles. |
| Modelo de Treino isolado | `training_data.py` prepara snapshots de 28 dias; `training_history.py` complementa dados públicos; `training_candles.py` mantém o buffer privado de 5 minutos; `training_policy.py` controla progresso e capacidade; `training_registry.py` reserva bundles; `shadow_monitor.py` e `training_canary.py` fazem a validação forward. |
| Decisão e execução | `ai_predictor.py` faz inferência; `strategy.py` executa módulos independentes e exige maioria estrita na mesma direção; `risk_manager.py` controla limites persistentes; `executor.py` acompanha proposals e contratos; `pipeline.py` orquestra o fluxo. |
| Simulação e interface | `backtest.py` executa simulações offline; `dashboard/server.py` serve API e frontend para monitoramento e ações autorizadas. |
| Verificação | `tests/` cobre contratos, estratégia, risco, pipeline, execução, treino e endpoints do dashboard. |

## Dados e limites

Os contratos atuais usam arquivos v2, como `ticks_v2.csv`, `dataset_v2.csv` e
`operacoes_v2.csv`, além de metadados e estado operacional. Caminhos de estado
permanecem junto aos entrypoints conforme o código atual; movê-los requer uma
mudança explícita de compatibilidade e validação, não apenas reorganização de
pastas.

`transformer_model.py` não faz parte da implementação atual. Menções a esse
módulo, TFT ou Stacking foram preservadas somente em
[documentação arquitetural antiga](docs/historico/ARQUITETURA_ANTIGA.md). Modelos
legados não autorizam promoção nem operação. Backtests são offline e não
substituem validação operacional.

Para instruções de manutenção, consulte [AGENTS.md](AGENTS.md), o
[guia técnico](docs/GUIA_TECNICO.md) e o [README](README.md). Mudanças em
autenticação, risco, execução ou persistência exigem cobertura direcionada e
preservação dos arquivos operacionais.
