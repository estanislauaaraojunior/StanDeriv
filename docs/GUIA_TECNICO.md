# Guia técnico para tarefas no StanDeriv

Consulte somente as áreas afetadas pela tarefa. Caminhos abaixo são relativos à raiz. Leia a implementação antes de tomar os nomes ou descrições como contratos definitivos.

Score Ponderado e coleta de dados estão homologados e congelados desde
15/10/2026. A leitura do [registro de homologação](HOMOLOGACAO_SCORE_COLETA_2026-10-15.md)
é obrigatória antes de tocar essas áreas ou seus consumidores. Não altere seus
resultados por refatoração, ajuste de parâmetros ou atualização de dependências.

## Mapa de leitura e validação

| Área da tarefa | Código a consultar | Testes existentes a consultar |
| --- | --- | --- |
| Configuração e limites | `config.py` e consumidores dos parâmetros alterados | `tests/test_config_thresholds.py` |
| Indicadores e sinais | `indicators.py`, `strategy.py`, `feature_engine.py`, `docs/SCORE_PONDERADO_BOLLINGER.md` | `tests/test_indicators.py`, `tests/test_strategy.py`, `tests/test_strategy_edge_cases.py`, `tests/test_weighted_bollinger.py`, `tests/test_entry_recovery.py` |
| Coleta, features e modelos | `market_data.py`, `model_contract.py`, `collector.py`, `training_data.py`, `training_history.py`, `dataset_builder.py`, `feature_engine.py`, `train_model.py`, `training_evaluation.py`, `training_experiments.py`, `payout_quotes.py`, `ai_predictor.py` | `tests/test_training_history.py`, `tests/test_training_data.py`, `tests/test_training_evaluation.py`, `tests/test_contract_regressions.py`, `tests/test_indicators.py`, `tests/test_pipeline_integration.py`, `tests/test_tick_backfill.py`; conferir a cobertura real e criar regressão específica se necessário |
| Histórico OHLC por ativo | `asset_history.py`, `market_data.py`, `pipeline.py`, `executor.py` | `tests/test_tick_backfill.py`, `tests/test_executor_trade_flow.py` |
| Calendário e pausas de mercado | `market_calendar.py`, `asset_history.py`, `pipeline.py`, `executor.py` | `tests/test_market_calendar.py`, `tests/test_tick_backfill.py`, `tests/test_executor_trade_flow.py` |
| Keepalive e recuperação de lacunas | `api_keepalive.py`, `asset_history.py`, `executor.py`, `pipeline.py`, `request_throttle.py`, `operational_diagnostics.py` | `tests/test_history_recovery.py`, `tests/test_entry_recovery.py`, `tests/test_executor_trade_flow.py`, `tests/test_pipeline_live_feed.py`, `tests/test_scan_rate_limit.py`, `tests/test_operational_diagnostics.py` |
| Status por sessão e aquecimento | `runtime_status.py`, `executor.py`, `pipeline.py`, `dashboard/server.py` | `tests/test_runtime_status.py`, `tests/test_executor_trade_flow.py`, `tests/test_server_endpoints.py` |
| Autenticação e execução | `deriv_session.py`, `executor.py`, `bot.py`, `config.py` | `tests/test_deriv_session.py`, `tests/test_executor_trade_flow.py` |
| Modelo de Treino isolado (5 min), shadow e canary | `training_data.py`, `training_history.py`, `training_candles.py`, `training_policy.py`, `training_registry.py`, `shadow_monitor.py`, `training_canary.py`, `ai_predictor.py`, trechos da IA em `pipeline.py`/`executor.py` | `tests/test_ai_timeframe.py`, `tests/test_ai_timeframe_executor.py`, `tests/test_ai_training_bootstrap.py`, `tests/test_ai_prediction_cache.py`, `tests/test_training_history_incremental.py`, `tests/test_training_policy.py`, `tests/test_training_progress.py`, `tests/test_training_scheduler_policy.py`, `tests/test_training_candidate_selection.py`, `tests/test_training_registry.py`, `tests/test_shadow_monitor.py`, `tests/test_shadow_resume.py`, `tests/test_training_canary.py`, `tests/test_canary_early_rejection.py` |
| Avaliação e estudos de treino | `train_model.py`, `training_evaluation.py`, `training_experiments.py`, `payout_quotes.py` | `tests/test_training_evaluation.py`, `tests/test_training_recommendations.py` |
| Risco e persistência | `risk_manager.py`, `executor.py` | `tests/test_risk_manager.py`, `tests/test_executor_trade_flow.py` |
| Orquestração e seleção de símbolos | `pipeline.py`, `collector.py`, `executor.py`, `config.py` | `tests/test_pipeline_integration.py`, `tests/test_pipeline_live_feed.py`, `tests/test_market_failover.py`, `tests/test_executor_trade_flow.py` |
| Backtest | `backtest.py`, `strategy.py`, `risk_manager.py`, `ai_predictor.py` | `tests/test_backtest.py` |
| Dashboard | `dashboard/server.py`, `dashboard/app.js`, `dashboard/index.html`, `dashboard/style.css`, `dashboard/start.sh` | `tests/test_server_endpoints.py`, `tests/test_backtest.py` quando envolver simulação |

## Execução, risco e orquestração

O código atual mantém `ENFORCE_RISK_GATE=False`: pausas e stops são registrados,
mas não impedem novas ordens. A homologação de Score/coleta não aprova essa
política. Não deduza bloqueio de ordens somente do estado de pausa no painel.
O executor e o backtest recebem fechamentos de candles para o Score; tick é
um evento de avaliação, não a unidade dos indicadores operacionais.

Verifique o ciclo proposta → compra → acompanhamento → fechamento, inclusive erros, timeouts, mensagens repetidas e reconexão. Preserve a coerência entre contrato ativo, símbolo, duração, saldo e registro de resultado. Não flexibilize stops, stake, pausas ou filtros apenas para aumentar entradas ou fazer testes passarem.

Ao alterar seleção de símbolos, examine coleta, buffers de velas, features, inferência e envio de ordens. Ao alterar autenticação, confira `deriv_session.py` e use documentação oficial da Deriv se o contrato externo precisar ser confirmado; nunca deduza o tipo real da conta apenas da opção local de modo demo.

## Dados, treinamento e avaliação

Mantenha `feature_engine.py` como fonte compartilhada de features. Mudanças em nomes, ordem, normalização ou quantidade exigem revisar consumidores e compatibilidade com os modelos serializados. Explicite eventual necessidade de retreino, sem substituir artefatos em uso como efeito colateral de manutenção.

Preserve a ordem temporal na avaliação, evite vazamento de dados futuros e separe períodos de treino e avaliação. Compare métricas com período, amostra, payout, custos e configuração identificados. Para backtest, confira entrada após o sinal e ausência de uso de dados futuros. Ganhos históricos não demonstram rentabilidade futura.

## Dashboard

Confira validação de parâmetros, autenticação das rotas de controle e tratamento de erros no backend e frontend. Para ações que iniciam subprocessos, verifique exclusão entre execuções incompatíveis, caminhos de saída e encerramento. Prefira o cliente de testes Flask para endpoints; mudanças visuais exigem inspeção de apresentação quando houver ambiente disponível.

## Validação e proteção de artefatos

O repositório possui testes com pytest e unittest. Prefira o executor isolado,
que copia somente código e testes para `/tmp`, exclui `.env`, dados, modelos e
estados, usa token fictício e bloqueia a rede com `unshare -rn`:

```bash
bash .claude/skills/standeriv-testes-isolados/scripts/run_isolated.sh tests/test_strategy.py tests/test_strategy_edge_cases.py -q
bash .claude/skills/standeriv-testes-isolados/scripts/run_isolated.sh   # suíte completa
```

Em 26/11/2026 a suíte completa teve 831 testes aprovados em cerca de 90 segundos
nesse executor (Python 3.14.4). Sem ele, a forma direta equivalente é
`.venv/bin/python -m pytest tests -q`, apenas em cópia isolada.

Para conferir a homologação, `python3 .claude/skills/standeriv-homologacao/scripts/check_homologacao.py`
compara a tabela de hashes e, por AST, as definições alteradas desde o commit de
homologação. `executor.py`, `pipeline.py`, `model_contract.py` e
`runtime_status.py` já divergem da tabela por mudanças exclusivas da IA isolada;
os módulos congelados e os trechos de coleta do pipeline seguem idênticos.

Antes de executar, confira fixtures, mocks e caminhos de escrita dos testes escolhidos. A presença de um teste não garante isolamento. Use diretório temporário ou checkout isolado quando necessário. O pytest está declarado em `requirements-dev.txt`. Se faltar uma dependência, relate ou instale conforme as permissões e o escopo da sessão.

Não execute `bot.py`, `pipeline.py`, `collector.py` ou `dashboard/start.sh` como teste genérico. Não execute `dataset_builder.py` ou `train_model.py` apenas para validar imports: podem produzir ou substituir artefatos.

Trate `.env`, `ticks.csv`, `dataset.csv`, `dataset.csv.tmp`, `operacoes_log.csv`, `state.json`, `risk_state.json`, `model_metrics.json`, arquivos `.pkl`, logs, PIDs e relatórios existentes como dados operacionais. Prefira amostras pequenas e saídas temporárias quando precisar examiná-los. Não os inclua em alterações documentais ou commits de código por acidente.

Para alterações apenas em Markdown, confira caminhos, links, comandos citados e `git diff --check`; não é necessário executar a suíte da aplicação. Ao renomear módulos ou testes, atualize este mapa na mesma entrega.

## Contratos v2

Consulte `docs/CORRECOES_CONTRATOS_2026-09-08.md` para recuperação, esquema temporal e limitações. Novos dados usam `ticks_v2.csv`, `dataset_v2.csv` e `operacoes_v2.csv`; preserve também esses arquivos, metadados, `recovered_data/`, `risk_accounts.json` e `executor.lock`. A origem `legacy_unknown` não pode ser promovida a tick por inferência de espaçamento. Treino e inferência usam `model_contract.py` com janela de 50 candles OHLC; o caminho técnico do executor exige aquecimento de 150 candles, e a IA isolada usa 50 candles privados de 5 minutos. Mudanças de esquema exigem nova versão e validação; a homologação limita alterações de Score/coleta. A suíte dispõe de `tests/conftest.py` para isolar artefatos, mas novas fixtures continuam exigindo inspeção.

O contrato v4 usa 50 candles por amostra: 5 minutos na IA isolada e 15 minutos nos modos combinados. A preparação da IA usa até 28 dias (8.064 ou 2.688 posições, respectivamente), independentemente dos sete dias/672 posições da coleta e do limite de 336 candles do histórico compartilhado por ativo. O piso estrutural é de 593 amostras em 5 minutos e 579 em 15 minutos; o trainer rejeita partições, classes ou operações insuficientes sem ampliar essa janela automaticamente. As features experimentais de contexto ficam em `feature_engine.py`, são gravadas pelo dataset e avaliadas apenas no development por `training_experiments.py`; não alteram as 27 features operacionais. `payout_quotes.py` consulta propostas sem comprar; `QuoteHistory` rejeita uso de cotações futuras ou com mais de 900 segundos. Preserve `payout_quotes.csv`, `shadow_state.json` e `training_reports/` como artefatos.

O estudo `training_experiments.py --study training` exige diretório novo, captura
CSV/metadados/cotações e chama o treinador sem acesso ao champion. O holdout só
é avaliado depois da escolha do estimador/calibrador no development; o estudo
padrão de features continua sem avaliar o holdout. Consulte também
`tests/test_training_recommendations.py` para calibração, baselines, seleção,
serialização e isolamento. O calendário simulado é
`model_contract.EVALUATION_RISK_TIMEZONE`, registrado separadamente do fuso local
operacional. As curvas de limiar são diagnósticas e preservam o piso de confiança.


## Recuperação exclusiva da IA

`training_history.py` consulta somente `ticks_history` no endpoint público,
sem autenticação ou uso dos sockets operacionais. `training_data.prepare_snapshot`
recebe o complemento antes de gravar o snapshot e preserva os vetos temporais.
O pipeline integra esse caminho apenas no diagnóstico e na preparação do treino.
O cache `.training_inputs/history/` tem lock próprio, hash por ativo/timeframe,
retenção de até 28 dias, orçamento de consultas/tempo, paginação e backoff.
Não redirecione seus caminhos para arquivos da coleta nem use seus dados para
substituir buffers operacionais. Respostas vazias, parciais, fora de faixa ou
inválidas não comprovam continuidade; os filtros do dataset continuam valendo.

Use `tests/test_training_history.py` para regressões de lacunas na inicialização,
retomada, rate limit, correlação, payloads inválidos, quarentena e isolamento.
Os testes devem injetar o cliente público simulado; a fixture padrão desativa
o provedor do pipeline para impedir acesso à rede. `feed_diagnostics` apenas
resume o trecho final do log; a latência de consultas históricas é uma métrica
separada, sem medir o custo dos callbacks do coletor.

O bootstrap da IA usa `training_policy.py`: antes de ajustar estimadores,
verifica se validação e teste comportam ao menos 30 contratos sem sobreposição.
É um limite superior, sem garantia de sinais ou aprovação. Reprovações somente
por amostra permitem novo lote de 5% (mínimo 30, máximo 120); reprovações de
desempenho mantêm 120 novos rótulos. O status preserva o motivo da última
reprovação e informa o alvo da próxima tentativa. A recuperação privada da IA
alterna a primeira lacuna consultada entre rodadas, mantendo orçamento e backoff.
Na IA isolada, a parada por prontidão da recuperação verifica também o progresso
persistido do ativo e o lote retornado por `retry_plan`. Não usar apenas
`readiness_capacity`: uma amostra antiga treinável pode não ter nenhum ID novo.
O mesmo predicado vale antes da primeira consulta e entre os lotes do bootstrap;
consultar progresso não pode consumir rótulos nem incrementar revisão.

O progresso da IA em `.training_inputs/progress.json` usa identidades de rótulos,
não a diferença do total na janela móvel. `TrainingProgress.mark_attempt` recebe
exatamente os IDs do dataset treinado; observações e diagnósticos não consomem
rótulos. Preserve revisão, isolamento por ativo/política e compactação de IDs
antigos. `readiness_capacity` reutiliza as partições temporais do treinador,
sem ajustar modelos. Consulte `tests/test_training_scheduler_policy.py` e
`tests/test_training_candidate_selection.py` para janela cheia, retomada,
capacidade e transferência do bootstrap para candidato apto. A priorização e as
trocas por prontidão valem só para a IA isolada; nos modos combinados de 900 s
a seleção homologada do scanner é preservada (regressões em
`tests/test_training_candidate_selection.py` e `tests/test_market_failover.py`).
`tests/test_training_diagnostic_cache.py` cobre progresso corrompido e cache de
diagnósticos inválidos; `tests/test_training_snapshot_cleanup.py`, a remoção da
cópia de preços dos snapshots de treino.

`training_registry.py` guarda bundles aprovados em `.training_inputs/models/`,
por símbolo/hash; a ativação valida o payload antes da cópia. Consulte
`tests/test_training_registry.py`. A política de avaliação 4 identifica os
geradores habilitados e dispensa o score técnico combinado apenas na IA isolada;
políticas anteriores precisam de reavaliação, sem alterar as features ou a coleta.

`shadow_monitor.py` preserva estados completos em `shadow_state.json.states/` e
restaura somente schema, símbolo, modelo, timeframe e duração idênticos. Archives
legados parciais não autorizam restauração. `saved_state` é uma consulta sem
escrita usada para evitar retomar bundles reprovados. Consulte
`tests/test_shadow_resume.py`; preserve também `tests/test_shadow_monitor.py`.
Esses estados são artefatos operacionais. A seleção por dados não interrompe um
bundle ativo em shadow nem uma reconciliação pendente; os fluxos homologados
do scanner, coletor e Score permanecem preservados.

## Perfil de 5 minutos exclusivo da IA isolada

O canary de `training_canary.py` é exclusivo do perfil 300/900 da IA isolada.
Sua política versionada exige holdout elegível com pelo menos 30 contratos e
8 resultados forward novos, 7 acertos e saldo unitário positivo. Não baixar
`SHADOW_MIN_RESULTS`: ele continua em 30 para a validação original e os outros
modos. O canary usa `shadow_state.json.canary/`, separado do estado schema 1;
a validação original segue acumulando sem importar resultados anteriores para
a sequência curta. O ledger impede readmissão de períodos históricos
sobrepostos em outros bundles. Reprovação é terminal por bundle/política;
estado inválido falha fechado. O pipeline deve consultar também esse ledger
ao selecionar ou promover bundles. Preserve arquivos existentes e pendências.

A decisão terminal também ocorre se `acertos + restantes < mínimo`: somente
reprovação pode ser antecipada. O schema 2 aceita esse término com motivo e
época do primeiro resultado que tornou a meta impossível. Consultas somente
leitura devem reconhecer estados antigos inviáveis sem regravá-los; o monitor
persiste a normalização quando carregar o arquivo. A política e sua identidade
permanecem as mesmas, pois o conjunto de sequências aprovadas não muda.
Pendências simuladas de um canary encerrado ficam preservadas, sem continuar
alterando seus resultados. Contratos financeiros seguem a reconciliação
existente antes de qualquer troca de ativo. Na inicialização da IA isolada,
o pipeline não recomeça bootstrap com o bundle ativo já reprovado; busca outro
mercado quando não há contrato financeiro pendente. Solicitações explícitas
de retreino/coleta nova preservam seu fluxo próprio.

`tests/test_canary_early_rejection.py` verifica as 256 sequências de oito
resultados, retomada após cada prefixo, migração, pendências simuladas e
financeiras, persistência terminal, callback único e seleção no startup.

Consulte `tests/test_training_canary.py` para 7/8 versus 6/8, congelamento,
retomada, corrupção, embargo, reutilização de holdout e gates de demo/entrada.
`tests/test_ai_prediction_cache.py` cobre a inferência por janela fechada:
causalidade antes do cache, invalidação por dados/modelo/limiar e preservação
da inferência sem cache dos modos combinados. O tempo publicado em
`training_callback_timing` cobre somente `_handle_training_tick`, não o
callback inteiro; não atribuir causas de timeout sem medição adicional.
`tests/test_runtime_status.py` cobre a proteção dos candles privados contra
sobrescrita do heartbeat do pipeline. O diagnóstico de feed mantém os campos
do coletor e adiciona `executor_ping_timeouts` sem alterar os callbacks de coleta.

`model_contract.model_timeframe_sec()` retorna 300 somente com IA habilitada e
Score/Price Action/padrões desligados; os demais modos usam a base operacional
de 900 segundos. Não altere `CANDLE_TIMEFRAME_SEC` para ajustar o modelo.
Treino, inferência, progresso e shadow precisam da mesma identidade 300/900.
O perfil exige 50 candles fechados e piso estrutural de 593 rótulos, sem dispensar
capacidade, classes, embargo ou as divisões internas do treinador.

`training_history.py` reutiliza apenas ticks de caches de outro timeframe com
hash validado. `enrich_incremental` encadeia lotes com teto total, cancelamento,
backoff e parada sem progresso. `cached_records` lê sob lock compartilhado;
busy não autoriza ler CSV/manifest inconsistentes. `recovery_start` limita
somente consultas, preservando a retenção de 28 dias.

`training_candles.py` fornece o contexto privado de 5 minutos ao executor,
incluindo gaps, quarentena e fechamento causal. Sua leitura e complemento
ocorrem antes do WebSocket ou em worker; `feed/window` usam memória. O caminho
técnico de 900 segundos, aquecimento de 150 candles e coleta homologada devem
permanecer com os mesmos resultados. Consulte `tests/test_ai_timeframe.py`,
`tests/test_ai_timeframe_executor.py`, `tests/test_ai_training_bootstrap.py` e
`tests/test_training_history_incremental.py`, além das regressões de execução.

A origem ambígua não é corrigida por adivinhação. O pipeline pode preparar uma
base independente exclusivamente da IA, sem a origem inteira e com hash/motivo
registrados, usando somente registros públicos/cache verificados. Não remova
vetos de épocas nem edite os CSVs operacionais para fazer a preparação passar.
