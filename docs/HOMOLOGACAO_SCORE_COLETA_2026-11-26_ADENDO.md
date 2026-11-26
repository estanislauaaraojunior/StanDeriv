# Adendo à homologação de Score Ponderado e coleta — 26/11/2026

Complementa o [registro de 15/10/2026](HOMOLOGACAO_SCORE_COLETA_2026-10-15.md),
que permanece válido e inalterado. Este adendo atualiza somente a referência de
integridade dos arquivos **compartilhados** que receberam código exclusivo do
Modelo de Treino isolado (perfil 300/900). A regra continua a mesma: Score
Ponderado e coleta estão congelados e não podem ser alterados sem nova
autorização explícita.

## Aceitação

Em 26/11/2026 (America/Sao_Paulo), o usuário autorizou registrar este adendo
após a revisão de código e as correções do commit `a8d2e2d`. A aceitação é
funcional e restrita à preservação do Score e da coleta; não homologa modelos,
política de risco, rentabilidade ou execução real.

## O que mudou nos arquivos compartilhados

Comparação por AST (sem docstrings) contra o commit de homologação `20f9293`,
gerada por `.claude/skills/standeriv-homologacao/scripts/check_homologacao.py`:

| Arquivo | Definições alteradas | Definições novas |
|---|---|---|
| `executor.py` | `DerivBot.__init__`, `run`, `_handle_tick`, `_request_missing_history_or_subscribe`, `_write_runtime_status`, `_capture_shadow_prediction`, `_get_shadow_monitor`, `_publish_shadow_status` | `_evaluate_entry`, `_get_training_candles`, `_handle_training_tick`, `_request_training_history_refresh`, `_uses_training_candles` |
| `pipeline.py` | `_main_session`, `main`, `_wait_for_ticks`, `_wait_for_compatible_model`, `_run_training`, `_retrain_loop`, `_should_retrain_now`, `_count_labelable_contracts_for_training`, `_load_latest_final_metrics`, `_first_live_symbol_with_compatible_model`, `_active_model_shadow_approved` | funções `_training_*`, `_diagnose_training_symbol`, `_prepare_model_snapshot`, `_recover_model_candles`, `_maybe_request_training_candidate`, `_prioritize_training_candidates`, `_request_rejected_canary_switch`, `_discard_snapshot_bulk` e estados `_training_*` |
| `model_contract.py` | `specification`, `minimum_training_rows` | `model_timeframe_sec` |
| `runtime_status.py` | `update_status` | — |

Evidências de preservação do Score e da coleta:

- Os 12 módulos congelados (`strategy.py`, `indicators.py`, `config.py`,
  `collector.py`, `market_data.py`, `asset_history.py`, `market_calendar.py`,
  `api_keepalive.py`, `request_throttle.py`, `deriv_session.py`,
  `feature_engine.py`, `operational_diagnostics.py`) têm os mesmos hashes de
  15/10/2026.
- Em `pipeline.py`, `_CollectorThread`, `_fetch_historical_ticks`, `_scan_group`,
  `_detect_top_trending_symbols`, `_validate_deriv_token` e `_history_store`
  estão idênticos.
- Em `executor.py`, as mudanças de `run`, `_handle_tick`,
  `_request_missing_history_or_subscribe` e `_write_runtime_status` são desvios
  condicionados a `_uses_training_candles()` (IA isolada). O bloco de avaliação
  de entrada foi movido de `_handle_tick` para `_evaluate_entry` sem alteração,
  exceto receber `candles` e `candle_closes` como parâmetros (as mesmas cópias
  em lista usadas antes). O caminho técnico de 900 s e o aquecimento de 150
  candles permanecem.
- `model_contract.model_timeframe_sec()` retorna 900 sempre que Score, Price
  Action ou padrões estão habilitados; `CANDLE_TIMEFRAME_SEC` não mudou.
- A priorização de candidatos e as trocas por prontidão de dados da IA foram
  restritas à IA isolada no commit `a8d2e2d`; os modos combinados usam o
  primeiro candidato do scanner, como em `20f9293`.

## Verificações

Executadas em 26/11/2026 em cópia isolada (sem `.env`, dados ou modelos
operacionais, rede bloqueada), Python 3.14.4:

- Regressões de Score e coleta (`test_weighted_bollinger`, `test_strategy`,
  `test_strategy_edge_cases`, `test_indicators`, `test_entry_recovery`,
  `test_tick_backfill`, `test_history_recovery`, `test_market_calendar`,
  `test_scan_rate_limit`, `test_pipeline_live_feed`, `test_runtime_status`):
  **184 aprovados**.
- Suíte completa: **848 aprovados**.

Nenhum serviço foi iniciado ou reiniciado, nenhuma conta foi consultada e nenhum
modelo foi treinado ou promovido para este adendo. A conferência por AST e os
testes demonstram equivalência estrutural e funcional nos caminhos cobertos;
não substituem observação operacional.

## Referência de integridade vigente

Esta tabela substitui a de 15/10/2026 como referência para
`check_homologacao.py`, que compara com o commit que introduziu este adendo.

| Arquivo | SHA-256 |
|---|---|
| `strategy.py` | `4e7f8e704df296e298f70845ca45352965f6b5e70f83f3402525d132f46dec06` |
| `indicators.py` | `dec0ee29f5bc5c7ba82e6d4c0f3c5b4c5f88ff1169bf4c1e2ce981a1a932217e` |
| `config.py` | `84481391cbe0c9d2e57e31bdd387de355a29254a4619e04de31fba7a6142a4bc` |
| `collector.py` | `7e82ef5d6771e73312c4f28afdfe675b73fdd76fdd5be1769e2d85cacdb2e8ee` |
| `pipeline.py` | `db8a4076639d6fffbd45f1d457e81dcb7c0605bb80b82f637c1ee986ae3139f9` |
| `market_data.py` | `df5170f7dca203d7e37a55a05f77b9e5077f27e1b5de33377e3b451ba349cb62` |
| `asset_history.py` | `f26c39d4c5947f8206a2537fbdffe9d4d08e28794dee065d0fa783c7c17a43aa` |
| `market_calendar.py` | `a4f6218e78164520a61afe964d458aa220377d85120578dd6c640001420339ac` |
| `executor.py` | `68f7a0d54968c247ffa61ca8aca370db774db071a6ff1aa2b5220f2a3c7a97a3` |
| `api_keepalive.py` | `10353f5c9e7519b9e72f4ea62900f1d1ceed91b8e3d22ec688c8dd431b4b1a04` |
| `request_throttle.py` | `f0f9d9e296d9a14ebb08afe6c5c5664ea662a6e769505f0cb49c9aec9faf12c2` |
| `deriv_session.py` | `894dd6e5b09fe6e459d3fbcadd7705894fb873a4a0acfadd4ad11d107d8f7a48` |
| `runtime_status.py` | `d901dbd9001a33f359413374759aabf3105fd05c07a9b0e81cbfc29a7600a71d` |
| `operational_diagnostics.py` | `ef61b773b3606bbd35b59ec6249949443f03379ab64de188248d790afbb67e17` |
| `feature_engine.py` | `9872f5e3af3498397245e8a7978c7a5c96f4633103f02a97fbb6acc0669e7fd2` |
| `model_contract.py` | `a39af4e431bef05b0254f00ebb984edc9a89f44e80d41d5e0fa267105a621753` |
