---
name: standeriv-homologacao
description: Verifica se o Score Ponderado e a coleta de dados homologados em 15/10/2026 continuam preservados no StanDeriv — confere SHA-256 da tabela de homologação e compara por AST, com o commit de homologação, as funções alteradas. Use antes e depois de qualquer alteração em strategy.py, indicators.py, config.py, collector.py, market_data.py, asset_history.py, market_calendar.py, api_keepalive.py, request_throttle.py, deriv_session.py, feature_engine.py, executor.py, pipeline.py, model_contract.py ou runtime_status.py; antes de commits/PRs; e sempre que alguém perguntar se a homologação, o Score ou a coleta foram afetados.
---

# Conferência da homologação de Score e coleta

A homologação (docs/HOMOLOGACAO_SCORE_COLETA_2026-10-15.md) congelou o Score
Ponderado e a coleta: não alterar lógica, parâmetros, base temporal, filtros ou
persistência, nem indiretamente por arquivos compartilhados. O trabalho da IA
isolada precisa tocar `executor.py`, `pipeline.py`, `model_contract.py` e
`runtime_status.py`, então hash diferente nesses arquivos é esperado — o que
importa é *quais* definições mudaram e se alguma pertence à coleta/Score.

## Rodar

```bash
python3 .claude/skills/standeriv-homologacao/scripts/check_homologacao.py          # texto
python3 .claude/skills/standeriv-homologacao/scripts/check_homologacao.py --json   # para análise
python3 .claude/skills/standeriv-homologacao/scripts/check_homologacao.py --ref HEAD  # só o que mudou desde o último commit
```

Somente leitura: lê a tabela do documento, calcula hashes e usa `git show` do
commit que criou o documento; não importa módulos do projeto nem lê `.env`/dados.
Saída 1 (`REVISAR`) quando um arquivo congelado mudou além de docstrings ou
quando um trecho crítico do pipeline mudou (`_CollectorThread`,
`_fetch_historical_ticks`, `_scan_group`, `_detect_top_trending_symbols`,
`_validate_deriv_token`, `_history_store`).

## Interpretar

- `OK`: idêntico ao homologado.
- `DOCSTRING`: só docstrings/formatação mudaram; comportamento preservado.
- `ALTERADO [congelado]`: pare. Só prossiga com autorização explícita do
  usuário para rever a homologação; caso contrário, reverta *sua* alteração
  (nunca alterações preexistentes de terceiros).
- `ALTERADO [compartilhado]`: leia o diff das definições listadas
  (`git diff <ref> -- arquivo`) e confirme que só afetam o caminho da IA
  isolada (300/900, `training_*`, shadow/canary) e que os caminhos técnicos de
  900 s, aquecimento de 150 candles e coleta permanecem iguais. Funções como
  `DerivBot._handle_tick`, `_wait_for_ticks` e `main` merecem leitura linha a linha.

A conferência por AST não prova equivalência comportamental quando o código
congelado chama funções compartilhadas que mudaram; complemente com
`standeriv-testes-isolados` (tests/test_weighted_bollinger.py,
test_strategy*.py, test_indicators.py, test_entry_recovery.py,
test_tick_backfill.py, test_history_recovery.py).

Ao relatar, liste os arquivos OK/alterados, as definições relevantes e a sua
conclusão, separando o que foi verificado do que é inferência.
