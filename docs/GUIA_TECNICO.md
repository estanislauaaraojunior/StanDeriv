# Guia técnico para tarefas no StanDeriv

Consulte somente as áreas afetadas pela tarefa. Caminhos abaixo são relativos à raiz. Leia a implementação antes de tomar os nomes ou descrições como contratos definitivos.

## Mapa de leitura e validação

| Área da tarefa | Código a consultar | Testes existentes a consultar |
| --- | --- | --- |
| Configuração e limites | `config.py` e consumidores dos parâmetros alterados | `tests/test_config_thresholds.py` |
| Indicadores e sinais | `indicators.py`, `strategy.py`, `feature_engine.py` | `tests/test_indicators.py`, `tests/test_strategy.py`, `tests/test_strategy_edge_cases.py` |
| Coleta, features e modelos | `market_data.py`, `model_contract.py`, `collector.py`, `dataset_builder.py`, `feature_engine.py`, `train_model.py`, `transformer_model.py`, `ai_predictor.py` | `tests/test_contract_regressions.py`, `tests/test_indicators.py`, `tests/test_pipeline_integration.py`; conferir a cobertura real e criar regressão específica se necessário |
| Autenticação e execução | `deriv_session.py`, `executor.py`, `bot.py`, `config.py` | `tests/test_executor_trade_flow.py`; autenticação pode exigir cobertura adicional com mocks |
| Risco e persistência | `risk_manager.py`, `executor.py` | `tests/test_risk_manager.py`, `tests/test_executor_trade_flow.py` |
| Orquestração e seleção de símbolos | `pipeline.py`, `collector.py`, `executor.py`, `config.py` | `tests/test_pipeline_integration.py`, `tests/test_executor_trade_flow.py` |
| Backtest | `backtest.py`, `strategy.py`, `risk_manager.py`, `ai_predictor.py` | `tests/test_backtest.py` |
| Dashboard | `dashboard/server.py`, `dashboard/app.js`, `dashboard/index.html`, `dashboard/style.css`, `dashboard/start.sh` | `tests/test_server_endpoints.py`, `tests/test_backtest.py` quando envolver simulação |

## Execução, risco e orquestração

Verifique o ciclo proposta → compra → acompanhamento → fechamento, inclusive erros, timeouts, mensagens repetidas e reconexão. Preserve a coerência entre contrato ativo, símbolo, duração, saldo e registro de resultado. Não flexibilize stops, stake, pausas ou filtros apenas para aumentar entradas ou fazer testes passarem.

Ao alterar seleção de símbolos, examine coleta, buffers de velas, features, inferência e envio de ordens. Ao alterar autenticação, confira `deriv_session.py` e use documentação oficial da Deriv se o contrato externo precisar ser confirmado; nunca deduza o tipo real da conta apenas da opção local de modo demo.

## Dados, treinamento e avaliação

Mantenha `feature_engine.py` como fonte compartilhada de features. Mudanças em nomes, ordem, normalização ou quantidade exigem revisar consumidores e compatibilidade com os modelos serializados. Explicite eventual necessidade de retreino, sem substituir artefatos em uso como efeito colateral de manutenção.

Preserve a ordem temporal na avaliação, evite vazamento de dados futuros e separe períodos de treino e avaliação. Compare métricas com período, amostra, payout, custos e configuração identificados. Para backtest, confira entrada após o sinal e ausência de uso de dados futuros. Ganhos históricos não demonstram rentabilidade futura.

## Dashboard

Confira validação de parâmetros, autenticação das rotas de controle e tratamento de erros no backend e frontend. Para ações que iniciam subprocessos, verifique exclusão entre execuções incompatíveis, caminhos de saída e encerramento. Prefira o cliente de testes Flask para endpoints; mudanças visuais exigem inspeção de apresentação quando houver ambiente disponível.

## Validação e proteção de artefatos

O repositório possui testes com pytest e unittest. Se o ambiente `.venv` estiver disponível, um exemplo de execução direcionada é:

```bash
.venv/bin/python -m pytest tests/test_strategy.py tests/test_strategy_edge_cases.py -q
```

Para uma mudança transversal que justifique a suíte completa:

```bash
.venv/bin/python -m pytest tests -q
```

Antes de executar, confira fixtures, mocks e caminhos de escrita dos testes escolhidos. A presença de um teste não garante isolamento. Use diretório temporário ou checkout isolado quando necessário. Verifique a disponibilidade do pytest: ele não está declarado no `requirements.txt` atual. Se faltar uma dependência, relate ou instale conforme as permissões e o escopo da sessão.

Não execute `bot.py`, `pipeline.py`, `collector.py` ou `dashboard/start.sh` como teste genérico. Não execute `dataset_builder.py` ou `train_model.py` apenas para validar imports: podem produzir ou substituir artefatos.

Trate `.env`, `ticks.csv`, `dataset.csv`, `dataset.csv.tmp`, `operacoes_log.csv`, `state.json`, `risk_state.json`, `model_metrics.json`, arquivos `.pkl`, logs, PIDs e relatórios existentes como dados operacionais. Prefira amostras pequenas e saídas temporárias quando precisar examiná-los. Não os inclua em alterações documentais ou commits de código por acidente.

Para alterações apenas em Markdown, confira caminhos, links, comandos citados e `git diff --check`; não é necessário executar a suíte da aplicação. Ao renomear módulos ou testes, atualize este mapa na mesma entrega.

## Contratos v2

Consulte `docs/CORRECOES_CONTRATOS_2026-09-08.md` para recuperação, esquema temporal e limitações. Novos dados usam `ticks_v2.csv`, `dataset_v2.csv` e `operacoes_v2.csv`; preserve também esses arquivos, metadados, `recovered_data/`, `risk_accounts.json` e `executor.lock`. A origem `legacy_unknown` não pode ser promovida a tick por inferência de espaçamento. Treino e inferência usam `model_contract.py` e 100 candles OHLC; mudanças exigem nova versão de esquema e validação. A suíte dispõe de `tests/conftest.py` para isolar artefatos, mas novas fixtures continuam exigindo inspeção.
