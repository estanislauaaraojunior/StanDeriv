# StanDeriv — Bot de Trading Automatizado para Deriv

Bot de contratos CALL/PUT para a Deriv, com análise técnica, Price Action, modelos calibrados e execução com controle persistente de risco.

A manutenção de contratos v2 alterou o esquema dos dados e a validação dos modelos. Leia [Correções e migração](docs/CORRECOES_CONTRATOS_2026-09-08.md) antes de iniciar. Modelos legados não autorizam votos da IA. Com IA habilitada, a execução depende de bundle compatível e dos gates de modelo; com IA desligada, o Score Ponderado pode gerar entradas técnicas sem treinamento. A [auditoria das perdas](docs/historico/ANALISE_LOSSES_2026-09-08.md) é um registro histórico.

## Homologação de Score Ponderado e coleta — 15/10/2026

O usuário homologou o **Score Ponderado** e a **coleta de dados**, relatando
funcionamento sem erros e continuidade na abertura dos contratos. Esses módulos
estão congelados: não alterar cálculo, parâmetros, base temporal, filtros ou
persistência sem nova autorização explícita. Consulte o
[registro de homologação e escopo](docs/HOMOLOGACAO_SCORE_COLETA_2026-10-15.md) e a
[especificação do Score Ponderado](docs/SCORE_PONDERADO_BOLLINGER.md). A aprovação
é funcional; não constitui avaliação de rentabilidade nem homologação dos
controles de risco ou dos modelos treinados.

## Documentação

- [Arquitetura atual](ARCHITECTURE.md): responsabilidades e fluxos vigentes.
- [Homologação de Score e coleta](docs/HOMOLOGACAO_SCORE_COLETA_2026-10-15.md): escopo congelado; o [adendo de 26/11/2026](docs/HOMOLOGACAO_SCORE_COLETA_2026-11-26_ADENDO.md) traz a referência de integridade vigente.
- [Score Ponderado](docs/SCORE_PONDERADO_BOLLINGER.md): fórmulas, pesos, votos e base temporal.
- [Guia técnico](docs/GUIA_TECNICO.md) e [instruções de manutenção](AGENTS.md): leitura por área e regras para alterações.
- [Correções e migração](docs/CORRECOES_CONTRATOS_2026-09-08.md): registro cronológico; seções posteriores substituem políticas antigas.
- [Refatoração](docs/REFATORACAO.md) e [`docs/historico/`](docs/historico/INDICE.md): evidências das etapas anteriores, sem redefinir os contratos atuais.

## Instruções para assistentes de IA

O ponto de entrada é [AGENTS.md](AGENTS.md), que orienta a leitura do
[prompt mestre autônomo do StanDeriv](docs/PROMPT_STANDERIV_AUTONOMO.md), do
[prompt de engenharia histórico](docs/PROMPT_STANDERIV.md) quando necessário e
da consulta por área do [guia técnico](docs/GUIA_TECNICO.md). O prompt autônomo
pode ser copiado integralmente para um assistente com acesso ao repositório, sem
campos a preencher.

O Codex descobre `AGENTS.md` ao iniciar o trabalho no projeto, conforme a
[documentação oficial](https://developers.openai.com/codex/guides/agents-md).
Os outros documentos são lidos por instrução desse arquivo, conforme a tarefa;
arquivos Markdown arbitrários não são carregados automaticamente. Após alterar
essas instruções, inicie uma nova sessão no repositório ou peça à sessão atual
para reler `AGENTS.md`. Em outros assistentes, indique esse arquivo explicitamente
se a ferramenta não oferecer suporte a essa convenção.

No Claude Code, `.claude/settings.json` bloqueia a leitura de `.env` e a execução
direta de bot, pipeline, coletor e dashboard, e pede confirmação antes de editar
módulos homologados/compartilhados ou arquivos operacionais. As skills do projeto
ficam em `.claude/skills/`: `standeriv-testes-isolados` (pytest em cópia
temporária com rede bloqueada), `standeriv-homologacao` (hashes e AST dos módulos
homologados) e `standeriv-auditoria-status` (resumo somente leitura do estado).

---

## Arquitetura

```
┌────────────┐     ┌──────────────┐     ┌────────────────┐
│ collector   │────▶│ dataset_     │────▶│ train_model    │
│ (ticks_v2.csv) │     │ builder      │     │ (model.pkl)    │
└────────────┘     └──────────────┘     └────────────────┘
                                                │
┌────────────┐     ┌──────────────┐     ┌───────▼────────┐
│ executor   │◀───▶│ strategy     │◀───▶│ ai_predictor   │
│ (WebSocket) │     │ (sinais)     │     │ (inferência)   │
└────────────┘     └──────────────┘     └────────────────┘
       │                    │
       ▼                    ▼
┌────────────┐     ┌──────────────┐
│ risk_      │     │ indicators   │
│ manager    │     │ (técnicos +  │
│ (log CSV)  │     │  Price Action│
└────────────┘     └──────────────┘
       │
       ▼
┌──────────────────┐
│ dashboard/       │
│ (Flask + HTML/JS)│
└──────────────────┘

ticks_v2.csv ──▶ backtest.py ──▶ backtest_reports/ (JSON + CSV)
```

Para detalhes técnicos complementares da organização do sistema, veja
[ARCHITECTURE.md](ARCHITECTURE.md). Em caso de divergência, o contrato v2
descrito neste README, nos testes e em `docs/CORRECOES_CONTRATOS_2026-09-08.md`
prevalece sobre descrições históricas.

## Estrutura do repositório

Os módulos Python executáveis permanecem na raiz para preservar os comandos CLI,
os imports legados e os caminhos de dados calculados a partir de `__file__`.
`dashboard/` reúne a interface e o servidor Flask; `tests/` contém a suíte; e
`docs/historico/` reúne relatórios datados e evidências de pesquisa. Os guias e
instruções vigentes permanecem diretamente em `docs/`. Arquivos de dados,
modelos e estado operacional não foram movidos.

## Requisitos

- Python 3.11+ (a suíte foi validada em 26/11/2026 com Python 3.14.4)
- Conta na Deriv (demo ou real) com token de API
- Para desenvolvimento e testes: `pip install -r requirements-dev.txt` (`pytest`)

`torch` saiu das dependências em 26/11/2026: o TFT foi removido do fluxo v2 e
nenhum módulo atual o importa.

### Instalação

```bash
git clone https://github.com/estanislauaaraojunior/StanDeriv.git
cd StanDeriv
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

### Configuração

1. Crie o arquivo `.env` na raiz:

```env
DERIV_TOKEN=seu_token_aqui
DERIV_APP_ID=1089
# auto = detecta token legado, PAT (pat_...) ou OAuth Bearer (ory_at_...)
# use "bearer" se seu token novo nao tiver prefixo reconhecido
DERIV_AUTH_MODE=auto
```

2. Consulte `config.py` e as flags da sessão antes de iniciar. Os parâmetros de
   Score Ponderado e coleta estão congelados pela homologação; não os ajuste
   como manutenção rotineira.

### Autenticação Deriv

O bot aceita os dois fluxos da documentação atual:

- Token legado da WebSocket API: conecta em `wss://ws.derivws.com/websockets/v3?app_id=...` e envia `authorize`.
- PAT/OAuth Bearer: usa REST em `https://api.derivws.com` com `Deriv-App-ID` e `Authorization: Bearer`, seleciona/cria uma conta Options e obtém a URL WebSocket por OTP antes de operar.

Para PAT/OAuth, o token precisa do escopo `trade`. Se a conta Options ainda não existir, também precisa de `account_manage`. Evite usar `DERIV_APP_ID=1089` com o fluxo novo; crie um App ID próprio na Deriv.

## Uso Rápido

### Pipeline (coleta e execução condicionada à validação)

```bash
python pipeline.py --demo --balance 1000
```

Com `USE_AI_MODEL=True`, a execução exige modelo aprovado compatível e confirmação shadow. Sem IA, o pipeline preserva os artefatos, ignora treino/retreino e permite entradas pelos módulos técnicos habilitados. Com IA ativa e sem bundle compatível, o pipeline tenta treino quando há dados
suficientes; `--force-retrain` força nova tentativa mesmo com modelo existente. Nenhum candidato substitui o modelo ativo sem passar pela avaliação temporal. A opção `--demo` é confrontada com a conta autenticada.

Opções do pipeline: `--demo`/`--real`, `--balance`, `--min-ticks`, `--skip-collect`,
`--force-retrain`, `--history-count`, `--no-scan`, `--top-symbols` (até 5),
`--reuse-training-data` (reutiliza dados e modelos; por padrão inicia nova coleta
e treino) e `--fresh-collect` (arquiva também os ticks e baixa a janela de novo).
Os geradores podem ser escolhidos por sessão com as variáveis
`STANDERIV_USE_AI_MODEL`, `STANDERIV_USE_WEIGHTED_SIGNAL`,
`STANDERIV_PRICE_ACTION_FILTER` e `STANDERIV_CANDLE_PATTERN_FILTER`; o painel as
usa ao iniciar uma sessão. Confira as flags efetivas no status, não apenas o
campo textual `execution_signal_mode`.

### Apenas o bot (módulos configurados)

```bash
python bot.py
```

O entrypoint aplica as flags de decisão de `config.py`; não presume que a IA
esteja desligada. Iniciar qualquer comando conectado exige conferir a conta e
a autorização operacional.

### Apenas coletar ticks

```bash
python collector.py
```

### Backtesting offline

O backtester não se conecta à Deriv nem envia ordens. Por padrão, ele avalia
somente a estratégia técnica e usa duração fixa, evitando usar inadvertidamente
um modelo treinado no mesmo período analisado:

```bash
python backtest.py \
  --input ticks_v2.csv \
  --symbol R_100 \
  --balance 1000 \
  --payout 0.85 \
  --duration 15 \
  --output-dir backtest_reports
```

`--payout 0.85` significa lucro líquido de 85% do stake quando o contrato vence.
Custos opcionais podem ser simulados com `--cost-rate` e `--fixed-cost`. A saída
contém `backtest_report.json`, com configuração, fingerprints e métricas
comparáveis, e `backtest_trades.csv`, com cada contrato simulado.

Para avaliar o modelo v2 compatível com o ativo e a duração:

```bash
python backtest.py --with-ai --input ticks_ood.csv
```

Use essas opções apenas quando o arquivo representar um período realmente fora
da amostra, não utilizado no treinamento dos modelos. O motor usa somente dados
disponíveis até o fechamento da vela, exige um segmento contínuo de ticks identificados, entra no primeiro tick seguinte, impede
contratos simultâneos e aplica stake percentual, stops diários e pausas por
perdas sobre o relógio histórico.

Os stops usam dias em **UTC** por padrão, independentemente do fuso da máquina.
Para reproduzir o calendário de um processo operacional em São Paulo, informe
`--risk-timezone America/Sao_Paulo`. O resultado pertence ao dia da liquidação;
a virada do dia reinicia os limites diários, mas preserva pausas pendentes.
O relatório registra o fuso, o hash dos ticks selecionados, o stake médio,
sequências globais por P&L líquido e resultados agrupados pelo dia de liquidação.
O payout fixo é uma hipótese configurável. A simulação não modela latência,
slippage, requotes, indisponibilidade, falhas WebSocket ou rejeição de propostas.

### Dashboard

```bash
cd dashboard && bash start.sh
# ou: python dashboard/server.py
# Acesse: http://localhost:5055
```

A aba **Backtest** permite configurar e executar a simulação offline, acompanhar
o processo e visualizar métricas, curva de equity e operações. A execução usa o
mesmo `DASHBOARD_TOKEN` das demais ações de controle e só é permitida enquanto o
bot ao vivo estiver parado. Os relatórios ficam isolados por execução em
`backtest_reports/dashboard/`.

## Módulos

| Arquivo | Descrição |
|---------|-----------|
| `config.py` | Configurações centralizadas (indicadores, risco, IA, Price Action) |
| `indicators.py` | Indicadores técnicos puros (EMA, RSI, MACD, ADX, Bollinger, Momentum) + Price Action (estrutura de mercado, S/R, supply/demand, FVG) + detecção de 6 padrões de vela |
| `market_data.py` | Leitura explícita, recuperação não destrutiva, conflitos e proveniência |
| `model_contract.py` | Contrato temporal, compatibilidade de modelo, payout e persistência atômica |
| `feature_engine.py` | Fonte única para calcular o mapa das 27 features usado pelo treino e pela inferência |
| `dataset_builder.py` | Extrai features de ticks_v2.csv e gera dataset_v2.csv para treino |
| `train_model.py` | Treina candidatos RF/XGB com calibração e avaliação temporal separadas |
| `ai_predictor.py` | Inferência de bundle aprovado para ativo, OHLC e vencimento |
| `strategy.py` | Motor de decisão: sinais BUY/SELL com filtros técnicos, PA e IA |
| `executor.py` | Cliente WebSocket da Deriv, execução de ordens, notificações de velas |
| `risk_manager.py` | Gestão de risco: sizing, stop loss/take profit diário, pausa escalável por losses, detecção de drift |
| `pipeline.py` | Orquestrador: coleta, trend scanning, treino automático, execução e re-treino periódico |
| `collector.py` | Coletor autônomo de ticks; o pipeline usa seu próprio `_CollectorThread` |
| `asset_history.py` | Histórico OHLC por ativo, sincronização incremental e persistência atômica |
| `market_calendar.py` | Calendário por símbolo e confirmação de pausas de mercado |
| `api_keepalive.py` | Ping JSON e medição de respostas da API |
| `request_throttle.py` | Intervalo e backoff de requisições |
| `runtime_status.py` | Status concorrente por sessão, símbolo e proprietário dos campos |
| `operational_diagnostics.py` | Diagnóstico de conexão com ocultação de informações sensíveis |
| `backtest.py` | Simulação offline de sinais, duração, payout, custos, risco e drawdown, com relatórios JSON/CSV |
| `bot.py` | Entrypoint alternativo de execução com as flags de decisão de `config.py` |
| `payout_quotes.py` | Amostragem de propostas CALL/PUT sem compra e consulta causal de payout (`QuoteHistory`) |
| `training_evaluation.py` | Simulação temporal de contratos, métricas, calibração e elegibilidade usadas pelo treinador |
| `training_experiments.py` | Estudos offline (contexto, receita de treino, ablação) em diretório novo, sem publicar bundle |
| `training_data.py` | Snapshot de treino por ativo com janela de 28 dias, quarentena e vetos temporais |
| `training_history.py` | Complemento público de OHLC/ticks exclusivo da IA, com cache em `.training_inputs/history/` |
| `training_candles.py` | Buffer privado de candles de 5 minutos da IA isolada no executor |
| `training_policy.py` | Progresso por identidade de rótulos, capacidade temporal e lote de retreino |
| `training_registry.py` | Reserva de bundles aprovados por símbolo/hash em `.training_inputs/models/` |
| `shadow_monitor.py` | Validação forward original (30 resultados, 55%) com estados por identidade |
| `training_canary.py` | Canary `isolated_ai_canary_v1` (8 resultados, 7 acertos) com reprovação antecipada |
| `dashboard/` | Interface web com Flask (server.py), autenticação por token e frontend (HTML/JS/CSS) |

## Price Action (11 Features)

Features numéricas normalizadas derivadas de velas OHLC sintéticas:

| Feature | Range | Descrição |
|---------|-------|-----------|
| `pa_market_structure` | -1 a +1 | HH/HL (bullish) vs LH/LL (bearish) |
| `pa_bos_strength` | -1 a +1 | Força do break of structure |
| `pa_trend_consistency` | -1 a +1 | Sequência de swings alinhados |
| `pa_sr_distance` | 0 a 1 | Distância ao S/R mais próximo |
| `pa_sr_touch_count` | 0 a 1 | Força do nível S/R (toques) |
| `pa_sr_position` | -1 a +1 | Perto de suporte (-) vs resistência (+) |
| `pa_demand_zone` | 0 a 1 | Proximidade de demand zone |
| `pa_supply_zone` | 0 a 1 | Proximidade de supply zone |
| `pa_fvg_bullish` | 0 a 1 | Fair Value Gap bullish por preencher |
| `pa_fvg_bearish` | 0 a 1 | Fair Value Gap bearish por preencher |
| `pa_candle_at_sr` | -1 a +1 | Padrão de rejeição em zona S/R |

## Padrões de Vela (6 detectados com notificação)

1. **Bullish Engulfing** — Reversão de alta
2. **Bearish Engulfing** — Reversão de baixa
3. **Hammer** — Reversão de alta (sombra inferior longa)
4. **Shooting Star** — Reversão de baixa (sombra superior longa)
5. **Doji** — Indecisão (corpo mínimo)
6. **Three White Soldiers / Three Black Crows** — Continuação forte

Cada padrão inclui `strength` (0-1) e notificação em tempo real no terminal e dashboard.

## Modelos e contrato temporal

O **Modelo de Treino isolado** usa 50 candles fechados de **5 minutos** para
avaliar contratos de **15 minutos**. Treino e inferência compartilham esse
perfil de 300/900 segundos, separado do timeframe operacional. Score Ponderado,
Price Action, padrões e modos combinados preservam candles de 15 minutos;
`CANDLE_TIMEFRAME_SEC=900` permanece inalterado.

A janela móvel da IA tem até **28 dias**: 8.064 posições em 5 minutos ou 2.688
em 15 minutos. O piso estrutural é de **593 rótulos na IA isolada** e 579 no
perfil de 15 minutos. Além do piso, são verificadas classes, divisões temporais
externas e internas e capacidade para pelo menos 30 contratos sem sobreposição
na validação e no teste. Rótulos de 15 minutos gerados a cada 5 minutos se
sobrepõem; três rótulos não equivalem a três operações independentes.

Esse limite pertence somente à preparação da IA em `training_data.py`; a coleta
continua em sete dias/672 posições, e o histórico compartilhado por ativo,
limitado a 336 candles. RandomForest, BalancedLogisticRegression, ExtraTrees,
HistGradientBoosting e XGBoost são comparados no mesmo snapshot; ajuste,
calibração e seleção ficam separados do holdout final por embargo. ROC-AUC é
uma métrica. Dados suficientes permitem tentar treino, sem garantir aprovação.

O pipeline prepara snapshots por ativo em `.training_inputs/` junto ao dataset, sem substituir `ticks_v2.csv`. Registros inválidos ou conflitantes com época conhecida ficam no relatório de quarentena; amostras que atravessem suas épocas são vetadas. Corrupção atribuível ao ativo sem época segura bloqueia a preparação. O histórico OHLC válido pode aquecer as features apesar de lacunas anteriores de ticks; entrada e vencimento ainda exigem ticks identificados e continuidade dentro do contrato. Candles derivados de ticks com cobertura incompleta não preenchem lacunas. A nova política é versionada nos metadados e exige reconstrução/reavaliação dos artefatos anteriores, sem promover modelos automaticamente. Depois de montar o dataset, o pipeline remove o `prices.csv` do snapshot de treino e mantém apenas `prices.csv.preparation.json` (hashes, quarentena e vetos); pastas criadas antes dessa correção não são apagadas automaticamente.

No buffer privado da IA isolada, o OHLC oficial do histórico público substitui a vela montada ao vivo quando os dois divergem (conexão no meio da vela, tick filtrado), sem quarentena. Conflitos entre duas versões derivadas de ticks continuam em quarentena. Se a prova histórica do canary não puder ser verificada momentaneamente, o canary salvo fica intacto e vale apenas a validação original; reprovação persistida ou arquivo danificado continuam terminais.

Para recuperar um snapshot offline em **diretório novo**, sem treinar ou conectar:

```bash
.venv/bin/python training_data.py --input ticks_v2.csv --symbol frxGBPCAD \
  --history 'Histórico Ativos/frxGBPCAD-bf233306bc.csv' \
  --output-dir /tmp/standeriv-gbpcad-preparado
```

Preserve `prices.csv.preparation.json` junto de `prices.csv`: ele contém hashes, quarentena e vetos temporais. O painel recebe o progresso real da preparação da IA (amostras, candles válidos e gaps), com diagnóstico atualizado a cada 60 segundos. A janela maior não garante dados suficientes ou aprovação e não aumenta a retenção da coleta existente.

A preparação da IA também usa `training_history.py` para complementar os candles fechados ausentes e recuperar ticks históricos identificados. O cliente usa somente o [WebSocket público da Deriv](https://developers.deriv.com/docs/options/websocket/), sem token, OTP, conta ou assinatura ao vivo. O cache incremental fica em `.training_inputs/history/`, separado de `ticks_v2.csv` e de `Histórico Ativos/`. O limite homologado da coleta e seu backfill desativado permanecem intactos.

A cada diagnóstico, a recuperação dispõe de até 12 consultas e 15 segundos, com intervalo mínimo de um segundo entre envios. Candles usam páginas de até 1.000 posições; ticks, páginas explícitas de 5.000 registros com paginação regressiva. Somente lacunas são consultadas. O cache conserva progresso entre rodadas e reinícios; páginas vazias permanecem pendentes por pelo menos cinco minutos, e rate limit impõe backoff compartilhado entre os ativos da IA. Respostas sem cobertura suficiente não criam candles ou ticks. Retenção e disponibilidade efetivas continuam sujeitas à API; a janela de 28 dias não demonstra que a corretora fornecerá esse período.

Na IA isolada, primeiro são reaproveitados os ticks locais e os caches
verificados, incluindo ticks do cache de 15 minutos. OHLC de 15 minutos não é
dividido em candles de 5 minutos: esses candles vêm do histórico público ou de
ticks com cobertura comprovada. Se ainda faltarem dados, o ativo atual pode
continuar até quatro lotes, com teto total de 48 consultas/45 segundos, sem
esperar novos ticks entre eles. O complemento para ao atingir prontidão,
esgotar o orçamento, não progredir ou encontrar busy/backoff. Um lock ocupado
preserva o último snapshot/diagnóstico verificado, quando disponível.

O executor da IA isolada carrega 50 candles privados de 5 minutos e completa
lacunas em worker separado, sem bloquear o callback de ticks nem exigir o
aquecimento técnico de 150 candles de 15 minutos. O complemento ao vivo
consulta as últimas seis horas, mantendo o cache de 28 dias. Modos técnicos
preservam seu aquecimento e recuperação existentes. Uma origem corrompida pode
ser excluída integralmente de um snapshot independente da IA, com hash e motivo
registrados, para reconstrução por histórico público validado; o CSV original
e os vetos conhecidos são preservados.

Para complementar dados públicos sem reiniciar o bot ou treinar modelos, use um **diretório de saída novo**:

```bash
.venv/bin/python training_history.py --input ticks_v2.csv --symbol frxEURJPY \
  --history 'Histórico Ativos/frxEURJPY-7a28647167.csv' \
  --cache-dir .training_inputs/history \
  --output-dir /tmp/standeriv-eurjpy-complementado --max-requests 12 --seconds 15
```

Para preparar exclusivamente o histórico de 5 minutos, sem iniciar o bot ou
treinar, o CLI também aceita `--timeframe 300 --duration 900 --rounds 4
--total-requests 48 --total-seconds 45`. A saída deve continuar em diretório
novo; o cache é separado por símbolo/timeframe e os lotes preservam seu progresso.

O relatório `history_recovery` registra consultas, páginas obtidas, intervalos ainda ausentes, erros, timeouts e latência das consultas públicas. O diagnóstico do pipeline também resume os timeouts já registrados pelo coletor no trecho final do log, limitado a 256 KB e filtrado pela sessão. Ele não altera ping/pong, reconexão ou callbacks e não atribui os timeouts a uma causa de rede sem medição. A rotina de preparação e os critérios de aprovação/shadow só entram no processo após ele carregar o código atualizado; o comando acima apenas prepara o cache e o snapshot.

Após aprovação offline, o bundle passa por shadow forward sem compras.
Na **IA isolada de 5 minutos**, `training_canary.py` admite um canary de
**8 resultados não sobrepostos**, exigindo **pelo menos 7 acertos e saldo
unitário positivo**, se os metadados comprovarem teste histórico elegível com
ao menos 30 contratos, seleção separada e embargo temporal. Seu mínimo teórico
é de **2 horas desde a primeira previsão elegível**; não equivale à evidência
de 30 resultados novos. Payout observado mínimo de 70%, confiança, gaps,
identidade do bundle e demais gates de execução permanecem obrigatórios.

A reprovação é antecipada quando `acertos + resultados restantes < 7`.
Por exemplo, o segundo erro já impede 7/8 e encerra a tentativa, acionando
a busca de outro mercado. A aprovação continua exigindo os oito resultados.
Estados antigos inviáveis são reconhecidos sem zerar contagens: conservam
resultados e eventuais previsões simuladas pendentes para auditoria. A troca
de mercado continua aguardando a reconciliação de qualquer contrato financeiro
ativo. O status informa os acertos, o máximo ainda possível e o motivo da
reprovação (`minimum_wins_unreachable`).

A política `isolated_ai_canary_v1` tem estado próprio em
`shadow_state.json.canary/`, com época de ativação, prova histórica e ledger
dos períodos já utilizados. Resultados anteriores não são importados como
forward do canary. A validação original continua preservada em
`shadow_state.json`, acumulando seus resultados e pendências; aprovações e
reprovações antigas não são convertidas. O mesmo bundle/política retoma a
sequência após reinício, sem repetir janelas até passar. Reprovação ou arquivo
canary inválido bloqueia a reutilização daquele bundle. Candidatos diferentes
não reutilizam um período histórico já admitido nessa política curta.

Sem prova histórica suficiente ou com período já consumido, permanece a
regra original de **30 resultados e 55% de acerto**, sem reduzir o piso por
simples contagem de candles. Os modos combinados mantêm a política original.
Ambos os caminhos permitem compras somente em conta demo; conta real permanece
bloqueada. As mudanças entram em uso quando o processo carregar o código;
não reiniciam nem substituem modelos em execução.

A inferência da IA isolada reutiliza a previsão enquanto os 50 candles
fechados, preços, modelo e limiar forem idênticos. A causalidade é conferida
antes do cache, e payout/risco/compra continuam sendo avaliados normalmente.
O status publica liquidações intermediárias, contagens separadas do canary e
da validação original, duração do trecho de análise do callback e timeouts
separados de coletor/executor. Essa medição não cobre o callback inteiro nem
comprova que timeouts de conexão tenham sido eliminados.

O target representa alta, queda ou empate entre o primeiro tick após o sinal e o primeiro tick no vencimento. Empate perde tanto CALL quanto PUT. A duração atual é fixa em 15 minutos (`DURATION=15`, `DURATION_UNIT="m"`). As duas próximas observações e sua média não são usadas como target.

No caminho com IA ativa, abstenção, falta de dados e modelo incompatível
impedem o voto do modelo; entradas dependem também do consenso dos módulos. A força técnica real participa do score; o score não representa probabilidade de vitória. Antes da compra, a probabilidade calibrada da direção é confrontada com preço e payout da proposta.

### Progresso e escolha de candidatos do Modelo de Treino

A avaliação do Modelo de Treino isolado usa sua previsão e os gates de
confiança, payout e retorno esperado; não exige o score combinado dos módulos
técnicos desligados. A política de avaliação passa à versão 4 e identifica os
geradores habilitados. Bundles avaliados sob políticas anteriores permanecem
preservados, mas exigem reavaliação antes de serem usados pelo código atualizado.

O progresso em `.training_inputs/progress.json` reconhece rótulos por identidade
temporal e resultado, separadamente por ativo. Entrar informação nova e expirar
informação antiga na janela de 28 dias não zera esse progresso. Uma tentativa
consome somente os rótulos de seu snapshot, mantendo os lotes de renovação já
definidos e os critérios de aprovação.

Após uma tentativa, o complemento da IA só considera os dados prontos quando
também há rótulos ainda não usados suficientes para o lote de retreino. Ter
1.323 rótulos antigos, por exemplo, não encerra a recuperação se faltam os
120 novos exigidos após reprovação de desempenho. O bootstrap pode completar
lacunas históricas antes de existir modelo aprovado ou executor iniciado,
sem consumir esses rótulos durante o diagnóstico.

Somente na IA isolada (300/900), entre os candidatos elegíveis do scanner
existente, a IA prioriza dados prontos e capacidade para pelo menos 30 contratos
sem sobreposição na validação e no teste. Nos modos combinados de 900 segundos
(IA com Score, Price Action ou padrões), a sessão usa o primeiro candidato do
scanner e não troca de mercado por prontidão de dados, como na homologação. Esse limite superior não garante sinais nem aprovação. Durante a espera
sem modelo, consulta até dois alternativos por rodada, com intervalo mínimo de
60 segundos entre rodadas; só pede troca quando outro candidato estiver pronto
para treino ou tiver um bundle compatível. A sessão anterior é encerrada antes
da troca. Contratos pendentes continuam tendo prioridade de reconciliação.
Na IA isolada, os candidatos já consultados saem temporariamente da próxima
pesquisa: o percurso alcança os elegíveis além dos cinco primeiros, preservando
os filtros do scanner. Um bundle avançando no shadow não é interrompido por
essa busca de dados.

Bundles aprovados são reservados por símbolo e hash em
`.training_inputs/models/`. A retomada valida contrato, política, dependências,
hash e payload. O shadow guarda estados completos em `shadow_state.json.states/`
e só recupera a identidade exata de símbolo, modelo, timeframe e vencimento.
Um bundle em shadow não é trocado por simples espera de dados; reprovação
shadow permanece associada ao bundle e não é apagada ao voltar ao ativo.
Os gates de shadow/demo, Score Ponderado e coleta permanecem preservados.

## Seleção e histórico de ativos

O scanner consulta `active_symbols` da Deriv para categoria, abertura e
suspensão; não classifica ativos pelo nome/código. O ADX é calculado sobre
candles fechados de 15 minutos e precisa corresponder ao último candle fechado,
com valor >= 25. O score composto existente só ordena os elegíveis; empates são
resolvidos pelo código do símbolo. Ativos não sintéticos são avaliados primeiro.
Sintéticos só são consultados quando nenhum não sintético passar, e precisam
estar abertos, sem suspensão, atender ao ADX e à sondagem CALL/PUT.

`Histórico Ativos/` mantém um CSV OHLC no esquema v2 por símbolo, além de
metadados temporários de epochs sem candle. A sincronização lê primeiro o disco,
busca gaps e candles posteriores e grava os resultados ordenados e deduplicados
atomicamente. O limite local é `ASSET_HISTORY_CANDLES=336` candles; o orçamento histórico
da coleta continua sendo de sete dias/672 candles, separado dos 28 dias da preparação da IA. Falha de sincronização
não remove candles recentes válidos. O executor reutiliza esse histórico no
buffer compartilhado. O treino continua exigindo ticks identificados; candles
não são convertidos em ticks.

O calendário oficial `trading_times` é consultado por data UTC e compartilhado
entre os ativos, com cache em memória de uma hora (falhas por até um minuto).
Intervalos inteiramente fechados não são solicitados como lacunas. Fins de
semana e pausas diárias seguem os horários do símbolo, sem tabela fixa de CFDs.
Sem calendário válido, ou com evento especial ambíguo, a ausência de dados
continua não confirmada. Dados retornados fora da faixa solicitada por
`adjust_start_time` são ignorados antes da validação; OHLC inválido dentro da
faixa continua sendo rejeitado.
O scanner e o aquecimento técnico podem atravessar somente fechamentos
confirmados, sem fabricar candles. Bundles de IA preservam a continuidade
temporal estrita do treinamento.
Referência: [Trading Times — Deriv](https://developers.deriv.com/docs/system/trading-times/).

Bot e coletor mantêm também um ping JSON da API a cada 30 segundos, além do
ping/pong do protocolo WebSocket. As confirmações são silenciosas; latência e
contadores ficam em memória, e falhas continuam no log. A leitura inicial do
CSV fica fora do callback de reconexão.
O aquecimento solicita faixas com início/fim explícitos, correlacionadas por
`req_id`, e preserva os candles dos dois lados das lacunas. Candles recuperados
são gravados no histórico do ativo; respostas atrasadas ou fora da faixa não
completam a recuperação. Uma página vazia é registrada como ainda ausente. No executor, a recuperação
incompleta mantém entradas bloqueadas e permite nova tentativa após 60 segundos.
Referência: [Ping — Deriv](https://developers.deriv.com/docs/system/ping/).

O status operacional usa um identificador por sessão. Decisões, erros, ticks e
diagnósticos shadow são reiniciados ao iniciar uma sessão ou trocar de símbolo.
O executor publica a quantidade real de candles no buffer e o aquecimento
restante; atualizações do pipeline preservam esses valores. Os escritores usam
o mesmo lock de arquivo para evitar perda de atualizações. Consultas do
dashboard preservam o horário de atividade informado pelo processo.

## Decisão de abertura de contrato

A abertura de contrato passa por camadas sucessivas. Nenhum módulo compra
sozinho: `executor.py` só envia proposta quando a conta está autenticada, o tick
é recente e válido, há pelo menos `MIN_CANDLES` velas, não existe contrato ativo
ou pendente, a duração prevista coincide com `DURATION`/`DURATION_UNIT` e os
gates de cadência e risco configurados permitem nova tentativa.

`strategy.py` executa independentemente cada módulo habilitado: `weighted_score`,
`trained_model`, `price_action` e `candle_pattern`. Os defaults atuais mantêm
score e IA ativos e PA/padrões desativados; cada chave pode ser configurada sem
desligar os demais. Cada módulo recebe os mesmos candles. Erro, ausência de
dados, abstenção ou sinal inválido permanece como voto não aprovado no total de
módulos ativos. Só há sinal quando mais da metade aprova a mesma direção; empate,
divergência ou nenhum módulo ativo bloqueia.

Quando apenas o score ponderado está ativo, o diagnóstico preserva o motivo
específico da abstenção (score insuficiente, ADX ou votação técnica). A troca
por avaliações desfavoráveis conta somente candles fechados, uma vez por candle,
respeitando o limite configurado e as proteções de contrato ativo e cooldown.
Lacunas no histórico ao vivo bloqueiam novas avaliações de entrada enquanto o
executor recupera somente os candles ausentes, preservando os dados válidos.
Respostas parciais, erros e timeouts mantêm o bloqueio e permitem nova tentativa
após 60 segundos, sem duplicar a assinatura de ticks.

Com `USE_AI_MODEL=True`, `trained_model` só pode votar com bundle v4 compatível,
50 candles OHLC contínuos, direção válida, `model_id` e confiança acima de
`AI_CONFIDENCE_MIN`. O voto e os detalhes de uma IA dissidente não são usados
como probabilidade da direção majoritária.

Com `USE_WEIGHTED_SIGNAL=True`, o módulo `weighted_score` calcula `tech_score`
com EMA9 versus EMA21 (25%), histograma MACD (20%), Bollinger (15%), preço
versus EMA9 (15%), momentum (15%) e RSI (10%), multiplicados pelo ADX normalizado.
O executor fornece fechamentos de candles de 15 minutos; o cálculo usa os
últimos 50 fechamentos e Bollinger usa os últimos 20 dessa sequência.
Score maior ou igual a `SIGNAL_SCORE_MIN` propõe `BUY`; score menor ou igual ao
negativo do limiar propõe `SELL`. A aprovação interna requer que pelo menos três
dos cinco componentes direcionais existentes concordem. ADX continua sendo
modulador/filtro interno; o piso ADX adaptativo de estratégia é separado do piso
25 usado para seleção de ativos.

Price Action usa as features existentes de estrutura, tendência, break of
structure, zonas e FVG. Padrões usam os detectores existentes e precisam de
ocorrências fortes em pelo menos duas velas distintas dentro de
`CANDLE_PATTERN_MAX_AGE_SEC`; uma ocorrência isolada abstém.

Depois do sinal, `executor.py` ainda pede uma proposta à Deriv. A compra só é
enviada se a proposta tiver preço e payout finitos, `0 < preço < payout`, payout
líquido mínimo `MIN_NET_PAYOUT` e preço dentro da stake. Com IA ativa, mesmo
quando seu voto não integra a maioria, continuam valendo os gates de
compatibilidade, probabilidade calibrada e shadow/demo. Com IA desativada,
valida-se o payout sem inventar probabilidade para outros módulos. A função
`profitable_proposal()` exige retorno esperado não negativo:
`probabilidade * payout_total / preço - 1 >= 0`. Rejeições nessa etapa não abrem
contrato e limpam a pendência local.

Os artefatos antigos de TFT, stacking e duração foram preservados, mas não participam do novo caminho operacional. A flag legada `USE_TRANSFORMER` não promove automaticamente um ensemble não validado.

O relatório separa IA, sinais técnicos e estratégia completa, com bloqueios, resultados por direção/horário/confiança e calibração. O retreino exige 120 novos contratos rotulados (`RETRAIN_NEW_LABELS_TRIGGER`), além do intervalo mínimo. O coletor consulta CALL/PUT a cada cinco minutos, sem compra, e registra `payout_quotes.csv`. Avaliações com cotações observadas usam somente dados anteriores à entrada, com até 900 segundos de idade, sem preencher lacunas. Os cenários fixos de payout líquido de 85% e 70% continuam explicitamente identificados como hipóteses; os critérios anteriores de promoção são preservados.

O conjunto operacional continua com 27 features. O experimento de contexto temporal acrescenta oito colunas ao dataset, mas só compara modelos no período de desenvolvimento; não publica um bundle de 35 features:

```bash
.venv/bin/python training_experiments.py --input /tmp/dataset-v2.csv \
  --output /tmp/feature-comparison.json --symbol frxUSDCAD
```

O dataset deve ser reconstruído com `dataset_builder.py` para conter as novas colunas. Relatórios de treino ficam em `training_reports/`. Veja [a implementação e a avaliação de 10/09](docs/historico/MELHORIAS_AVALIACAO_2026-09-10.md).

O treinador também compara regressão logística, versão balanceada e frequência de classes, além das árvores
RandomForest/ExtraTrees/HistGradientBoosting/XGBoost. Cada estimador é avaliado com calibração multinomial e
temperature scaling; probabilidades brutas e frequência de classes são controles
diagnósticos, sem seleção para publicação. A busca usa seed 42 e seis
configurações adicionais de árvores por padrão (`--search-trials`, de 0 a 12).
A receita é escolhida apenas no development e o ajuste final usa seus 80% iniciais,
seguido por calibração temporal separada, com embargo antes do teste final.

Para comparar a receita nova com a anterior em um snapshot, sem consultar ou
substituir o champion, use um **diretório de saída novo**:

```bash
.venv/bin/python training_experiments.py --study training \
  --input dataset_v2.csv --symbol frxUSDCAD \
  --output /tmp/standeriv-training-study --search-trials 4 \
  --payout-quotes payout_quotes.csv
```

O comando preserva os snapshots, grava `comparison.json` e um `model.candidate`
experimental no diretório indicado. Omita `--payout-quotes` se não houver histórico
de propostas. O relatório inclui probabilidades brutas/calibradas, percentis de
confiança, AUC por direção, ganho sobre frequência de classes e curvas de precisão,
cobertura e retorno para payouts fixos de 60%, 70% e 85%. Essas curvas não ajustam
limiares no teste. O fuso da avaliação é registrado separadamente como
`evaluation_risk_timezone=UTC`; `risk_timezone` continua descrevendo o calendário
local da operação. Veja [a aplicação e seus resultados](docs/historico/APLICACAO_TREINO_2026-09-11.md).

Para medir se as 27 features ajudam ou se alguma família atrapalha, use a
ablação offline. Ela compara baseline, remoção de Price Action, Price Action
isolado, técnicas enxutas, remoção de tendência/volatilidade e contexto temporal
quando o dataset tiver essas colunas. O relatório usa somente o development,
preserva o holdout e não publica bundle:

```bash
.venv/bin/python training_experiments.py --study ablation \
  --input dataset_v2.csv --symbol frxUSDCAD \
  --output /tmp/standeriv-feature-ablation.json --search-trials 0
```

### Preparação e treino offline

Use um snapshot estável, distinto do arquivo em gravação:

```bash
cp ticks_v2.csv /tmp/standeriv-ticks-snapshot.csv
python dataset_builder.py --input /tmp/standeriv-ticks-snapshot.csv --output /tmp/standeriv-dataset-v2.csv --symbol R_100
python train_model.py --input /tmp/standeriv-dataset-v2.csv --output /tmp/standeriv-model.candidate --symbol R_100
```

Esses comandos escrevem dataset, candidato e metadados em `/tmp`; não publicam o modelo. O pipeline com `--force-retrain` treina e só promove um candidato aprovado. A avaliação do modelo aplica os filtros técnicos, a cadência, contratos não sobrepostos e a gestão de risco, com payout líquido de referência de 85%. Os payouts reais e a latência de execução ainda exigem validação operacional.

### Recuperação de arquivos legados

```bash
python market_data.py --input ticks.csv --output-dir recovered_data/nova-recuperacao
```

A origem não é alterada. Registros sem identificação de tick/candle recebem `legacy_unknown` e não são aceitos para treinar contratos. Novas gravações usam `ticks_v2.csv` com campos explícitos de proveniência e OHLC. Dados antigos não permitem inventar os preços de entrada ou máximas/mínimas ausentes.

## Configurações Avançadas

`ENFORCE_RISK_GATE=False` é o valor atual: o gerenciador registra pausas,
resultados e stake, mas essas pausas e stops não impedem propostas/compras. O
painel pode exibir pausa enquanto a execução continua. A homologação de Score
e coleta não aprova essa política nem a altera. O comando Parar encerra o
processo; não liquida contratos na corretora. Preserve o contrato persistido
para reconciliação antes de retomar qualquer operação.

### ADX adaptativo

`ADX_MIN=30` exige tendência forte antes de liberar o score técnico. Com
`ADX_ADAPTIVE=True`, o percentil configurado do histórico de ADX pode elevar o
limiar, mas não reduzi-lo abaixo de `ADX_MIN`. O histórico é atualizado também
durante contratos e pausas. O ADX continua sendo a aproximação por fechamentos
implementada em `indicators.py`, não um ADX calculado a partir do range completo
de cada vela.

### Score técnico e IA

O Score Ponderado está homologado; pesos, fórmulas e parâmetros estão descritos
em [sua especificação](docs/SCORE_PONDERADO_BOLLINGER.md).
`SIGNAL_SCORE_MIN=0.06` controla a seleção técnica. `ENTRY_MODE="balanced"`
mantém o ADX mínimo como gate, em vez de apenas penalizar o score. `AI_CONFIDENCE_MIN` segue
o piso de aprovação (`MIN_APPROVAL_WIN_RATE=0.45`). O caminho operacional usa
votos independentes e maioria estrita: com apenas Score e IA ativos, ambos
precisam concordar. `AI_TECH_WEIGHT`, `AI_MODEL_WEIGHT` e `AI_SCORE_MIN`
permanecem no helper auxiliar e não definem a votação atual. Com IA desligada,
a execução técnica valida preço e payout sem exigir ou inventar probabilidade
calibrada; treino, promoção e shadow de modelo não liberam nem bloqueiam esse caminho.

### Candles Baseados em Tempo
`CANDLE_TIMEFRAME_SEC = 900` define o timeframe operacional atual: candles de 15 minutos para Score Ponderado, Price Action, padrões e modos combinados. O Modelo de Treino isolado (IA habilitada e demais geradores desligados) usa candles próprios de 300 segundos, definidos por `model_contract.model_timeframe_sec()`, sem alterar este parâmetro. O vencimento continua em 15 minutos nos dois perfis.

### Buffer de Preços
`PRICE_BUFFER_SIZE = 500` limita o buffer bruto. Features de modelos usam a janela canônica de 50 candles definida em `model_contract.py`.

### Cadência de Entradas
`ENFORCE_ENTRY_CADENCE=False` permite reavaliação a cada tick aceito, usando
os mesmos candles fechados até o próximo fechamento. Se habilitado,
`ENTRY_CANDLE_INTERVAL=1` exige uma nova vela entre tentativas; após pelo menos
duas perdas consecutivas, o intervalo aumenta em uma vela. A troca por mercado
desfavorável continua sendo contada somente nos fechamentos.

### Promoção de Modelo (Champion-Challenger)
Além de pelo menos 30 contratos de teste e taxa de acerto observada mínima de 45% na estratégia completa, o challenger precisa ser comparado com o champion no mesmo período fora da amostra. AUC abaixo de 0,50 é registrada como alerta de probabilidade fraca, mas não bloqueia sozinha a aprovação:
- `MODEL_PROMOTION_MIN_AUC_DELTA = 0.001` — AUC do novo modelo deve ser ≥ AUC atual + delta
- `MODEL_PROMOTION_MAX_ACC_DROP = 0.02` — queda máxima tolerada de acurácia
- `MODEL_PROMOTION_MAX_F1_DROP = 0.05` — queda máxima tolerada de F1

### Filtro Anti-Spike
`TICK_SPIKE_THRESHOLD = 0.05` rejeita ticks com variação >= 5% em relação ao anterior, protegendo os indicadores de dados ruins.

### Heartbeat / Watchdog
`HEARTBEAT_TIMEOUT_SEC = 30` — marca o feed como desatualizado se nenhum tick chegar nesse prazo. O keepalive ping/pong detecta conexão sem resposta; ao encerrar, o bot reconecta com backoff progressivo e jitter.

### Dashboard — Autenticação
Defina `DASHBOARD_TOKEN` no `.env` para proteger as rotas de controle do bot com `X-Auth-Token`.



| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `STAKE_PCT` | 1% | Percentual do saldo por operação |
| `STOP_LOSS_PCT` | 25% | Stop diário (perda máxima) |
| `TAKE_PROFIT_PCT` | 50% | Take profit diário |
| `MAX_CONSEC_LOSSES` | 3 | Losses consecutivos antes de pausar |
| `PAUSE_BASE_SEC` | 600 | Pausa base (10 min) no 1º gatilho |
| `PAUSE_SCALE_FACTOR` | 2 | Fator de escala da pausa (dobra a cada loss extra) |
| `RESUME_ON_WIN` | True | Retoma pausa imediatamente após 1 win |
| `DRIFT_WINDOW` | 20 | Janela de trades para calcular win rate recente |
| `DRIFT_WIN_RATE_MIN` | 40% | Alerta se win rate dos últimos N trades cair abaixo |
| `AI_TECH_WEIGHT` | 0.55 | Peso técnico no helper auxiliar; não define a maioria dos módulos |
| `AI_MODEL_WEIGHT` | 0.45 | Peso da IA no helper auxiliar; não define a maioria dos módulos |
| `AI_SCORE_MIN` | 0.30 | Limiar do helper auxiliar; não define o voto operacional da IA |
| `SIGNAL_SCORE_MIN` | 0.06 | Limiar mínimo do score técnico ponderado em modo balanceado |
| `ENTRY_MODE` | `balanced` | Modo de entrada; preserva ADX mínimo como gate antes do score ponderado |
| `OPPORTUNITY_SIGNAL_SCORE_MIN` | 0.05 | Limiar técnico usado se o modo `opportunity` for reativado |
| `OPPORTUNITY_ADX_WEIGHT_FLOOR` | 0.35 | Piso do peso de ADX no modo `opportunity`, evitando zerar o score por ADX baixo |
| `PRICE_ACTION_FILTER` | `False` | Habilita o voto independente de Price Action no consenso |
| `CANDLE_PATTERN_FILTER` | `False` | Habilita o voto independente de padrões de vela; alertas continuam separados |
| `ENFORCE_ENTRY_CADENCE` | `False` | Controla se o executor deve aguardar fechamento/intervalo de vela antes de reavaliar entrada |
| `ENFORCE_RISK_GATE` | `False` | Controla se `RiskManager.can_trade()` bloqueia envio/compra; desligado, risco ainda calcula stake e registra resultado |
| `UNFAVORABLE_SYMBOL_SWITCH_DECISIONS` | 12 | Fechamentos de candle com avaliação desfavorável antes de solicitar troca; `0` desativa |
| `CANDLE_TIMEFRAME_SEC` | 900 | Duração de cada vela de tempo em segundos |
| `MIN_CANDLES` | 150 | Aquecimento mínimo do executor para contratos de 15 minutos |
| `PRICE_BUFFER_SIZE` | 500 | Limite do buffer bruto de ticks; a estratégia operacional recebe fechamentos de candles |
| `ENTRY_CANDLE_INTERVAL` | 1 | Intervalo mínimo entre entradas (em velas) |
| `DURATION` | 15 | Vencimento fixo do contrato, em minutos |
| `ADX_MIN` | 30 | Tendência mínima exigida para sinal técnico em modo balanceado |
| `ADX_ADAPTIVE_PERCENTILE` | 50 | Percentil do histórico de ADX usado como piso adaptativo |
| `TARGET_LOOKFORWARD` | 1 | Candles de 15 minutos até o vencimento; o target é calculado por timestamps |
| `SHADOW_MIN_RESULTS` | 30 | Resultados forward da validação original; não reduzir para o canary |
| `SHADOW_MIN_ACCURACY` | 55% | Acerto mínimo da validação original |
| `SHADOW_MIN_NET_PAYOUT` | 70% | Payout líquido observado mínimo no shadow e no canary |
| `RETRAIN_NEW_LABELS_TRIGGER` | 120 | Rótulos novos exigidos para retreino após reprovação de desempenho |
| `TRAINING_HISTORY_DAYS` | 7 | Orçamento da coleta; a preparação da IA usa 28 dias em `training_data.py` |
| `ASSET_HISTORY_CANDLES` | 336 | Limite local do histórico OHLC por ativo |

## Histórico de Mudanças anteriores ao contrato v2

Esta lista preserva decisões antigas; valores, regras e componentes nela
citados não substituem a configuração e os contratos vigentes acima.

Os itens abaixo registram versões passadas; o comportamento atual está descrito acima e no guia de migração.

- `feature_engine.py` criado como fonte única de features (evita duplicação entre `dataset_builder`, `executor` e `ai_predictor`)
- ADX adaptativo por percentil do histórico recente (`ADX_ADAPTIVE`)
- Duração dinâmica prevista por modelo dedicado (`duration_model.pkl`)
- Pausa escalável com `PAUSE_SCALE_FACTOR` e retomada imediata com `RESUME_ON_WIN`
- Detecção de drift com janela deslizante de win rate (`DRIFT_WINDOW`)
- Filtro anti-spike no coletor (`TICK_SPIKE_THRESHOLD`)
- Heartbeat watchdog contra desconexão silenciosa
- Dashboard com autenticação por token (`DASHBOARD_TOKEN`)
- Score ponderado IA vs técnico (`AI_TECH_WEIGHT=0.55`, `AI_MODEL_WEIGHT=0.45`, `AI_SCORE_MIN=0.30`)
- Sinal ponderado por indicador (`USE_WEIGHTED_SIGNAL`, `SIGNAL_SCORE_MIN=0.06`)
- Target lookforward configurável (`TARGET_LOOKFORWARD=2`)
- Velas baseadas em tempo com `CANDLE_TIMEFRAME_SEC` (substitui modelo de ticks fixos)
- Buffer de preços configurável (`PRICE_BUFFER_SIZE=500`)
- Cadência de entradas por velas fechadas (`ENTRY_CANDLE_INTERVAL`)
- Promoção de modelo champion-challenger com critérios de AUC/acurácia/F1
- Durações candidatas atualizadas para minutos: `CANDIDATE_DURATIONS=[5, 15, 30]`
- ADX adaptativo atualizado para percentil 50 (`ADX_ADAPTIVE_PERCENTILE=50`)
- Adicionadas 11 features de Price Action (estrutura de mercado, S/R, supply/demand, FVG)
- Detecção de 6 padrões de vela com notificações em tempo real
- Scores contínuos (-1 a +1) na estratégia ponderada
- Filtro RSI relaxado (permite momentum forte)
- Filtro PA na estratégia (bloqueia sinais contra estrutura)
- Filtro ADX rising (bloqueia quando tendência enfraquece)
- Cadência adaptativa (aumenta intervalo após losses)
- Modelo incompatível, dados insuficientes ou sinal divergente → não opera
- Target com threshold de ruído (elimina variações insignificantes)
- Duração por heurística de volatilidade (sem lookahead bias)
- Log em modo append (preserva histórico entre reinícios)
- Janela temporal de 50k ticks para retreino
- Dashboard: painel de padrões de vela com polling automático
- 27 features no total (16 técnicas + 11 PA)


O payout líquido mínimo é `MIN_NET_PAYOUT=0.60`: retorno total de pelo menos
1,60 por unidade de stake. O filtro vale no backtest, na avaliação de treino e
nas compras em contas demo e reais. `MIN_APPROVAL_WIN_RATE=0.45` é o piso de
aprovação da estratégia completa; Wilson e AUC continuam como diagnóstico.
Acurácia de classificação e taxa de acerto dos contratos são métricas distintas.
Os filtros de retorno esperado, risco e comparação entre modelos permanecem.
Bundles anteriores exigem reavaliação sob a nova política.

## Regra atual de entrada com IA: confiança e payout

A confiança estimada pelo modelo operacional deve ser estritamente maior que
`AI_CONFIDENCE_MIN`, derivado de `MIN_APPROVAL_WIN_RATE`. Quando a IA atua como
filtro auxiliar de um sinal técnico na mesma direção, o filtro direto usa piso
efetivo de 55%. A proposta só permite compra quando
`probabilidade × payout_total / preço − 1 >= 0`, substituindo a margem anterior
de 2%. O payout líquido mínimo é 60%; os filtros técnicos seguem seus módulos.
O bloqueio de risco depende de `ENFORCE_RISK_GATE`, atualmente `False`. Isso representa retorno esperado não negativo segundo o
modelo; não garante ausência de perdas. A confiança de uma previsão difere da
taxa histórica de acerto: a aprovação do treino exige pelo menos 45% de acertos
na estratégia completa.
Mudanças no código não reiniciam o executor nem promovem modelos existentes.

O backtest com IA aplica a mesma função `profitable_proposal` do executor,
exigindo `win_probability` válida, confiança acima de `AI_CONFIDENCE_MIN` e
retorno esperado não negativo. Custos hipotéticos configurados no backtest são
descontados antes da aceitação. Rejeições aparecem como
`unprofitable_proposal` no relatório. O backtest técnico permanece sem previsão probabilística. No executor,
o caminho sem IA também pode comprar propostas com payout válido.
