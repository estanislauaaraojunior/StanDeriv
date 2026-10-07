# StanDeriv — Bot de Trading Automatizado para Deriv

Bot de contratos CALL/PUT para a Deriv, com análise técnica, Price Action, modelos calibrados e execução com controle persistente de risco.

A manutenção de contratos v2 alterou o esquema dos dados e a validação dos modelos. Leia [Correções e migração](docs/CORRECOES_CONTRATOS_2026-09-08.md) antes de iniciar. Modelos legados não autorizam novas entradas; sem modelo compatível, o pipeline mantém somente a coleta. A [auditoria das perdas](docs/historico/ANALISE_LOSSES_2026-09-08.md) é um registro histórico.

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

- Python 3.11+
- Conta na Deriv (demo ou real) com token de API

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

2. Ajuste parâmetros em `config.py` conforme necessário.

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

Sem modelo v2 aprovado para o ativo, esse comando mantém apenas a coleta. Treino inicial requer `--force-retrain` e dados suficientes. Nenhum candidato substitui o modelo ativo sem passar pela avaliação temporal. A opção `--demo` é confrontada com a conta autenticada.

### Apenas o bot (modelo já treinado)

```bash
python bot.py
```

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
| `collector.py` | Coleta contínua de ticks via WebSocket com filtro anti-spike |
| `backtest.py` | Simulação offline de sinais, duração, payout, custos, risco e drawdown, com relatórios JSON/CSV |
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

O treino atual compara RandomForest, variantes balanceadas, ExtraTrees, HistGradientBoosting e XGBoost em três janelas walk-forward com calibração e embargo temporal. A seleção considera amostra, estabilidade do resultado, drawdown, qualidade das probabilidades e AUC, usando somente o período de desenvolvimento. O candidato escolhido é avaliado no teste final reservado. O bundle registra ativo, janela de 100 candles OHLC, features, versões de dependências e vencimento.

O target representa alta, queda ou empate entre o primeiro tick após o sinal e o primeiro tick no vencimento. Empate perde tanto CALL quanto PUT. A duração atual é fixa em 15 minutos (`DURATION=15`, `DURATION_UNIT="m"`). As duas próximas observações e sua média não são usadas como target.

Abstenção, divergência, falta de dados e modelo incompatível bloqueiam entradas. A força técnica real participa do score; o score não representa probabilidade de vitória. Antes da compra, a probabilidade calibrada da direção é confrontada com preço e payout da proposta.

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

### ADX adaptativo

`ADX_ADAPTIVE=True` utiliza o percentil configurado do histórico de ADX. O histórico é atualizado também durante contratos e pausas. O ADX continua sendo a aproximação por fechamentos implementada em `indicators.py`, não um ADX calculado a partir do range completo de cada vela.

### Score técnico e IA

`SIGNAL_SCORE_MIN=0.05` controla a seleção técnica. `AI_CONFIDENCE_MIN` segue
o piso de aprovação (`MIN_APPROVAL_WIN_RATE=0.45`) e a concordância explícita
entre IA e sinal técnico continua obrigatória quando a IA é usada. A combinação
usa `AI_TECH_WEIGHT=0.55`, `AI_MODEL_WEIGHT=0.45` e `AI_SCORE_MIN=0.30`.
Desligar a IA permite análise/backtest técnico, mas não fornece a probabilidade
exigida para comprar contratos ao vivo.

### Candles Baseados em Tempo
`CANDLE_TIMEFRAME_SEC = 60` define a duração de cada vela em segundos (substituindo o modelo de ticks fixos). Recomendado: 60s (1min) para mais entradas, 300s (5min) para mais seletividade.

### Buffer de Preços
`PRICE_BUFFER_SIZE = 500` limita o buffer bruto. Features de modelos usam a janela canônica de 100 candles definida em `model_contract.py`.

### Cadência de Entradas
`ENTRY_CANDLE_INTERVAL = 1` controla o intervalo mínimo entre entradas em velas fechadas (1 = avalia a cada nova vela).

### Promoção de Modelo (Champion-Challenger)
Além de pelo menos um contrato de teste e taxa de acerto observada mínima de 45% na estratégia completa, o challenger precisa ser comparado com o champion no mesmo período fora da amostra. AUC abaixo de 0,50 é registrada como alerta de probabilidade fraca, mas não bloqueia sozinha a aprovação:
- `MODEL_PROMOTION_MIN_AUC_DELTA = 0.001` — AUC do novo modelo deve ser ≥ AUC atual + delta
- `MODEL_PROMOTION_MAX_ACC_DROP = 0.02` — queda máxima tolerada de acurácia
- `MODEL_PROMOTION_MAX_F1_DROP = 0.05` — queda máxima tolerada de F1

### Filtro Anti-Spike
`TICK_SPIKE_THRESHOLD = 0.05` rejeita ticks com variação > 5% em relação ao anterior, protegendo os indicadores de dados ruins.

### Heartbeat / Watchdog
`HEARTBEAT_TIMEOUT_SEC = 30` — alerta no log se nenhum tick for recebido por 30 segundos (indica possível desconexão WebSocket).

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
| `AI_TECH_WEIGHT` | 0.55 | Peso do sinal técnico no score final |
| `AI_MODEL_WEIGHT` | 0.45 | Peso do sinal da IA no score final |
| `AI_SCORE_MIN` | 0.30 | Score mínimo ponderado (IA + técnico) para abrir operação |
| `SIGNAL_SCORE_MIN` | 0.05 | Limiar mínimo do score técnico ponderado |
| `ENTRY_MODE` | `opportunity` | Modo de entrada; em `opportunity`, ADX fraco penaliza score em vez de bloquear |
| `OPPORTUNITY_SIGNAL_SCORE_MIN` | 0.02 | Limiar técnico usado no modo `opportunity` |
| `OPPORTUNITY_ADX_WEIGHT_FLOOR` | 0.35 | Piso do peso de ADX no modo `opportunity`, evitando zerar o score por ADX baixo |
| `PRICE_ACTION_FILTER` | `False` | Controla se Price Action pode bloquear sinais operacionais; as features continuam calculadas para telemetria e treino |
| `CANDLE_PATTERN_FILTER` | `False` | Controla se padrões de vela conflitantes podem bloquear sinais; desligado, viram apenas telemetria/alerta |
| `ENFORCE_ENTRY_CADENCE` | `False` | Controla se o executor deve aguardar fechamento/intervalo de vela antes de reavaliar entrada |
| `ENFORCE_RISK_GATE` | `False` | Controla se `RiskManager.can_trade()` bloqueia envio/compra; desligado, risco ainda calcula stake e registra resultado |
| `UNFAVORABLE_SYMBOL_SWITCH_DECISIONS` | 5 | Quantidade de avaliações desfavoráveis antes de trocar ativo; `0` desativa a troca automática |
| `CANDLE_TIMEFRAME_SEC` | 60 | Duração de cada vela de tempo em segundos |
| `PRICE_BUFFER_SIZE` | 500 | Ticks mantidos em memória para indicadores |
| `ENTRY_CANDLE_INTERVAL` | 1 | Intervalo mínimo entre entradas (em velas) |
| `DURATION` | 15 | Vencimento fixo do contrato, em minutos |
| `ADX_ADAPTIVE_PERCENTILE` | 50 | Percentil do histórico de ADX usado como piso adaptativo |
| `TARGET_LOOKFORWARD` | 15 | Compatibilidade com o vencimento; o target é calculado por timestamps |

## Histórico de Mudanças anteriores ao contrato v2

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
- Sinal ponderado por indicador (`USE_WEIGHTED_SIGNAL`, `SIGNAL_SCORE_MIN=0.05`)
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

### Regra de entrada: confiança e payout

A confiança estimada para cada entrada deve ser **estritamente maior que 55%**.
A proposta só permite compra quando `probabilidade × payout_total / preço − 1 >= 0`,
substituindo a margem anterior de 2%. O payout líquido mínimo de 60%, os filtros
técnicos e os controles de risco continuam ativos. Isso representa retorno
esperado não negativo segundo o modelo; não garante ausência de perdas.
A confiança de uma previsão difere da taxa histórica de acerto: a aprovação
do treino exige pelo menos 45% de acertos na estratégia completa.
Mudanças no código não reiniciam o executor nem promovem modelos existentes.

O backtest com IA aplica a mesma função `profitable_proposal` do executor,
exigindo `win_probability` válida, confiança acima de 55% e retorno esperado
não negativo. Custos hipotéticos configurados no backtest são descontados
antes da aceitação. Rejeições aparecem como `unprofitable_proposal` no relatório.
O modo técnico permanece um comparador sem previsão probabilística.
