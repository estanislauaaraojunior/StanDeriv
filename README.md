# StanDeriv — Bot de Trading Automatizado para Deriv

Bot de opções binárias para índices sintéticos da plataforma Deriv, com análise técnica, Price Action e inteligência artificial (ensemble ML + Transformer).

---

## Arquitetura

```
┌────────────┐     ┌──────────────┐     ┌────────────────┐
│ collector   │────▶│ dataset_     │────▶│ train_model    │
│ (ticks.csv) │     │ builder      │     │ (model.pkl)    │
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
```

## Requisitos

- Python 3.10+
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
```

2. Ajuste parâmetros em `config.py` conforme necessário.

## Uso Rápido

### Pipeline completo (coleta + treino + bot)

```bash
python pipeline.py --demo --balance 1000
```

### Apenas o bot (modelo já treinado)

```bash
python bot.py
```

### Apenas coletar ticks

```bash
python collector.py
```

### Dashboard

```bash
cd dashboard && bash start.sh
# ou: python dashboard/server.py
# Acesse: http://localhost:5055
```

## Módulos

| Arquivo | Descrição |
|---------|-----------|
| `config.py` | Configurações centralizadas (indicadores, risco, IA, Price Action) |
| `indicators.py` | Indicadores técnicos puros (EMA, RSI, MACD, ADX, Bollinger, Momentum) + Price Action (estrutura de mercado, S/R, supply/demand, FVG) + detecção de 6 padrões de vela |
| `feature_engine.py` | Fonte única de features: consolida `compute_feature_map` e `extract_feature_vector` (usados por `dataset_builder`, `executor` e `ai_predictor`) |
| `dataset_builder.py` | Extrai features de ticks.csv e gera dataset.csv para treino |
| `train_model.py` | Treina RandomForest, XGBoost, StackingClassifier e Temporal Fusion Transformer |
| `ai_predictor.py` | Inferência em tempo real com ensemble blend (clássico + TFT) |
| `strategy.py` | Motor de decisão: sinais BUY/SELL com filtros técnicos, PA e IA |
| `executor.py` | Cliente WebSocket da Deriv, execução de ordens, notificações de velas |
| `risk_manager.py` | Gestão de risco: sizing, stop loss/take profit diário, pausa escalável por losses, detecção de drift |
| `pipeline.py` | Orquestrador: coleta, trend scanning, treino automático, execução e re-treino periódico |
| `transformer_model.py` | Implementação do Temporal Fusion Transformer com Variable Selection Network (PyTorch) |
| `collector.py` | Coleta contínua de ticks via WebSocket com filtro anti-spike |
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

## Modelos de IA

- **RandomForest** (300 árvores, max_depth=10)
- **XGBoost** (300 estimators, learning_rate=0.05)
- **StackingClassifier** (RF + XGB + LogisticRegression)
- **Temporal Fusion Transformer** (PyTorch, 2 camadas, 4 heads, d_model=64, seq_len=50, early stopping)
- Ensemble blend: TFT × 0.55 + Clássico × 0.45
- Discordância TFT/Clássico → não opera
- Score mínimo ponderado (IA + técnico) de 0.55 para abrir operação

### Retreinar

```bash
python dataset_builder.py         # gerar dataset.csv
python train_model.py             # treinar modelos
```

## Configurações Avançadas

### ADX Adaptativo
`ADX_ADAPTIVE = True` ajusta automaticamente `ADX_MIN` ao percentil 40 do histórico recente do símbolo ativo (piso: 15), tornando o filtro de tendência mais preciso do que um valor fixo.

### Duração Dinâmica
O modelo de duração (`duration_model.pkl`) prevê o melhor prazo de expiração entre os candidatos definidos em `CANDIDATE_DURATIONS = [1, 3, 5, 10]` (ticks). Treinado em conjunto com os demais modelos pelo `train_model.py`.

### Score Ponderado
`AI_TECH_WEIGHT = 0.60` e `AI_MODEL_WEIGHT = 0.40` definem o peso do sinal técnico e da IA no score final. Apenas operações com `score ≥ AI_SCORE_MIN (0.55)` são abertas.

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
| `RESUME_ON_WIN` | True | Retoma imediatamente após 1 win durante pausa |
| `DRIFT_WINDOW` | 20 | Janela de trades para calcular win rate recente |
| `DRIFT_WIN_RATE_MIN` | 40% | Alerta se win rate dos últimos N trades cair abaixo |

## Histórico de Mudanças

- `feature_engine.py` criado como fonte única de features (evita duplicação entre `dataset_builder`, `executor` e `ai_predictor`)
- ADX adaptativo por percentil do histórico recente (`ADX_ADAPTIVE`)
- Duração dinâmica prevista por modelo dedicado (`duration_model.pkl`)
- Pausa escalável com `PAUSE_SCALE_FACTOR` e retomada imediata com `RESUME_ON_WIN`
- Detecção de drift com janela deslizante de win rate (`DRIFT_WINDOW`)
- Filtro anti-spike no coletor (`TICK_SPIKE_THRESHOLD`)
- Heartbeat watchdog contra desconexão silenciosa
- Dashboard com autenticação por token (`DASHBOARD_TOKEN`)
- Score ponderado IA vs técnico (`AI_TECH_WEIGHT`, `AI_MODEL_WEIGHT`, `AI_SCORE_MIN`)
- Adicionadas 11 features de Price Action (estrutura de mercado, S/R, supply/demand, FVG)
- Detecção de 6 padrões de vela com notificações em tempo real
- Scores contínuos (-1 a +1) na estratégia ponderada
- Filtro RSI relaxado (permite momentum forte)
- Filtro PA na estratégia (bloqueia sinais contra estrutura)
- Filtro ADX rising (bloqueia quando tendência enfraquece)
- Cadência adaptativa (aumenta intervalo após losses)
- Discordância TFT/Clássico → não opera (sem penalidade)
- Target com threshold de ruído (elimina variações insignificantes)
- Duração por heurística de volatilidade (sem lookahead bias)
- Log em modo append (preserva histórico entre reinícios)
- Janela temporal de 50k ticks para retreino
- Dashboard: painel de padrões de vela com polling automático
- 27 features no total (16 técnicas + 11 PA)