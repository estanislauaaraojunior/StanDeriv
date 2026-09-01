# StanDeriv — Bot de Trading Automatizado para Deriv

Bot de opções binárias para índices sintéticos da plataforma Deriv, com análise técnica, Price Action e inteligência artificial (ensemble ML + Transformer).

---

## 🎯 O Que Faz Este Bot

O **StanDeriv** automatiza operações de trading em opções binárias combinando:

- **Análise Técnica Clássica**: EMA, RSI, MACD, ADX, Bollinger Bands, Momentum
- **Price Action Avançada**: 11 features derivadas de estrutura de mercado, S/R dinâmico, supply/demand, Fair Value Gaps (FVG)
- **Machine Learning Ensemble**: RandomForest + XGBoost + StackingClassifier
- **Deep Learning**: Temporal Fusion Transformer (TFT) com PyTorch para padrões temporais
- **Gestão de Risco Profissional**: Sizing fixo, stop/take profit diário, pausa escalável por losses consecutivos
- **Detecção de Degradação**: Alerta automático quando win rate cai abaixo do limiar

**Resultado prático**: 52-60% de acurácia em índices sintéticos (R_100, BOOM, CRASH) ou Forex quando mercado está aberto.

---

## 📊 Arquitetura

```
┌────────────────────────────────────────────────────────────────┐
│              PIPELINE AUTOMÁTICO COMPLETO                      │
└────────────────────────────────────────────────────────────────┘

1. COLETA DE DADOS
   ├─ collector.py        Ticks em streaming via WebSocket
   └─ ticks.csv          Histórico de preços (~50k ticks)

2. CONSTRUÇÃO DO DATASET
   ├─ dataset_builder.py   27 features (técnicas + Price Action)
   ├─ indicators.py        Indicadores + Price Action
   └─ dataset.csv         Pronto para treino

3. TREINAMENTO
   ├─ train_model.py       4 modelos (RF, XGB, Stacking, TFT)
   ├─ transformer_model.py PyTorch TFT
   └─ model.pkl / transformer_model.pkl / duration_model.pkl

4. EXECUÇÃO TEMPO REAL
   ├─ executor.py         Cliente WebSocket, abertura de ordens
   ├─ strategy.py         Sinal (BUY/SELL) com ponderação IA
   ├─ ai_predictor.py     Ensemble + TFT
   ├─ risk_manager.py     Gestão de risco, sizing, pausas
   └─ operacoes_log.csv   Log de cada trade

5. MONITORAMENTO
   ├─ dashboard/server.py   API Flask (localhost:5055)
   ├─ dashboard/index.html  Interface web interativa
   └─ Padrões de vela com notificações em tempo real

6. ORQUESTRAÇÃO
   └─ pipeline.py         Coordena tudo + retreino automático
```

Para detalhes técnicos completos, veja **[ARCHITECTURE.md](ARCHITECTURE.md)**.

---

## 💾 Requisitos

- **Python 3.10+**
- **Conta Deriv** (demo ou real) com token de API
- ~500 MB para dependências (especialmente PyTorch)

### Instalação Rápida

```bash
git clone https://github.com/estanislauaaraojunior/StanDeriv.git
cd StanDeriv
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# ou: .venv\Scripts\activate  (Windows)

pip install -r requirements.txt
```

Se PyTorch não for necessário (USE_TRANSFORMER=False em config.py):

```bash
pip install -r requirements.txt --no-deps torch
# ou instale apenas: pip install websocket-client scikit-learn xgboost joblib pandas numpy flask
```

---

## 🔑 Configuração Inicial

### 1. Crie o arquivo `.env` na raiz

```env
# Fluxo legado (token WebSocket simples)
DERIV_TOKEN=SEUEIDENTIFICADORDEAPIAQUI
DERIV_APP_ID=1089
DERIV_AUTH_MODE=auto

# Ou fluxo novo (OAuth Bearer PAT)
# DERIV_TOKEN=pat_SEUPATAQUICOMPPREFIXO
# DERIV_APP_ID=SEU_APP_ID_NOVO
# DERIV_AUTH_MODE=auto   (detecta automaticamente, ou force "bearer")
```

**Como obter seu token:**
1. Acesse https://developers.deriv.com
2. Crie um app novo (tipo PAT para fluxo novo)
3. Gere um API Token com escopos: `trade` e `account_manage`
4. Copie para `.env`

### 2. Ajuste `config.py` conforme necessário

```python
# Modo de operação
DEMO_MODE = True                    # False = usar dinheiro real (requer confirmação)
SYMBOL = "R_100"                    # Índice sintético ou forex (ex: "frxEURUSD")

# Risco por trade
STAKE_PCT = 0.01                    # 1% do saldo por operação
STOP_LOSS_PCT = 0.25                # Stop diário: -25% do saldo
TAKE_PROFIT_PCT = 0.50              # Take profit diário: +50% do saldo

# Análise técnica
CANDLE_TIMEFRAME_SEC = 60           # 1 minuto (60s) = velas de 1 min
ADX_MIN = 18                        # Filtro de tendência
RSI_OVERSOLD = 38                   # Limite inferior RSI
RSI_OVERBOUGHT = 62                 # Limite superior RSI

# Inteligência Artificial
USE_AI_MODEL = True                 # Usar modelo ou só indicadores?
AI_SCORE_MIN = 0.30                 # Score mínimo para operar
AI_TECH_WEIGHT = 0.55               # Peso técnico vs IA
AI_MODEL_WEIGHT = 0.45              # Peso modelo

# Deep Learning (TFT)
USE_TRANSFORMER = True              # Usar Temporal Fusion Transformer?
TRANSFORMER_BLEND_WEIGHT = 0.55     # Peso TFT no ensemble

# Pausa escalável
MAX_CONSEC_LOSSES = 3               # Losses antes de pausar
PAUSE_BASE_SEC = 600                # Pausa inicial (10 min)
PAUSE_SCALE_FACTOR = 2              # Dobra a cada loss extra
```

Mais parâmetros em **config.py** (~100 opções tuneáveis).

---

## 🚀 Modo de Uso

### Pipeline Completo (Recomendado)

```bash
# Demo mode — coleta, treina e executa automaticamente
python pipeline.py --demo --balance 1000

# Modo real — requer confirmação no terminal
python pipeline.py --real --balance 500

# Apenas coleta 500 candles históricos antes de treinar
python pipeline.py --demo --history-count 500

# Força retreino mesmo que model.pkl exista
python pipeline.py --demo --force-retrain
```

**O que faz:**
1. Coleta histórico de ticks (CANDLE_TIMEFRAME_SEC segundos por candle)
2. Escaneia tendência (prioriza Forex em horário comercial)
3. Constrói dataset (aguarda MIN_CANDLES=50 velas)
4. Treina modelos (RF, XGB, Stacking, TFT)
5. Inicia bot de execução
6. Monitora degradação e retreina automaticamente

---

### Apenas Bot (Modelo Já Treinado)

```bash
python bot.py --demo
python bot.py --real
```

---

### Apenas Coleta

```bash
python collector.py
# Coleta ticks em ticks.csv continuamente
```

---

### Apenas Treino

```bash
# Gera dataset.csv a partir de ticks.csv
python dataset_builder.py

# Treina modelos e salva em model.pkl, transformer_model.pkl, duration_model.pkl
python train_model.py
```

---

### Dashboard

```bash
# Inicia Flask em localhost:5055
python dashboard/server.py

# Ou via script de inicialização
cd dashboard && bash start.sh
```

Acesse: **http://localhost:5055**

Funcionalidades:
- Status em tempo real (balance, win rate, trades)
- Gráficos de P&L e drawdown
- Histórico de operações
- Notificações de padrões de vela
- Indicadores técnicos (EMA, RSI, ADX, MACD)
- Controles (start, stop, pause)

### Backtesting Offline

```bash
# Backtest simples sobre o histórico coletado em ticks.csv
python backtester.py --ticks ticks.csv --balance 1000

# Compara resultado com e sem o filtro de IA (USE_AI_MODEL)
python backtester.py --ticks ticks.csv --balance 1000 --compare-ai

# Salva relatório em JSON
python backtester.py --ticks ticks.csv --balance 1000 --output backtest_reports/run1.json
```

Reutiliza `strategy.get_signal`, `RiskManager` e `ai_predictor.predict_duration` sobre
velas agregadas de `ticks.csv` (mesma agregação de `executor.py`/`dataset_builder.py`),
sem qualquer conexão com a Deriv. Métricas geradas: total de trades, win rate, retorno
total, profit factor e drawdown máximo.

**Limitações importantes** (não é uma garantia de resultado ao vivo):
- Payout assumido fixo via `BACKTEST_PAYOUT_RATIO` (config.py) — a Deriv só informa o
  payout real por proposal ao vivo, que varia por ativo/duração/mercado.
- Não modela slippage, requotes, latência de rede ou indisponibilidade de duração.
- O stop/take profit diário do `RiskManager` usa a data real do sistema (não a data
  simulada dos candles); em backtests que cobrem vários dias históricos, esses limites
  funcionam como um guard-rail sobre o saldo inicial da simulação inteira, não por dia
  civil simulado.

---

## 📚 Módulos Principais

| Arquivo | Responsabilidade |
|---------|------------------|
| **config.py** | Centraliza ~100 parâmetros de tuning (indicadores, risco, IA, Price Action) |
| **indicators.py** | 16 indicadores técnicos + 11 features Price Action + 6 padrões de vela |
| **feature_engine.py** | Fonte única de 27 features (evita duplicação entre dataset_builder, executor, ai_predictor) |
| **dataset_builder.py** | Extrai features de ticks.csv, calcula target, gera dataset.csv com validação temporal |
| **train_model.py** | Treina RandomForest, XGBoost, StackingClassifier, TFT; promoção champion-challenger automática |
| **ai_predictor.py** | Carrega model.pkl + transformer_model.pkl; ensemble blend com discordância = não opera |
| **strategy.py** | Calcula sinal técnico (BUY/SELL) com ponderação IA e filtros (padrões de vela, Price Action, ADX) |
| **executor.py** | Cliente WebSocket Deriv; abertura de ordens; heartbeat watchdog; notificações em tempo real |
| **risk_manager.py** | Sizing fixo, stop/take profit diário, pausa escalável, detecção de drift, log de trades |
| **pipeline.py** | Orquestrador principal; coleta → treino → execução + retreino automático adaptativamente |
| **transformer_model.py** | Implementação Temporal Fusion Transformer com Variable Selection Network (PyTorch) |
| **collector.py** | Coleta contínua de ticks via WebSocket; filtro anti-spike; armazena em ticks.csv |
| **backtester.py** | Backtesting offline: simula sinais/risco/duração sobre ticks.csv sem conexão com a Deriv |
| **deriv_session.py** | Autenticação Deriv: suporte a token legado + OAuth Bearer (REST + OTP) |
| **dashboard/** | Interface web (Flask + HTML/JS/CSS) com autenticação por token |

---

## 🔧 Price Action (11 Features)

Features normalizadas (-1 a +1) derivadas de velas OHLC sintéticas:

| Feature | Range | Significado |
|---------|-------|-------------|
| `pa_market_structure` | -1 a +1 | HH/HL (bullish) vs LH/LL (bearish) |
| `pa_bos_strength` | -1 a +1 | Força do break of structure |
| `pa_trend_consistency` | -1 a +1 | Sequência de swings alinhados |
| `pa_sr_distance` | 0 a 1 | Distância ao S/R mais próximo |
| `pa_sr_touch_count` | 0 a 1 | Força do nível S/R (número de toques) |
| `pa_sr_position` | -1 a +1 | Perto de suporte (-) vs resistência (+) |
| `pa_demand_zone` | 0 a 1 | Proximidade de demand zone |
| `pa_supply_zone` | 0 a 1 | Proximidade de supply zone |
| `pa_fvg_bullish` | 0 a 1 | Fair Value Gap bullish por preencher |
| `pa_fvg_bearish` | 0 a 1 | Fair Value Gap bearish por preencher |
| `pa_candle_at_sr` | -1 a +1 | Padrão de rejeição em zona S/R |

---

## 🕯️ Padrões de Vela (6 Detectados)

Cada padrão inclui `strength` (0-1) e notificação em tempo real:

1. **Bullish Engulfing** — Reversão de alta (vela branca envolve anterior preta)
2. **Bearish Engulfing** — Reversão de baixa (vela preta envolve anterior branca)
3. **Hammer** — Reversão de alta (corpo pequeno, sombra inferior longa)
4. **Shooting Star** — Reversão de baixa (corpo pequeno, sombra superior longa)
5. **Doji** — Indecisão (corpo mínimo, aberturas/fechamentos iguais)
6. **Three White Soldiers / Three Black Crows** — Continuação forte (3 velas seguidas mesma cor)

---

## 🤖 Modelos de IA

### Modelos Clássicos (Scikit-Learn)

- **RandomForest** — 300 árvores, max_depth=10, balanceado
- **XGBoost** — 300 estimators, learning_rate=0.05, subsample=0.8
- **StackingClassifier** — RF + XGB como base, LogisticRegression como meta-learner

**Melhor modelo é promovido automaticamente** (champion-challenger):
- AUC novo ≥ AUC atual + 0.001
- Acurácia queda máx. 2%
- F1 queda máx. 5%

### Deep Learning (Opcional)

- **Temporal Fusion Transformer (TFT)** — PyTorch
  - Entrada sequencial: 50 ticks históricos (seq_len=50)
  - 2 camadas Transformer, 4 heads de atenção
  - d_model=64, dropout=0.15
  - Early stopping (patience=10)
  - Treinado com AdamW + cosine decay

### Ensemble

- **Blend ponderado**: TFT×0.55 + Clássico×0.45
- **Discordância TFT vs Clássico** → não opera (reduz falsos alarmes)
- Score mínimo: AI_SCORE_MIN = 0.30

---

## 📈 Retreino Adaptativo

O bot monitora continuamente e retreina automaticamente quando:

1. **Win rate recente < 42%** — modelo degradando
2. **Confiança média IA < 55%** — modelo inseguro
3. **>400 novos ticks acumulados** — suficiente novo sinal
4. **Tempo máximo 1h sem retreino** — backstop temporal

```
Gatilho → dataset_builder.py → train_model.py → champion-challenger → novo model.pkl
     ↓ se melhor → substitui e continua operando (sem parada)
```

---

## 🛡️ Gestão de Risco

### Sizing Fixo
```
stake = saldo × STAKE_PCT  (padrão: 1%)
mínimo = $0.35 (limite da Deriv)
```

### Stops Diários
```
Stop Loss    : -25% do saldo do dia
Take Profit  : +50% do saldo do dia
Novo dia     : reset automático à meia-noite
```

### Pausa Escalável
```
Gatilho      : 3 losses consecutivos
Pausa inicial: 10 minutos
Escala       : dobra a cada loss adicional (ex: 10m, 20m, 40m, 80m...)
Retomada     : 1 win durante pausa → retoma imediatamente
```

### Detecção de Drift
```
Janela       : últimos 20 trades
Limiar       : win rate < 40%
Alerta       : "considere retreinar o modelo"
```

---

## ⚙️ Parâmetros Avançados

### ADX Adaptativo (P10)
```python
ADX_ADAPTIVE = True
ADX_ADAPTIVE_PERCENTILE = 50   # percentil 50 do histórico
ADX_ADAPTIVE_MAX = 22.0         # teto (não bloqueia tendência moderada)
```
Ajusta o filtro de tendência automaticamente por símbolo.

### Duração Dinâmica
O modelo `duration_model.pkl` prediz o melhor prazo entre `CANDIDATE_DURATIONS = [5, 15, 30]` minutos baseado em condições de mercado.

### Score Ponderado
```python
USE_WEIGHTED_SIGNAL = True
SIGNAL_SCORE_MIN = 0.05
# Score = EMA×0.30 + MACD×0.25 + Preço×0.20 + Momentum×0.15 + RSI×0.10
```

### Filtro Anti-Spike
```python
TICK_SPIKE_THRESHOLD = 0.05  # Rejeita variações > 5% em relação ao anterior
```
Protege indicadores contra dados ruins.

### Target Lookforward
```python
TARGET_LOOKFORWARD = 2  # Rotula o target com movimento de 2+ velas à frente
```
Captura padrões de médio prazo ao invés de ruído de tick único.

---

## 🔐 Segurança

- ✅ **DEMO_MODE=True por padrão** — evita acidentes iniciais
- ✅ **Confirmação para modo real** — pede confirmação no terminal
- ✅ **Dashboard com autenticação por token** — X-Auth-Token header (DASHBOARD_TOKEN env var)
- ✅ **Histórico de logs** — operacoes_log.csv + model_metrics.json preservados
- ✅ **Tokens no .env** — nunca commit em git; use .gitignore

---

## 📊 Métricas Esperadas

**Em índices sintéticos (R_100, BOOM, CRASH):**
- Acurácia: 52-60%
- Win rate prático: 45-55% (depende de parâmetros)
- Drawdown máximo observado: 15-25%
- Tempo de resposta: ~100-300ms por tick

**Fatores de variação:**
- Período do dia (volatilidade muda)
- Condições de mercado (tendência vs lateral)
- Qualidade dos dados coletados
- Parâmetros de tuning específicos do seu caso

---

## 🐛 Troubleshooting

### "DERIV_TOKEN não definido"
→ Crie `.env` com seu token (veja [Configuração Inicial](#-configuração-inicial))

### "model.pkl não encontrado"
→ Execute `python pipeline.py --demo` ou `python train_model.py`

### "Modelo tem N features, código espera M"
→ Sua lista de features mudou; retreine com `python dataset_builder.py && python train_model.py`

### "WebSocket desconectado"
→ Verifique DERIV_TOKEN, APP_ID e conexão de internet; bot reconecta automaticamente

### "Acurácia baixa"
→ Colete mais dados (500+ ticks mínimo), ajuste CANDLE_TIMEFRAME_SEC (5min recomendado), tuneie EMA_FAST/EMA_SLOW

### "Dashboard não carrega"
→ Certifique-se que `python dashboard/server.py` está rodando; acesse http://localhost:5055

---

## 📝 Histórico de Mudanças

**v1.0** (Atual)
- ✅ 27 features totais (16 técnicas + 11 Price Action)
- ✅ 6 padrões de vela com notificações
- ✅ Ensemble RF + XGB + TFT (PyTorch)
- ✅ ADX adaptativo por percentil
- ✅ Duração dinâmica por volatilidade
- ✅ Pausa escalável com retomada imediata
- ✅ Detecção de drift automática
- ✅ Champion-challenger com critérios rigorosos
- ✅ Score ponderado IA vs técnico
- ✅ Filtro anti-spike
- ✅ Heartbeat watchdog
- ✅ Dashboard com autenticação
- ✅ Retreino adaptativo (win rate, confiança, ticks novos)

---

## 🚀 Próximos Passos

1. **Otimização de hiperparâmetros** — grid search em EMA, ADX, RSI com validação temporal
2. **Suporte a múltiplos símbolos** — 1 modelo para Forex + índices
3. **Quantização de modelos** — compactar model.pkl (menor tamanho)
4. **Alerts avançados** — Telegram, Discord, Email
5. **Portfolio management** — gerenciar múltiplas contas/símbolos
6. **Reinforcement Learning** — ajuste dinâmico de parâmetros (somente após validação rigorosa via `backtester.py`)

> ✅ **Backtest robusto** (`backtester.py`) já implementado — veja a seção [Backtesting Offline](#backtesting-offline).

---

## 📧 Suporte

Dúvidas ou bugs? Abra uma **Issue** no GitHub.

---

**Versão**: 1.0 | **Última atualização**: 2026-08-27 | **Autor**: Stanis
