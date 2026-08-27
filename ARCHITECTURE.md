# StanDeriv — Arquitetura Completa

## Visão Geral do Sistema

**StanDeriv** é um bot de trading automatizado para opções binárias (índices sintéticos da Deriv) que combina:
- **Análise técnica clássica**: EMA, RSI, MACD, ADX, Bollinger Bands, Momentum
- **Price Action avançada**: 11 features (estrutura de mercado, S/R dinâmico, supply/demand, FVG, padrões de vela)
- **Machine Learning em ensemble**: RandomForest + XGBoost + StackingClassifier
- **Deep Learning**: Temporal Fusion Transformer (TFT) com PyTorch
- **Gestão de risco profissional**: Sizing fixo, stop/take profit diário, pausa escalável

O bot opera **24/7** em índices sintéticos (R_100, BOOM, CRASH) via WebSocket, ou prioriza mercados reais (Forex) durante horários de abertura.

---

## Arquitetura de Componentes

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          PIPELINE PRINCIPAL                              │
└──────────────────────────────────────────────────────────────────────────┘

1. COLETA DE DADOS
   ├─ collector.py        Coleta contínua de ticks via WebSocket
   │  └─ deriv_session.py  Suporte a dois fluxos de autenticação (legado + OAuth Bearer)
   └─ ticks.csv          Histórico de preços (~50k ticks)

2. CONSTRUÇÃO DO DATASET
   ├─ dataset_builder.py   Extrai 27 features dos ticks
   ├─ feature_engine.py    Fonte única de features (centralizado)
   ├─ indicators.py        Cálculo de indicadores técnicos + Price Action
   └─ dataset.csv         Arquivo de treino (~1k-10k amostras)

3. TREINAMENTO DO MODELO
   ├─ train_model.py       Treina 4 modelos (RF, XGB, Stacking, TFT)
   ├─ transformer_model.py Implementação do Temporal Fusion Transformer
   ├─ model.pkl           Melhor modelo clássico (persistido)
   ├─ transformer_model.pkl TFT treinado (persistido)
   ├─ duration_model.pkl   Modelo para duração dinâmica (persistido)
   └─ model_metrics.json   Histórico de métricas de todos os treinos

4. EXECUÇÃO EM TEMPO REAL
   ├─ executor.py         Cliente WebSocket Deriv, abertura de operações
   ├─ strategy.py         Motor de decisão (sinal BUY/SELL)
   ├─ ai_predictor.py     Inferência em tempo real (RF/XGB + TFT ensemble)
   ├─ risk_manager.py     Controle de risco, sizing, pausas
   ├─ operacoes_log.csv   Log de cada operação (resultado, indicadores)
   └─ risk_state.json     Estado de risco para restauração

5. MONITORAMENTO
   ├─ dashboard/server.py   API Flask (porta 5055)
   ├─ dashboard/app.js      Frontend interativo
   ├─ dashboard/index.html  Interface HTML
   └─ dashboard/style.css   Estilo CSS

6. ORQUESTRAÇÃO
   └─ pipeline.py         Orquestrador principal: coleta → treino → execução
```

---

## Fluxo de Dados Completo

### Fase 1: Coleta e Preparação

```
WebSocket (Deriv)
    ↓
collector.py (ticks em tempo real)
    ↓
Validação anti-spike (↓5% variação)
    ↓
ticks.csv (append)
    ↓
MIN_CANDLES = 50 velas de aquecimento
```

**Responsáveis:**
- `collector.py`: Conexão WebSocket, tratamento de desconexões
- `deriv_session.py`: Suporte a legado (token WebSocket) e OAuth Bearer (REST + OTP)
- `config.py:TICK_SPIKE_THRESHOLD`: Rejeita outliers (proteção contra spikes de dados)

---

### Fase 2: Construção do Dataset

```
ticks.csv
    ↓
indicators.py + feature_engine.py
    ├─ 16 features técnicas (EMA, RSI, MACD, ADX, BB, Momentum, etc.)
    ├─ 11 features de Price Action (estrutura, S/R, supply/demand, FVG, padrões)
    └─ target (label para treino): sobe/desce nos próximos TARGET_LOOKFORWARD candles
    ↓
dataset_builder.py (agrupa em velas de tempo: CANDLE_TIMEFRAME_SEC segundos)
    ↓
dataset.csv (~1k-10k amostras)
```

**Features totais: 27**
- 16 técnicas: `ema9`, `ema21`, `rsi`, `adx`, `macd_line`, `macd_hist`, `bb_upper`, `bb_mid`, `bb_lower`, `momentum`, etc.
- 11 Price Action: `pa_market_structure`, `pa_bos_strength`, `pa_trend_consistency`, `pa_sr_distance`, `pa_sr_touch_count`, `pa_sr_position`, `pa_demand_zone`, `pa_supply_zone`, `pa_fvg_bullish`, `pa_fvg_bearish`, `pa_candle_at_sr`

**Responsáveis:**
- `feature_engine.py`: Centraliza `compute_feature_map()` e `extract_feature_vector()` — evita duplicação entre dataset_builder, executor e ai_predictor
- `indicators.py`: Implementação de todos os indicadores + Price Action
- `dataset_builder.py`: Leitura de ticks.csv, agrupamento em velas, geração do target com `TARGET_LOOKFORWARD`

---

### Fase 3: Treinamento

```
dataset.csv (treino + teste com gap temporal)
    ↓
train_model.py (4 arquiteturas em paralelo)
    ├─ RandomForest (300 árvores, max_depth=10)
    ├─ XGBoost (300 estimators, lr=0.05)
    ├─ StackingClassifier (RF+XGB base, LogisticRegression meta)
    └─ TFT (Temporal Fusion Transformer, PyTorch)
    ↓
Avaliação: Acurácia, Precision, Recall, F1, ROC-AUC
    ├─ model.pkl (melhor clássico por AUC)
    ├─ transformer_model.pkl (TFT se USE_TRANSFORMER=True)
    ├─ duration_model.pkl (RF dedicado para duração dinâmica)
    └─ model_metrics.json (histórico de métricas)
```

**Responsáveis:**
- `train_model.py`: Orquestrador do treino, avaliação, promoção champion-challenger
- `transformer_model.py`: Implementação do Temporal Fusion Transformer com Variable Selection Network
- `config.py`: Hiperparâmetros de cada modelo

**Validação temporal (P11):**
- Sem shuffle → preserva série temporal
- Gap de 5% entre treino e teste → evita data leakage

---

### Fase 4: Execução em Tempo Real

```
executor.py (WebSocket client)
    ├─ Coleta ticks em streaming
    ├─ Agrupa em velas de tempo (CANDLE_TIMEFRAME_SEC = 60s por padrão)
    ├─ MIN_CANDLES = 50 aquecimento
    └─ A cada nova vela fechada:
        │
        ├─ strategy.py (get_signal)
        │  ├─ Calcula todos os indicadores técnicos
        │  ├─ ADX adaptativo (percentil 50 do histórico recente, P10)
        │  ├─ Filtro de padrões de vela (soft conflict, P3)
        │  ├─ Sinal ponderado ou AND rígido (P14)
        │  └─ Retorna ("BUY" | "SELL" | None, dict indicadores)
        │
        ├─ ai_predictor.py (predict)
        │  ├─ Carrega model.pkl (melhor clássico) + transformer_model.pkl (TFT)
        │  ├─ Ensemble blend: TFT×0.55 + Clássico×0.45 (P5)
        │  ├─ Discordância → não opera (sem penalidade)
        │  └─ Retorna (direção, confiança)
        │
        ├─ Ponderação IA vs técnico (P4)
        │  └─ score = AI_TECH_WEIGHT×1.0 + AI_MODEL_WEIGHT×conf
        │  └─ score ≥ AI_SCORE_MIN (0.30) → autoriza operação
        │
        ├─ risk_manager.py (can_trade, get_stake)
        │  ├─ can_trade(): verifica stop/take profit diário, pausa escalável
        │  ├─ get_stake(): STAKE_PCT % do saldo (1% por padrão)
        │  └─ record_result(): atualiza saldo, log, detecção de drift (P13)
        │
        ├─ duration_model.pkl (optional)
        │  └─ Prediz duração ótima entre CANDIDATE_DURATIONS = [5, 15, 30] min
        │
        └─ Abre contrato Deriv
            ├─ symbol (R_100, BOOM, CRASH, ou Forex se em horário comercial)
            ├─ direction (BUY/UP | SELL/DOWN)
            ├─ stake (calculado por risk_manager)
            ├─ duration (model ou fallback)
            └─ Escuta resultado na próxima vela
```

**Responsáveis:**
- `executor.py`: Cliente WebSocket, abertura de operações, tratamento de eventos
- `strategy.py`: Cálculo do sinal técnico
- `ai_predictor.py`: Inferência do modelo, ensemble
- `risk_manager.py`: Controle de risco, sizing, pausas escaláveis, detecção de drift
- `config.py`: Centenas de parâmetros de tuning

---

## Componentes Principais

### 1. **collector.py** — Coleta contínua de ticks
```python
DerivCollector:
  - Conecta via WebSocket (legado ou Bearer)
  - Escuta ticks em streaming
  - Filtra spikes (TICK_SPIKE_THRESHOLD = 5%)
  - Salva em ticks.csv (append)
  - Heartbeat watchdog (HEARTBEAT_TIMEOUT_SEC = 30s)
```

### 2. **executor.py** — Cliente WebSocket e motor de execução
```python
DerivBot:
  - Autentica via legado (token WebSocket) ou OAuth Bearer (REST + OTP)
  - Conecta ao WebSocket Deriv
  - Recebe ticks em tempo real
  - Avalia sinal a cada vela fechada
  - Abre/fecha contratos
  - Notificações em tempo real (terminal + dashboard)
  - Detecção de padrões de vela (6 tipos com força 0-1)
```

### 3. **indicators.py** — Banco de indicadores técnicos + Price Action
```python
# Técnicos clássicos
ema(prices, period)          → média móvel exponencial
rsi(prices, period)          → índice de força relativa
macd(prices, ...)            → MACD + sinal + histograma
adx(prices, period)          → índice de força de tendência
bollinger(prices, ...)       → bandas superior/inferior/média
momentum(prices, period)     → taxa de mudança

# Price Action (11 features normalizadas -1 a +1)
price_action_features(candles, sr_tolerance)
  ├─ pa_market_structure     (HH/HL bullish vs LH/LL bearish)
  ├─ pa_bos_strength         (força do break of structure)
  ├─ pa_trend_consistency    (sequência de swings alinhados)
  ├─ pa_sr_distance          (distância ao S/R mais próximo)
  ├─ pa_sr_touch_count       (força do nível S/R — toques)
  ├─ pa_sr_position          (perto de suporte vs resistência)
  ├─ pa_demand_zone          (proximidade de demand zone)
  ├─ pa_supply_zone          (proximidade de supply zone)
  ├─ pa_fvg_bullish          (Fair Value Gap bullish por preencher)
  ├─ pa_fvg_bearish          (Fair Value Gap bearish por preencher)
  └─ pa_candle_at_sr         (padrão de rejeição em zona S/R)

# Padrões de vela (6 detectados com força)
detect_candle_patterns(ohlc_candle)
  ├─ Bullish Engulfing       (reversão de alta)
  ├─ Bearish Engulfing       (reversão de baixa)
  ├─ Hammer                  (reversão de alta — sombra inferior)
  ├─ Shooting Star           (reversão de baixa — sombra superior)
  ├─ Doji                    (indecisão — corpo mínimo)
  └─ 3 White Soldiers / 3 Black Crows  (continuação)
```

### 4. **feature_engine.py** — Centralização de features
```python
# Fonte única de features (evita duplicação)
FEATURES = [...27 features...]

compute_feature_map(prices, candles, ...)
  → dict com todos os indicadores calculados

extract_feature_vector(feature_map)
  → numpy array [27 features] para treino/inferência

# Usado por:
#  - dataset_builder.py (treino)
#  - executor.py (tempo real)
#  - ai_predictor.py (inferência)
```

### 5. **strategy.py** — Motor de decisão
```python
get_signal(prices, adx_min, adx_history, candle_alerts)
  │
  ├─ Calcula todos os indicadores
  ├─ ADX adaptativo (P10): percentil do histórico recente, piso 15
  │
  ├─ Filtro ADX rising: bloqueia quando tendência enfraquece
  ├─ Filtro RSI: bloqueia sinais contra extremos
  │
  ├─ [SE USE_WEIGHTED_SIGNAL] Score ponderado (P14)
  │  └─ EMA 30% + MACD 25% + Preço 20% + Momentum 15% + RSI 10%
  │
  ├─ [SE USE_AI_MODEL] _apply_ai_filter (P4)
  │  ├─ Filtro de padrões de vela (soft conflict, P3)
  │  ├─ Filtro Price Action
  │  ├─ Carrega IA (RF + XGB + TFT)
  │  └─ score = AI_TECH×1.0 + AI_MODEL×conf
  │  └─ score ≥ AI_SCORE_MIN → autoriza
  │
  └─ Retorna ("BUY" | "SELL" | None, indicadores)
```

### 6. **ai_predictor.py** — Inferência em tempo real
```python
predict(prices)
  │
  ├─ Carrega model.pkl (RF, XGB ou Stacking)
  ├─ Carrega transformer_model.pkl (TFT)
  │
  ├─ Ensemble blend (P5)
  │  ├─ TFT confidence (confiança da predição)
  │  ├─ Classical confidence (max(proba))
  │  └─ conf_final = TFT×0.55 + Classical×0.45
  │
  └─ Retorna (direção, confiança)
```

### 7. **risk_manager.py** — Controle de risco
```python
RiskManager:
  ├─ Sizing fixo: STAKE_PCT % do saldo (1% por padrão)
  ├─ Stop diário: STOP_LOSS_PCT (25% de perda = para o dia)
  ├─ Take profit diário: TAKE_PROFIT_PCT (50% de lucro = para o dia)
  │
  ├─ Pausa escalável (P8)
  │  ├─ MAX_CONSEC_LOSSES = 3 losses → pausa
  │  ├─ PAUSE_BASE_SEC = 600s (10 min)
  │  ├─ PAUSE_SCALE_FACTOR = 2 (dobra a pausa a cada loss extra)
  │  └─ RESUME_ON_WIN = True (retoma imediatamente após 1 win)
  │
  ├─ Detecção de drift (P13)
  │  ├─ DRIFT_WINDOW = 20 trades
  │  ├─ DRIFT_WIN_RATE_MIN = 40%
  │  └─ Alerta quando win rate cai abaixo do limiar
  │
  ├─ Log completo (operacoes_log.csv)
  │  └─ timestamp, symbol, direction, stake, duration, resultado, profit, etc.
  │
  └─ Estado persistido (risk_state.json)
     └─ Restaura consec_losses, pause_until, recent_results ao reiniciar
```

### 8. **train_model.py** — Orquestrador de treino
```python
train(dataset_path, output_path, test_ratio)
  │
  ├─ Lê dataset.csv
  ├─ Split temporal (treino / gap / teste)
  │
  ├─ Treina 3 modelos clássicos
  │  ├─ RandomForest (300 árvores, max_depth=10)
  │  ├─ XGBoost (300 estimators, lr=0.05)
  │  └─ StackingClassifier (RF+XGB base, LogisticRegression meta)
  │
  ├─ [SE USE_TRANSFORMER] Treina TFT
  │  ├─ Converte dataset plano em janelas temporais (seq_len=50)
  │  ├─ TFTPredictor (PyTorch, 2 camadas, 4 heads, d_model=64)
  │  └─ Early stopping (patience=10)
  │
  ├─ Avaliação: Acurácia, Precision, Recall, F1, ROC-AUC
  ├─ Champion-Challenger (P6)
  │  ├─ MODEL_PROMOTION_MIN_AUC_DELTA = 0.001
  │  ├─ MODEL_PROMOTION_MAX_ACC_DROP = 0.02
  │  └─ MODEL_PROMOTION_MAX_F1_DROP = 0.05
  │
  ├─ [SE has_duration] Treina modelo de duração (RF dedicado)
  │
  └─ Salva model.pkl, transformer_model.pkl, duration_model.pkl
     └─ Atualiza model_metrics.json (histórico de treinos)
```

### 9. **transformer_model.py** — Temporal Fusion Transformer
```python
TFTPredictor (PyTorch):
  │
  ├─ VariableSelectionNetwork (seleciona features relevantes)
  ├─ Transformer Encoder (2 camadas, 4 heads, d_model=64)
  ├─ Processamento sequencial (seq_len=50)
  │
  ├─ Treinamento
  │  ├─ AdamW optimizer + cosine learning rate decay
  │  ├─ Early stopping (patience=10, validation loss)
  │  └─ Batch size = 128
  │
  └─ Predição
     ├─ predict_proba(X_seq) → shape (N, 2)
     ├─ Calibração com classes desbalanceadas
     └─ Ensemble com modelos clássicos
```

### 10. **pipeline.py** — Orquestrador principal
```python
pipeline(args)
  │
  ├─ [1] Coleta inicial de ticks (collector.py)
  ├─ [2] Trend scanning (avalia múltiplos símbolos)
  │      └─ Prioriza Forex em horário comercial (PREFER_REAL_MARKETS)
  ├─ [3] Treino automático (dataset_builder.py + train_model.py)
  ├─ [4] Execução em tempo real (executor.py)
  ├─ [5] Monitoramento de degradação (drift detection)
  ├─ [6] Retreino periódico (RETRAIN_*_TRIGGER)
  │
  └─ Loop infinito
     ├─ Monitora win rate, confiança da IA, novos ticks
     ├─ Dispara retreino se (win_rate < 42% OU conf < 55% OU >400 ticks novos)
     ├─ Promoção champion-challenger automática
     └─ Retoma execução após retreino
```

### 11. **dashboard/server.py** — API Flask
```python
Flask app (localhost:5055):
  │
  ├─ GET /api/bot/status
  │  └─ Estado atual do bot (running, paused, balance, etc.)
  ├─ GET /api/bot/stats
  │  └─ Estatísticas de trades (win rate, drawdown, etc.)
  ├─ GET /api/operations
  │  └─ Últimas operações com indicadores
  ├─ GET /api/candle_patterns
  │  └─ Padrões de vela recentes com força
  ├─ POST /api/bot/control
  │  └─ Controles (start, stop, pause)
  │
  └─ Autenticação por token (X-Auth-Token header)
     └─ DASHBOARD_TOKEN env var
```

### 12. **dashboard/index.html** — Interface web
```html
<!-- Layout responsivo -->
├─ Painel de status em tempo real
├─ Gráfico de balance e drawdown
├─ Tabela de operações recentes
├─ Notificações de padrões de vela
├─ Indicadores técnicos (EMA, RSI, ADX, MACD)
├─ Controles (start, stop, pause, settings)
└─ Polling automático (WebSocket não é viável no REST)
```

---

## Fluxo de Autenticação Deriv

### Fluxo Legado (token WebSocket)
```
1. DERIV_TOKEN = <token da WebSocket API>
2. executor.py → DerivBot.authorize()
3. WebSocket → {"authorize": "<token>"}
4. Sucesso → connected
```

### Fluxo Novo (OAuth Bearer + REST + OTP)
```
1. DERIV_TOKEN = "pat_..." ou "ory_at_..."
2. deriv_session.py → setup_bearer_options_session()
   ├─ REST: GET /trading/v1/options/accounts (lista contas)
   ├─ Se não houver: POST /trading/v1/options/accounts (cria)
   ├─ Seleciona conta demo/real
   ├─ REST: POST /options/accounts/{account_id}/otp (obtém URL WebSocket)
   └─ Retorna URL com autenticação embutida
3. executor.py → WebSocket URL com autenticação
4. Sucesso → connected
```

**Autoswitching:**
- `DERIV_AUTH_MODE=auto` detecta automaticamente pelo prefixo do token
- `DERIV_AUTH_MODE=bearer` força OAuth Bearer
- `DERIV_AUTH_MODE=legacy` força WebSocket legado

---

## Mecanismos de Adaptatividade

### P10: ADX Adaptativo
```python
get_adaptive_adx_min(adx_history)
  ├─ Calcula percentil 50 do histórico recente (últimas 50 velas)
  ├─ Piso: 15 (não fica muito fraco)
  ├─ Teto: ADX_ADAPTIVE_MAX = 22 (não bloqueia tendência moderada)
  └─ Resultado: filtro de tendência personalizado por símbolo
```

### P11: Gap Temporal no Split
```
Dataset split:
  ├─ Treino: 0 a 80% (sem shuffle, preserva série)
  ├─ Gap: 80% a 85% (5% de amostras ignoradas)
  └─ Teste: 85% a 100%
  
Propósito: evita data leakage (informação do futuro vazando no treino)
```

### P13: Detecção de Drift
```python
_check_drift():
  ├─ Janela deslizante dos últimos N trades (DRIFT_WINDOW=20)
  ├─ Se win_rate < DRIFT_WIN_RATE_MIN (40%)
  └─ → Alerta para retreinar o modelo

Benefício: detecta degradação antes de perdas catastróficas
```

### P14: Sinal Ponderado
```python
_weighted_signal():
  ├─ USE_WEIGHTED_SIGNAL = True (ativa)
  ├─ Score = EMA×0.30 + MACD×0.25 + Preço×0.20 + Momentum×0.15 + RSI×0.10
  ├─ Modulado por ADX (confiança aumenta em tendência)
  └─ SIGNAL_SCORE_MIN = 0.05 (limiar suavizado)

Vantagem: menos binário, mais nuances, mais entradas
```

---

## Parâmetros Críticos (config.py)

| Parâmetro | Padrão | Impacto |
|-----------|--------|--------|
| `DEMO_MODE` | True | Segurança: operar em demo vs real |
| `SYMBOL` | R_100 | Qual índice/forex tradear |
| `CANDLE_TIMEFRAME_SEC` | 60 | Frequência de entradas (segundos) |
| `STAKE_PCT` | 0.01 | Risco por trade (1% do saldo) |
| `ADX_MIN` | 18 | Limiar de tendência (adaptativo se P10) |
| `RSI_OVERSOLD` | 38 | Limite inferior RSI (não vende muito) |
| `RSI_OVERBOUGHT` | 62 | Limite superior RSI (não compra muito) |
| `AI_SCORE_MIN` | 0.30 | Confiança mínima para operar |
| `AI_TECH_WEIGHT` | 0.55 | Peso técnico vs IA (55% vs 45%) |
| `TRANSFORMER_BLEND_WEIGHT` | 0.55 | Peso TFT no ensemble (55% vs 45% clássico) |
| `MAX_CONSEC_LOSSES` | 3 | Losses antes de pausar |
| `PAUSE_BASE_SEC` | 600 | Pausa base (10 min) |
| `PAUSE_SCALE_FACTOR` | 2 | Fator de escala (dobra a cada loss) |

---

## Fluxo de Retreino Adaptativo

```
Condições que acionam retreino (monitoradas continuamente):

1. Win rate recente cai < RETRAIN_WIN_RATE_TRIGGER (42%)
2. Confiança média da IA < RETRAIN_CONFIDENCE_TRIGGER (55%)
3. >RETRAIN_NEW_TICKS_TRIGGER (400) novos ticks desde último treino
4. Tempo máximo sem treino > RETRAIN_MAX_INTERVAL_SEC (1h)

Quando acionado:
  ├─ dataset_builder.py (lê ticks.csv, gera novo dataset.csv)
  ├─ train_model.py (treina RF, XGB, Stacking, TFT)
  ├─ Champion-Challenger (compara com modelo anterior)
  ├─ Se melhor → promove novo modelo
  ├─ Aguarda RETRAIN_MIN_INTERVAL_SEC (5 min) antes de próximo treino
  └─ Retoma execução
```

---

## Limitações e Considerações

### Performance
- **WebSocket latência**: ~100-300ms por tick em índices sintéticos
- **Processamento**: ~10-50ms por vela (cálculo de indicadores + IA)
- **Treino TFT**: ~2-5 min com 1k amostras e seq_len=50

### Precisão
- **Acurácia esperada**: 52-60% em índices sintéticos (ruído de mercado)
- **Win rate prático**: 45-55% (depende de parâmetros e condições de mercado)
- **Drawdown máximo observado**: 15-25% em períodos de adversidade

### Segurança
- **Tokens**: Nunca commit no .env; use variáveis de ambiente
- **Demo vs Real**: DEMO_MODE=True por padrão (segurança)
- **Dashboard**: Autenticação por token (X-Auth-Token)

---

## Próximos Passos para Desenvolvimento

1. **Otimização de Hiperparâmetros**: Grid search em EMA_FAST, EMA_SLOW, ADX_MIN
2. **Suporte a múltiplos símbolos**: Treinar modelo único para Forex, índices sintéticos, ações
3. **Quantização de modelos**: Reduzir tamanho de model.pkl, transformer_model.pkl
4. **Backtest robusto**: Framework completo para validar estratégias antes de produção
5. **Alerts avançados**: Integração com Telegram, Discord, Email
6. **Portfolio management**: Gerenciar múltiplas contas/símbolos simultaneamente
7. **Reinforcement Learning**: Ajuste dinâmico de parâmetros via RL (futuro)

---

**Versão**: 1.0 | **Última atualização**: 2026-08-27
