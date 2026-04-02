# ============================================================
#  config.py — configurações centralizadas do bot Deriv
#  Edite APENAS este arquivo para ajustar o comportamento.
# ============================================================

import os
from dotenv import load_dotenv

load_dotenv()  # carrega variáveis de .env (ignorado se não existir)

# ----- Conta -----
# True  → opera na conta DEMO (seguro — dinheiro virtual)
# False → opera na conta REAL (requer TOKEN de conta real + confirmação no terminal)
DEMO_MODE = True

# ----- Conexão -----
APP_ID = os.environ.get("DERIV_APP_ID", "1089")
TOKEN  = os.environ["DERIV_TOKEN"]    # defina em .env — crie em: developers.deriv.com

# ----- Instrumento -----
SYMBOL        = "R_100"  # índice sintético 24/7 (sem impacto de notícias)

# Símbolo ativo (pode ser alterado pelo scan de tendência em runtime)
_active_symbol: str = SYMBOL


def get_active_symbol() -> str:
    """Retorna o símbolo ativo atual (pode diferir de SYMBOL se o scan o alterou)."""
    return _active_symbol


def set_active_symbol(symbol: str) -> None:
    """Atualiza o símbolo ativo em runtime (chamado pelo pipeline após scan)."""
    global _active_symbol
    _active_symbol = symbol
DURATION      = 5        # duração de fallback (usada quando o modelo ainda não existe)
DURATION_UNIT = "t"      # "t" = ticks | "s" = segundos | "m" = minutos
BASIS         = "stake"  # base do contrato
CURRENCY      = "USD"

# ----- Duração dinâmica (escolhida pela IA) -----
# Durações candidatas (em ticks) que o modelo de duração pode prever.
# Adicione ou remova valores para ampliar/restringir o leque de opções.
CANDIDATE_DURATIONS = [1, 3, 5, 10]
DURATION_MODEL_PATH = "duration_model.pkl"

# ----- Parâmetros dos indicadores -----
EMA_FAST     = 9
EMA_SLOW     = 21
RSI_PERIOD   = 14
MACD_FAST    = 12
MACD_SLOW    = 26
MACD_SIGNAL  = 9
ADX_PERIOD   = 14
BB_PERIOD    = 20
BB_STD       = 2.0

# ----- Price Action -----
CANDLE_SIZE            = 10      # ticks por vela sintética
CANDLE_LOOKBACK        = 5       # velas mínimas para padrões
PA_SR_TOLERANCE        = 0.001   # tolerância % para clustering S/R
TARGET_NOISE_THRESHOLD = 0.0001  # variação mínima para classificar target
# Horizonte padrão do target no dataset: 1=próximo tick, N>1=média dos próximos N ticks.
# Valores maiores tendem a reduzir ruído de microestrutura.
TARGET_LOOKFORWARD     = 5
CANDLE_NOTIFY          = True    # notificações de padrões de vela no terminal

# ----- Filtros de entrada -----
RSI_OVERSOLD    = 35   # abaixo → sobrevendido (não abre SELL aqui)
RSI_OVERBOUGHT  = 65   # acima → sobrecomprado (não abre BUY aqui)
ADX_MIN         = 20   # abaixo → mercado lateral → sem entrada

# ----- ADX adaptativo (P10) -----
# True  → ADX_MIN ajustado ao percentil do histórico recente (mais preciso por símbolo)
# False → usa ADX_MIN fixo acima
ADX_ADAPTIVE            = True
ADX_ADAPTIVE_PERCENTILE = 40   # percentil do histórico de ADX; piso = 15

# ----- Gestão de risco -----
STAKE_PCT         = 0.01   # 1% do saldo por operação
STOP_LOSS_PCT     = 0.25   # -25% do saldo diário → para o dia
TAKE_PROFIT_PCT   = 0.50   # +50% do saldo diário → para o dia
MAX_CONSEC_LOSSES = 3      # losses consecutivos antes de pausar

# Pausa escalável: base * scale_factor^(losses_extras) — cap de 2h (P8)
PAUSE_BASE_SEC     = 600   # 10 min de pausa base (1º gatilho)
PAUSE_SCALE_FACTOR = 2     # dobra a pausa a cada loss além do limite
RESUME_ON_WIN      = True  # retoma pausa imediatamente após 1 win

# ----- Aquecimento -----
MIN_TICKS = 50  # ticks mínimos antes de operar (garante indicadores estáveis)

# ----- Buffer de preços (P12) -----
PRICE_BUFFER_SIZE = 500  # ticks mantidos em memória para indicadores e IA

# ----- Cadência de entradas -----
# Primeira entrada: imediata após MIN_TICKS (modelo já treinado com histórico).
# Entradas seguintes: somente após acumular ENTRY_TICK_INTERVAL novos ticks.
ENTRY_TICK_INTERVAL = 100  # ticks entre entradas

# ----- Timeout de operações (P1, P7) -----
PROPOSAL_TIMEOUT_SEC = 10   # segundos sem resposta da API antes de cancelar proposal

# ----- Heartbeat / watchdog (P9) -----
HEARTBEAT_TIMEOUT_SEC = 30  # segundos sem tick antes de alertar

# ----- Coletor — qualidade de dados (P2) -----
TICK_SPIKE_THRESHOLD = 0.05  # rejeitar ticks com variação > 5% em relação ao anterior

# ----- Inteligência Artificial -----
# USE_AI_MODEL = False  → bot funciona exatamente como antes (só indicadores)
# USE_AI_MODEL = True   → IA pondera o sinal dos indicadores antes de operar
USE_AI_MODEL       = True
AI_MODEL_PATH      = "model.pkl"
AI_CONFIDENCE_MIN  = 0.50   # limiar mínimo realista para modelo com ~52% de acurácia

# Ponderação IA vs técnico (P4) — substitui gate duro por score suavizado
AI_TECH_WEIGHT  = 0.60   # peso do sinal técnico no score final
AI_MODEL_WEIGHT = 0.40   # peso do sinal da IA no score final
AI_SCORE_MIN    = 0.52   # score mínimo ponderado para aceitar a operação

# ----- Sinal ponderado (P14) -----
# False = comportamento original (AND rígido — menos entradas, mais seletivo)
# True  = score ponderado por indicador (mais entradas, requer ajuste fino)
USE_WEIGHTED_SIGNAL = False
SIGNAL_SCORE_MIN    = 0.65  # limiar mínimo do score técnico ponderado

# ----- Detecção de drift (P13) -----
DRIFT_WINDOW       = 20    # janela de trades para calcular win rate recente
DRIFT_WIN_RATE_MIN = 0.40  # alerta se win rate dos últimos N trades < 40%

# ----- Promoção de modelo (champion-challenger) -----
# Challenger só substitui o champion quando superar esses critérios mínimos.
MODEL_PROMOTION_MIN_AUC_DELTA = 0.002  # AUC_novo >= AUC_atual + delta
MODEL_PROMOTION_MAX_ACC_DROP  = 0.005  # queda máxima tolerada de acurácia
MODEL_PROMOTION_MAX_F1_DROP   = 0.005  # queda máxima tolerada de F1

# ----- Arquivos de log -----
TICKS_CSV        = "ticks.csv"
OPERATIONS_LOG   = "operacoes_log.csv"
DATASET_CSV      = "dataset.csv"

# ─────────────────────────────────────────────────────────────────
#  Temporal Fusion Transformer (TFT) — nível hedge fund
# ─────────────────────────────────────────────────────────────────

# True  → treina e usa TFT em ensemble com RF/XGB (requer torch)
# False → comportamento idêntico ao anterior (sem torch necessário)
USE_TRANSFORMER          = True
TRANSFORMER_MODEL_PATH   = "transformer_model.pkl"

# Blend do ensemble: conf_final = TFT*BLEND + classical*(1-BLEND)
# 0.55 → TFT tem peso levemente maior; ajuste conforme performance observada
TRANSFORMER_BLEND_WEIGHT = 0.55

# Janela temporal: número de vetores de features passados ao TFT como sequência.
# Valores maiores capturam padrões de longo prazo, mas requerem mais dados.
# Requer PRICE_BUFFER_SIZE >= TRANSFORMER_SEQ_LEN + 100 (padrão: 500 > 150 ✓)
TRANSFORMER_SEQ_LEN = 50

# Arquitetura interna do TFT
TRANSFORMER_D_MODEL  = 64    # dimensão dos embeddings internos
TRANSFORMER_N_HEADS  = 4     # cabeças de atenção (d_model deve ser divisível por n_heads)
TRANSFORMER_N_LAYERS = 2     # camadas do Transformer Encoder
TRANSFORMER_DROPOUT  = 0.15  # dropout (maior = mais regularização)

# Treinamento
TRANSFORMER_EPOCHS     = 20    # épocas máximas (early stopping pode parar antes)
TRANSFORMER_BATCH_SIZE = 128   # batch size (reduzir para 64 se pouca RAM)
TRANSFORMER_LR         = 3e-4  # learning rate inicial (AdamW + cosine decay)
TRANSFORMER_PATIENCE   = 10    # épocas sem melhoria antes do early stopping
