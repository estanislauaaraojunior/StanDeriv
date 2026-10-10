"""Configurações centralizadas do bot Deriv."""
# ============================================================
#  config.py — configurações centralizadas do bot Deriv
#  Edite APENAS este arquivo para ajustar o comportamento.
# ============================================================

import os
import threading

from dotenv import load_dotenv


load_dotenv()  # carrega variáveis de .env (ignorado se não existir)


def _env_bool(name, default):
    """Lê variável de ambiente como booleano (1/true/yes/on/sim); retorna `default` se ausente."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on", "sim"}


# ----- Conta -----
# True  → opera na conta DEMO (seguro — dinheiro virtual)
# False → opera na conta REAL (requer TOKEN de conta real + confirmação no terminal)
DEMO_MODE = True

# Limites comuns à simulação, conta demo e conta real.
MIN_NET_PAYOUT = 0.60
MIN_APPROVAL_WIN_RATE = 0.45


# ----- Conexão -----
_RAW_APP_ID = os.environ.get("DERIV_APP_ID", "1089").strip()
_RAW_TOKEN = os.environ.get("DERIV_TOKEN", "").strip()

DERIV_AUTH_MODE = os.environ.get(
    "DERIV_AUTH_MODE",
    "auto",
).strip().lower()

_TOKEN_PREFIXES = ("pat_", "ory_at_")


# Protege contra inversão acidental no .env:
#   DERIV_APP_ID=pat_...
#   DERIV_TOKEN=<app_id>
if (
    _RAW_APP_ID.startswith(_TOKEN_PREFIXES)
    and not _RAW_TOKEN.startswith(_TOKEN_PREFIXES)
):
    APP_ID = _RAW_TOKEN
    TOKEN = _RAW_APP_ID

    print(
        "[CONFIG] Aviso: DERIV_APP_ID e DERIV_TOKEN parecem "
        "invertidos no .env; usando troca automática."
    )
else:
    APP_ID = _RAW_APP_ID
    TOKEN = _RAW_TOKEN


# ----- Instrumento -----
SYMBOL = "R_100"  # índice sintético 24/7

# Símbolo ativo (pode ser alterado pelo scan de tendência em runtime)
_active_symbol: str = SYMBOL

# Lock para alterações/leitura do símbolo ativo.
_symbol_lock = threading.RLock()


def get_active_symbol() -> str:
    """
    Retorna o símbolo ativo atual.

    Pode diferir de SYMBOL quando o pipeline altera o símbolo
    depois do scan de tendência.
    """
    with _symbol_lock:
        return _active_symbol


def set_active_symbol(symbol: str) -> None:
    """
    Atualiza o símbolo ativo em runtime.

    Chamado pelo pipeline após a detecção de tendência.
    """
    global _active_symbol

    if symbol is None:
        return

    symbol = str(symbol).strip()

    if not symbol:
        return

    with _symbol_lock:
        _active_symbol = symbol


def is_market_open(what: str, utc_now=None) -> bool:
    """
    Verifica se o mercado associado a `what` está aberto no momento UTC.

    `what` pode ser:
        - um prefixo de símbolo como 'frx' para forex
        - o nome do mercado 'forex'
        - um símbolo completo (ex: 'frxEURUSD', 'R_100')

    Regras simplificadas:

        Forex:
            Domingo 22:00 UTC → Sexta 22:00 UTC

        Índices sintéticos:
            considerados 24/7

        Outros mercados:
            considerados abertos apenas de segunda a sexta.

    Nota:
        Esta função é deliberadamente simplificada. Não considera
        feriados específicos nem horários individuais de bolsas.
    """
    from datetime import datetime as _dt

    now = utc_now or _dt.utcnow()

    weekday = now.weekday()  # 0=segunda ... 6=domingo
    minutes = now.hour * 60 + now.minute

    key = str(what or "").strip()
    key_low = key.lower()

    # Alias "forex"
    if key_low == "forex":
        key = "frx"

    key_low = key.lower()

    # --------------------------------------------------------
    # Forex
    # Domingo 22:00 UTC → Sexta 22:00 UTC
    # --------------------------------------------------------
    if key_low.startswith("frx"):
        # Sábado: fechado
        if weekday == 5:
            return False

        # Domingo: abre às 22:00 UTC
        if weekday == 6:
            return minutes >= 22 * 60

        # Sexta: fecha às 22:00 UTC
        if weekday == 4:
            return minutes < 22 * 60

        # Segunda a quinta
        return True

    # --------------------------------------------------------
    # Índices sintéticos
    # --------------------------------------------------------
    synthetic_prefixes = (
        "r_",
        "1hz",
        "boom",
        "crash",
        "jd",
        "stp",
    )

    if any(
        key_low.startswith(prefix)
        for prefix in synthetic_prefixes
    ):
        return True

    # --------------------------------------------------------
    # Outros mercados
    # --------------------------------------------------------
    # Simplificação: apenas dias úteis.
    return weekday < 5


# ----- Duração -----
# Duração de fallback utilizada quando ainda não existe
# previsão do modelo de duração.
DURATION = 15
PRIORITIZE_DURATION = 15

DURATION_UNIT = "m"
BASIS = "stake"
CURRENCY = "USD"


# ----- Duração dinâmica (escolhida pela IA) -----
# Durações candidatas (em minutos) que o modelo de duração pode prever.
# Lista legada para leitura de artefatos; o contrato v2 usa DURATION fixo.
CANDIDATE_DURATIONS = [5, 15, 30]

DURATION_MODEL_PATH = "duration_model.pkl"


# ----- Parâmetros dos indicadores -----
EMA_FAST = 9
EMA_SLOW = 21

RSI_PERIOD = 14

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

ADX_PERIOD = 14

BB_PERIOD = 20
BB_STD = 2.0


# ----- Candles baseados em tempo -----
# Duração de cada vela em segundos.
#
# 60   = 1 minuto
# 300  = 5 minutos
# 900  = 15 minutos
#
# 900 = candles de 15 minutos, usados no treino e na inferência.
CANDLE_TIMEFRAME_SEC = 900
TRAINING_HISTORY_DAYS = 7
ASSET_HISTORY_CANDLES = 336
SHADOW_MIN_NET_PAYOUT = 0.70
SHADOW_MIN_ACCURACY = 0.55
SHADOW_MIN_RESULTS = 30


# ----- Price Action -----
CANDLE_SIZE = 10
CANDLE_LOOKBACK = 5

PA_SR_TOLERANCE = 0.001

ENTRY_MODE = "balanced"

OPPORTUNITY_SIGNAL_SCORE_MIN = 0.05
OPPORTUNITY_ADX_WEIGHT_FLOOR = 0.35

PRICE_ACTION_FILTER = _env_bool("STANDERIV_PRICE_ACTION_FILTER", False)

TARGET_NOISE_THRESHOLD = 0.0001

# Horizonte do target no dataset em candles.
# 1 = candle seguinte.
TARGET_LOOKFORWARD = DURATION * 60 // CANDLE_TIMEFRAME_SEC  # compatibilidade; target usa timestamps

CANDLE_NOTIFY = True


# ----- Filtro de padrões de vela -----
# True  = bloqueia entradas quando o padrão contradiz o sinal técnico
# False = padrões apenas geram alertas visuais
CANDLE_PATTERN_FILTER = _env_bool("STANDERIV_CANDLE_PATTERN_FILTER", False)

CANDLE_PATTERN_MIN_STRENGTH = 0.55

CANDLE_PATTERN_MAX_AGE_SEC = 300


# ----- Filtros de entrada -----
RSI_OVERSOLD = 38
RSI_OVERBOUGHT = 62

ADX_MIN = 30


# ----- ADX adaptativo -----
# True  → ADX_MIN ajustado ao percentil do histórico recente
# False → usa ADX_MIN fixo
ADX_ADAPTIVE = True

ADX_ADAPTIVE_PERCENTILE = 50


# ----- Gestão de risco -----
STAKE_PCT = 0.01
DERIV_MAX_PAYOUT = 100.00
DERIV_PAYOUT_CAP_SAFETY = 0.98
DERIV_PAYOUT_CAP_ESTIMATED_MULTIPLIER = 2.0

STOP_LOSS_PCT = 0.25
TAKE_PROFIT_PCT = 0.50

MAX_CONSEC_LOSSES = 3


# Pausa escalável:
# base * scale_factor^(losses_extras)
PAUSE_BASE_SEC = 600
PAUSE_SCALE_FACTOR = 2

RESUME_ON_WIN = True

ENFORCE_RISK_GATE = False


# ----- Aquecimento -----
MIN_TICKS = 50

MIN_CANDLES = 150


# ----- Buffer de preços -----
PRICE_BUFFER_SIZE = 500


# ----- Cadência de entradas -----
# Primeira entrada:
# imediatamente após MIN_CANDLES candles fechados.
#
# Entradas seguintes:
# somente após fechar ENTRY_CANDLE_INTERVAL nova(s) vela(s).
ENTRY_TICK_INTERVAL = 100

ENTRY_CANDLE_INTERVAL = 1

ENFORCE_ENTRY_CADENCE = False


# ----- Troca de ativo por falta de oportunidade -----
# Se > 0, troca o ativo após N avaliações fechadas sem condição de entrada.
# 0 desativa a troca automática para evitar reiniciar coleta/treino cedo demais.
UNFAVORABLE_SYMBOL_SWITCH_DECISIONS = 12

UNFAVORABLE_SYMBOL_SWITCH_REASONS = {
    "no_technical_signal",
    "ai_disagreement_or_low_confidence",
    "candle_conflict",
    "adx_falling",
    "missing_ai_confidence",
}

MARKET_SWITCH_COOLDOWN_SEC = 900
MARKET_SWITCH_MAX_PER_WINDOW = 4
MARKET_SWITCH_WINDOW_SEC = 3600


# ----- Timeout de operações -----
PROPOSAL_TIMEOUT_SEC = 10

PROPOSAL_API_ERROR_SWITCH_THRESHOLD = 3


# ----- Heartbeat / watchdog -----
HEARTBEAT_TIMEOUT_SEC = 30


# ----- Coletor — qualidade de dados -----
TICK_SPIKE_THRESHOLD = 0.05


# ----- Inteligência Artificial -----
# EXECUTION_SIGNAL_MODE mantém um único caminho operacional:
# "trained_model" → bundle aprovado libera execução e fornece probabilidade/payout.
EXECUTION_SIGNAL_MODE = os.environ.get(
    "STANDERIV_EXECUTION_SIGNAL_MODE",
    "trained_model",
).strip().lower()

VALID_EXECUTION_SIGNAL_MODES = {
    "trained_model",
}

if EXECUTION_SIGNAL_MODE not in VALID_EXECUTION_SIGNAL_MODES:
    print(
        "[CONFIG] Aviso: STANDERIV_EXECUTION_SIGNAL_MODE inválido; "
        "usando 'trained_model'."
    )
    EXECUTION_SIGNAL_MODE = "trained_model"

# USE_AI_MODEL = False → análise técnica offline; execução exige probabilidade calibrada
# USE_AI_MODEL = True  → usa bundle treinado aprovado na decisão
USE_AI_MODEL = _env_bool("STANDERIV_USE_AI_MODEL", True)

AI_MODEL_PATH = "model.pkl"

AI_CONFIDENCE_MIN = MIN_APPROVAL_WIN_RATE


# ----- Ponderação IA vs técnico -----
AI_TECH_WEIGHT = 0.55
AI_MODEL_WEIGHT = 0.45

AI_SCORE_MIN = 0.30


# ----- Sinal ponderado -----
# False = comportamento original, AND rígido
# True  = score ponderado por indicador
USE_WEIGHTED_SIGNAL = _env_bool("STANDERIV_USE_WEIGHTED_SIGNAL", True)

SIGNAL_SCORE_MIN = 0.06


# ----- Detecção de drift -----
DRIFT_WINDOW = 20

DRIFT_WIN_RATE_MIN = 0.40


# ----- Promoção de modelo (champion-challenger) -----
MODEL_PROMOTION_MIN_AUC_DELTA = 0.001

MODEL_PROMOTION_MAX_ACC_DROP = 0.02

MODEL_PROMOTION_MAX_F1_DROP = 0.05


# ----- Retreino adaptativo -----
# Janela de trades recentes analisados para detecção de degradação.
RETRAIN_EVAL_WINDOW = 20

# Aciona retreino se win rate recente cair abaixo deste valor.
RETRAIN_WIN_RATE_TRIGGER = 0.42

# Aciona retreino se confiança média da IA cair abaixo deste valor.
RETRAIN_CONFIDENCE_TRIGGER = 0.55

# Aciona retreino após acumular N novos ticks desde o último treino.
RETRAIN_NEW_TICKS_TRIGGER = 400

# Tempo mínimo entre dois retreinos.
RETRAIN_MIN_INTERVAL_SEC = 300

# Backstop: força retreino mesmo sem gatilho.
RETRAIN_MAX_INTERVAL_SEC = 3600


# ----- Arquivos de log -----
LEGACY_TICKS_CSV = "ticks.csv"
TICKS_CSV = "ticks_v2.csv"

OPERATIONS_LOG = "operacoes_v2.csv"  # IDs e preços oficiais; histórico legado preservado

DATASET_CSV = "dataset_v2.csv"


# ============================================================
# Temporal Fusion Transformer (TFT)
# ============================================================

# Configuração legada: TFT não integra o bundle v2 de contratos.
# Reativação exige calibração e avaliação conjunta; não basta alterar esta flag.
USE_TRANSFORMER = False  # bundles antigos não possuem contrato temporal validado

TRANSFORMER_MODEL_PATH = "transformer_model.pkl"


# Blend do ensemble:
#
# conf_final =
#     TFT * BLEND
#     +
#     classical * (1 - BLEND)
#
TRANSFORMER_BLEND_WEIGHT = 0.55


# Número de vetores de features passados ao TFT.
#
# PRICE_BUFFER_SIZE deve ser suficientemente maior que
# TRANSFORMER_SEQ_LEN.
TRANSFORMER_SEQ_LEN = 50


# ----- Arquitetura interna do TFT -----
TRANSFORMER_D_MODEL = 64

TRANSFORMER_N_HEADS = 4

TRANSFORMER_N_LAYERS = 2

TRANSFORMER_DROPOUT = 0.15


# ----- Treinamento -----
TRANSFORMER_EPOCHS = 20

TRANSFORMER_BATCH_SIZE = 128

TRANSFORMER_LR = 3e-4

TRANSFORMER_PATIENCE = 10

# Retreino exige informação nova, não apenas mais ticks do mesmo candle.
RETRAIN_NEW_LABELS_TRIGGER = 120


# ----- Bootstrap sem modelo aprovado -----
# Evita que o pipeline fique indefinidamente preso em um único ativo quando
# sucessivos treinos não geram bundle elegível para execução.
BOOTSTRAP_MAX_FAILED_PROMOTIONS = 3
