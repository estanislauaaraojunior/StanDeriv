# ============================================================
#  config.py — configurações centralizadas do bot Deriv
#  Edite APENAS este arquivo para ajustar o comportamento.
# ============================================================

import os
import threading

from dotenv import load_dotenv


load_dotenv()  # carrega variáveis de .env (ignorado se não existir)


# ----- Conta -----
# True  → opera na conta DEMO (seguro — dinheiro virtual)
# False → opera na conta REAL (requer TOKEN de conta real + confirmação no terminal)
DEMO_MODE = True


# ----- Conexão -----
_RAW_APP_ID = os.environ.get("DERIV_APP_ID", "1089").strip()
_RAW_TOKEN = os.environ["DERIV_TOKEN"].strip()

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
DURATION = 5
PRIORITIZE_DURATION = 15

DURATION_UNIT = "m"
BASIS = "stake"
CURRENCY = "USD"


# ----- Duração dinâmica (escolhida pela IA) -----
# Durações candidatas (em minutos) que o modelo de duração pode prever.
# A IA aprende qual duração é mais adequada para cada condição de mercado.
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
# 60 = mais candles e avaliações mais frequentes.
CANDLE_TIMEFRAME_SEC = 60


# ----- Price Action -----
CANDLE_SIZE = 10
CANDLE_LOOKBACK = 5

PA_SR_TOLERANCE = 0.001

TARGET_NOISE_THRESHOLD = 0.0001

# Horizonte do target no dataset em candles.
# 1 = candle seguinte.
TARGET_LOOKFORWARD = 2

CANDLE_NOTIFY = True


# ----- Filtro de padrões de vela -----
# True  = bloqueia entradas quando o padrão contradiz o sinal técnico
# False = padrões apenas geram alertas visuais
CANDLE_PATTERN_FILTER = True

CANDLE_PATTERN_MIN_STRENGTH = 0.55

CANDLE_PATTERN_MAX_AGE_SEC = 300


# ----- Filtros de entrada -----
RSI_OVERSOLD = 38
RSI_OVERBOUGHT = 62

ADX_MIN = 18


# ----- ADX adaptativo -----
# True  → ADX_MIN ajustado ao percentil do histórico recente
# False → usa ADX_MIN fixo
ADX_ADAPTIVE = True

ADX_ADAPTIVE_PERCENTILE = 50


# ----- Gestão de risco -----
STAKE_PCT = 0.01

STOP_LOSS_PCT = 0.25
TAKE_PROFIT_PCT = 0.50

MAX_CONSEC_LOSSES = 3


# Pausa escalável:
# base * scale_factor^(losses_extras)
PAUSE_BASE_SEC = 600
PAUSE_SCALE_FACTOR = 2

RESUME_ON_WIN = True


# ----- Aquecimento -----
MIN_TICKS = 50

MIN_CANDLES = 50


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


# ----- Timeout de operações -----
PROPOSAL_TIMEOUT_SEC = 10


# ----- Heartbeat / watchdog -----
HEARTBEAT_TIMEOUT_SEC = 30


# ----- Coletor — qualidade de dados -----
TICK_SPIKE_THRESHOLD = 0.05


# ----- Inteligência Artificial -----
# USE_AI_MODEL = False → funcionamento apenas com indicadores
# USE_AI_MODEL = True  → IA pondera o sinal técnico
USE_AI_MODEL = True

AI_MODEL_PATH = "model.pkl"

AI_CONFIDENCE_MIN = 0.55


# ----- Ponderação IA vs técnico -----
AI_TECH_WEIGHT = 0.55
AI_MODEL_WEIGHT = 0.45

AI_SCORE_MIN = 0.30


# ----- Sinal ponderado -----
# False = comportamento original, AND rígido
# True  = score ponderado por indicador
USE_WEIGHTED_SIGNAL = True

SIGNAL_SCORE_MIN = 0.05


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
TICKS_CSV = "ticks.csv"

OPERATIONS_LOG = "operacoes_log.csv"

DATASET_CSV = "dataset.csv"


# ============================================================
# Temporal Fusion Transformer (TFT)
# ============================================================

# True  → treina e usa TFT em ensemble com RF/XGB
# False → funcionamento sem Transformer
USE_TRANSFORMER = True

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