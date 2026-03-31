"""
indicators.py — funções puras de análise técnica.

Todas as funções:
  - Recebem uma lista simples de floats (preços de fechamento / ticks)
  - Retornam um valor ou None quando dados insuficientes
  - Sem estado interno — seguras para backtesting e uso em modelos de IA
"""

import math
from typing import Optional, Tuple


# ─────────────────────────────────────────────────────────────
#  Helpers internos
# ─────────────────────────────────────────────────────────────

def _ema(data: list, period: int) -> float:
    """EMA sem verificação de tamanho — use apenas internamente."""
    k = 2.0 / (period + 1)
    val = data[0]
    for price in data:
        val = price * k + val * (1.0 - k)
    return val


# ─────────────────────────────────────────────────────────────
#  Indicadores públicos
# ─────────────────────────────────────────────────────────────

def ema(prices: list, period: int) -> Optional[float]:
    """
    Exponential Moving Average.

    EMA rápida (9) vs lenta (21) indica direção da tendência.
    Requer pelo menos `period` pontos.
    """
    if len(prices) < period:
        return None
    return _ema(prices, period)


def rsi(prices: list, period: int = 14) -> Optional[float]:
    """
    Relative Strength Index (Wilder's smoothing).

    Retorna valor em [0, 100].
      < 30 → sobrevendido
      > 70 → sobrecomprado
    Filtro usado: aceitar apenas RSI entre 35 e 65 (zona neutra).
    """
    if len(prices) < period + 1:
        return None

    gains, losses = [], []
    for i in range(1, len(prices)):
        delta = prices[i] - prices[i - 1]
        if delta >= 0:
            gains.append(delta)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(-delta)

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0.0:
        return 100.0

    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def macd(
    prices: list,
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
) -> Optional[Tuple[float, float, float]]:
    """
    MACD = EMA(fast) − EMA(slow).
    Signal = EMA(MACD_history, signal_period).
    Histogram = MACD − Signal.

    Retorna (macd_line, signal_line, histogram) ou None.
    Histograma positivo → momentum de alta; negativo → baixa.
    """
    min_len = slow + signal_period
    if len(prices) < min_len:
        return None

    # Constrói histórico do MACD line a partir de fatias crescentes
    macd_history: list = []
    for i in range(slow, len(prices) + 1):
        chunk = prices[:i]
        macd_history.append(_ema(chunk, fast) - _ema(chunk, slow))

    if len(macd_history) < signal_period:
        return None

    current_macd = macd_history[-1]
    sig = _ema(macd_history, signal_period)
    histogram = current_macd - sig
    return current_macd, sig, histogram


def adx(prices: list, period: int = 14) -> Optional[float]:
    """
    Average Directional Index — mede a FORÇA da tendência (não a direção).

    ADX > 20 → tendência presente → sinal válido
    ADX < 20 → mercado lateral → sem entrada

    Aproximação para dados de tick:
      high[i] ≈ max(prices[i-1], prices[i])
      low[i]  ≈ min(prices[i-1], prices[i])
    """
    min_len = period * 2 + 1
    if len(prices) < min_len:
        return None

    plus_dm_list: list  = []
    minus_dm_list: list = []
    tr_list: list       = []

    for i in range(1, len(prices)):
        hi_cur  = max(prices[i - 1], prices[i])
        lo_cur  = min(prices[i - 1], prices[i])
        hi_prev = max(prices[max(0, i - 2): i])
        lo_prev = min(prices[max(0, i - 2): i])

        up_move   = hi_cur - hi_prev
        down_move = lo_prev - lo_cur

        plus_dm  = up_move   if up_move > down_move   and up_move > 0   else 0.0
        minus_dm = down_move if down_move > up_move   and down_move > 0 else 0.0

        tr = max(
            hi_cur - lo_cur,
            abs(hi_cur  - prices[i - 1]),
            abs(lo_cur  - prices[i - 1]),
        )

        plus_dm_list.append(plus_dm)
        minus_dm_list.append(minus_dm)
        tr_list.append(tr)

    if len(tr_list) < period:
        return None

    # Wilder's smoothing inicial
    s_tr    = sum(tr_list[:period])
    s_plus  = sum(plus_dm_list[:period])
    s_minus = sum(minus_dm_list[:period])

    dx_list: list = []
    for i in range(period, len(tr_list)):
        s_tr    = s_tr    - s_tr / period    + tr_list[i]
        s_plus  = s_plus  - s_plus / period  + plus_dm_list[i]
        s_minus = s_minus - s_minus / period + minus_dm_list[i]

        if s_tr == 0.0:
            dx_list.append(0.0)
            continue

        plus_di  = 100.0 * s_plus  / s_tr
        minus_di = 100.0 * s_minus / s_tr
        di_sum   = plus_di + minus_di

        dx_list.append(100.0 * abs(plus_di - minus_di) / di_sum if di_sum else 0.0)

    if not dx_list:
        return None

    # ADX = Wilder smoothed DX
    adx_val = sum(dx_list[:period]) / min(period, len(dx_list))
    for dx_val in dx_list[period:]:
        adx_val = (adx_val * (period - 1) + dx_val) / period

    return adx_val


def bollinger(
    prices: list,
    period: int = 20,
    std_dev: float = 2.0,
) -> Optional[Tuple[float, float, float]]:
    """
    Bollinger Bands.
    Retorna (upper, middle, lower) ou None.

    Preço perto da banda superior → sobrecomprado.
    Preço perto da banda inferior → sobrevendido.
    Bandas estreitas → baixa volatilidade → mercado lateral.
    """
    if len(prices) < period:
        return None

    window   = prices[-period:]
    mid      = sum(window) / period
    variance = sum((p - mid) ** 2 for p in window) / period
    std      = math.sqrt(variance)

    return mid + std_dev * std, mid, mid - std_dev * std


def momentum(prices: list, period: int = 3) -> Optional[float]:
    """
    Momentum simples: preço atual − preço N períodos atrás.
    Positivo → ticks subindo; negativo → ticks caindo.
    Usado como confirmação rápida de direção.
    """
    if len(prices) < period + 1:
        return None
    return prices[-1] - prices[-(period + 1)]
