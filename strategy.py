"""
strategy.py — motor de decisão.

Recebe histórico de preços (ticks) e retorna o sinal de entrada
junto com os valores dos indicadores calculados.

Condições de entrada (modo AND rígido — USE_WEIGHTED_SIGNAL=False):

  BUY  → EMA9 > EMA21  AND  preço > EMA9  AND  RSI [35–65]
          AND  ADX > adx_min  AND  MACD_hist > 0  AND  momentum > 0
          [SE USE_AI_MODEL] IA pondera o sinal via score (P4)

  SELL → EMA9 < EMA21  AND  preço < EMA9  AND  RSI [35–65]
          AND  ADX > adx_min  AND  MACD_hist < 0  AND  momentum < 0
          [SE USE_AI_MODEL] IA pondera o sinal via score (P4)

  None → qualquer filtro falhou (mercado lateral, indefinido ou score insuficiente)

Modo alternativo (USE_WEIGHTED_SIGNAL=True — P14):
  Score ponderado por indicador; mais entradas, menos seletivo.
"""

from typing import Optional, Tuple
import indicators as ind
from config import (
    EMA_FAST, EMA_SLOW,
    RSI_PERIOD, RSI_OVERSOLD, RSI_OVERBOUGHT,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    ADX_PERIOD, ADX_MIN,
    BB_PERIOD, BB_STD,
    USE_AI_MODEL,
    USE_WEIGHTED_SIGNAL, SIGNAL_SCORE_MIN,
    AI_TECH_WEIGHT, AI_MODEL_WEIGHT, AI_SCORE_MIN,
)
import ai_predictor

# Tipo de retorno: (sinal, dict com valores dos indicadores)
SignalResult = Tuple[Optional[str], dict]


# ─────────────────────────────────────────────────────────────────
#  P10 — ADX mínimo adaptativo
# ─────────────────────────────────────────────────────────────────

def get_adaptive_adx_min(adx_history: list) -> float:
    """
    Retorna o ADX mínimo adaptado ao percentil do histórico recente.

    Se ADX_ADAPTIVE=False ou histórico vazio, retorna ADX_MIN fixo.
    Piso de 15 para evitar filtros excessivamente frouxos.
    """
    from config import ADX_ADAPTIVE, ADX_ADAPTIVE_PERCENTILE
    if not adx_history or not ADX_ADAPTIVE:
        return ADX_MIN
    sorted_h = sorted(adx_history)
    idx = max(0, int(len(sorted_h) * ADX_ADAPTIVE_PERCENTILE / 100) - 1)
    return max(sorted_h[idx], 15.0)


# ─────────────────────────────────────────────────────────────────
#  Interface pública
# ─────────────────────────────────────────────────────────────────

def get_signal(prices: list, adx_min: float = ADX_MIN) -> SignalResult:
    """
    Avalia o estado do mercado e retorna ("BUY" | "SELL" | None, indicadores).

    Args:
        prices:  lista de floats com histórico de preços
        adx_min: limiar mínimo de ADX (P10: pode ser adaptativo)

    O dict `indicadores` está sempre populado quando há dados suficientes,
    mesmo quando o sinal é None — útil para exibição e logging.
    """
    # ── Calcular todos os indicadores ──────────────────────
    ema9      = ind.ema(prices, EMA_FAST)
    ema21     = ind.ema(prices, EMA_SLOW)
    rsi_val   = ind.rsi(prices, RSI_PERIOD)
    macd_res  = ind.macd(prices, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    adx_val   = ind.adx(prices, ADX_PERIOD)
    bb_res    = ind.bollinger(prices, BB_PERIOD, BB_STD)
    mom       = ind.momentum(prices, 3)

    # Dados insuficientes para qualquer cálculo
    if any(v is None for v in [ema9, ema21, rsi_val, macd_res, adx_val, mom]):
        return None, {}

    macd_line, macd_sig, macd_hist = macd_res
    last_price = prices[-1]

    indicators: dict = {
        "ema9":      round(ema9, 5),
        "ema21":     round(ema21, 5),
        "rsi":       round(rsi_val, 2),
        "adx":       round(adx_val, 2),
        "macd_line": round(macd_line, 6),
        "macd_hist": round(macd_hist, 6),
        "momentum":  round(mom, 6),
    }

    if bb_res is not None:
        bb_upper, bb_mid, bb_lower = bb_res
        indicators.update({
            "bb_upper":  round(bb_upper, 5),
            "bb_mid":    round(bb_mid, 5),
            "bb_lower":  round(bb_lower, 5),
        })

    # ── Filtro 1: ADX — bloquear mercado lateral (P10: adx_min adaptativo) ──
    if adx_val < adx_min:
        return None, indicators

    # ── Filtro 2: RSI — evitar extremos ────────────────────
    if not (RSI_OVERSOLD <= rsi_val <= RSI_OVERBOUGHT):
        return None, indicators

    # ── P14: Score ponderado (opcional) ────────────────────
    if USE_WEIGHTED_SIGNAL:
        return _weighted_signal(ema9, ema21, macd_hist, mom, adx_val, last_price, indicators, prices)

    # ── Sinal de COMPRA (AND rígido) ─────────────────────────────────────────
    if (
        ema9 > ema21           # tendência de alta confirmada
        and last_price > ema9  # preço acima da média rápida
        and macd_hist > 0      # momentum positivo
        and mom > 0            # últimos ticks subindo
    ):
        return _apply_ai_filter("BUY", prices, indicators)

    # ── Sinal de VENDA (AND rígido) ──────────────────────────────────────────
    if (
        ema9 < ema21           # tendência de baixa confirmada
        and last_price < ema9  # preço abaixo da média rápida
        and macd_hist < 0      # momentum negativo
        and mom < 0            # últimos ticks caindo
    ):
        return _apply_ai_filter("SELL", prices, indicators)

    # Sem sinal válido (ex: cruzamento recente, aguardar confirmação)
    return None, indicators


# ─────────────────────────────────────────────────────────────────
#  P14 — Score ponderado por indicador
# ─────────────────────────────────────────────────────────────────

def _weighted_signal(
    ema9: float, ema21: float, macd_hist: float, mom: float,
    adx_val: float, price: float, indicators: dict, prices: list,
) -> SignalResult:
    """
    Gera sinal via score ponderado (USE_WEIGHTED_SIGNAL=True).

    Pesos: EMA cross 30% | preço vs EMA 20% | MACD 25% | momentum 15% | ADX 10%
    """
    ema_up    = 1.0 if ema9 > ema21       else 0.0
    price_up  = 1.0 if price > ema9       else 0.0
    macd_up   = 1.0 if macd_hist > 0      else 0.0
    mom_up    = 1.0 if mom > 0            else 0.0
    adx_norm  = min(adx_val / 40.0, 1.0)  # normalizar ADX em [0, 1]

    buy_score  = ema_up * 0.30 + price_up * 0.20 + macd_up * 0.25 + mom_up * 0.15 + adx_norm * 0.10
    sell_score = (1.0 - ema_up) * 0.30 + (1.0 - price_up) * 0.20 + \
                 (1.0 - macd_up) * 0.25 + (1.0 - mom_up) * 0.15 + adx_norm * 0.10

    indicators["tech_score"] = round(max(buy_score, sell_score), 4)

    if buy_score >= SIGNAL_SCORE_MIN:
        return _apply_ai_filter("BUY", prices, indicators)
    if sell_score >= SIGNAL_SCORE_MIN:
        return _apply_ai_filter("SELL", prices, indicators)
    return None, indicators


# ─────────────────────────────────────────────────────────────────
#  P4 — Filtro de IA como peso ponderado (não gate duro)
# ─────────────────────────────────────────────────────────────────

def _apply_ai_filter(
    signal: str,
    prices: list,
    indicators: dict,
) -> SignalResult:
    """
    Pondera o sinal dos indicadores técnicos com o modelo de IA.

    Se USE_AI_MODEL=False, passa o sinal sem alteração.
    Se USE_AI_MODEL=True:
      - IA concorda → score = AI_TECH_WEIGHT + AI_MODEL_WEIGHT * ai_conf
      - IA diverge  → score = AI_TECH_WEIGHT - AI_MODEL_WEIGHT * (1 - ai_conf)
      - score < AI_SCORE_MIN → sinal bloqueado

    Substitui o gate duro (exigia concordância exata) por penalidade proporcional,
    permitindo operar quando a IA diverge com baixa confiança.
    """
    if not USE_AI_MODEL:
        indicators["ai_confidence"] = None
        indicators["ai_score"]      = None
        return signal, indicators

    ai_direction, ai_conf = ai_predictor.predict(prices)
    indicators["ai_confidence"] = round(ai_conf, 4)

    if ai_direction == signal:
        # IA concorda: score pleno
        score = AI_TECH_WEIGHT * 1.0 + AI_MODEL_WEIGHT * ai_conf
    else:
        # IA diverge: penalidade proporcional à confiança da divergência
        score = AI_TECH_WEIGHT * 1.0 - AI_MODEL_WEIGHT * (1.0 - ai_conf)

    indicators["ai_score"] = round(score, 4)

    if score >= AI_SCORE_MIN:
        return signal, indicators

    return None, indicators
