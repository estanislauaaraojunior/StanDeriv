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
    CANDLE_SIZE, PA_SR_TOLERANCE,
    CANDLE_PATTERN_FILTER, CANDLE_PATTERN_MIN_STRENGTH, CANDLE_PATTERN_MAX_AGE_SEC,
)
import time
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

def get_signal(prices: list, adx_min: float = ADX_MIN, adx_history: list = None, candle_alerts: list = None) -> SignalResult:
    """
    Avalia o estado do mercado e retorna ("BUY" | "SELL" | None, indicadores).

    Args:
        prices:      lista de floats com histórico de preços
        adx_min:     limiar mínimo de ADX (P10: pode ser adaptativo)
        adx_history: histórico recente de ADX para filtro de tendência crescente

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

    # ── Filtro ADX rising — tendência perdendo força (limiar suavizado para índices sintéticos) ──
    if adx_history and len(adx_history) >= 10:
        adx_avg_recent = sum(adx_history[-10:]) / 10
        if adx_val < adx_avg_recent * 0.75:
            indicators["adx_falling"] = True
            return None, indicators

    # ── Filtro 2: RSI — bloquear sinais contra extremos ─────
    # RSI < 30 → bloquear BUY (sobrevendido demais para comprar)
    # RSI > 70 → bloquear SELL (sobrecomprado demais para vender)
    # Permite BUY com RSI > 65 (momentum forte) e SELL com RSI < 35

    # ── P14: Score ponderado (opcional) ────────────────────
    if USE_WEIGHTED_SIGNAL:
        return _weighted_signal(ema9, ema21, macd_hist, mom, adx_val, rsi_val, last_price, indicators, prices, candle_alerts)

    # ── Sinal de COMPRA (AND rígido) ─────────────────────────────────────────
    if (
        ema9 > ema21           # tendência de alta confirmada
        and last_price > ema9  # preço acima da média rápida
        and macd_hist > 0      # momentum positivo
        and mom > 0            # últimos ticks subindo
        and rsi_val > RSI_OVERSOLD   # não sobrevendido contra
    ):
        return _apply_ai_filter("BUY", prices, indicators, candle_alerts)

    # ── Sinal de VENDA (AND rígido) ──────────────────────────────────────────
    if (
        ema9 < ema21           # tendência de baixa confirmada
        and last_price < ema9  # preço abaixo da média rápida
        and macd_hist < 0      # momentum negativo
        and mom < 0            # últimos ticks caindo
        and rsi_val < RSI_OVERBOUGHT  # não sobrecomprado contra
    ):
        return _apply_ai_filter("SELL", prices, indicators, candle_alerts)

    # Sem sinal válido (ex: cruzamento recente, aguardar confirmação)
    return None, indicators


# ─────────────────────────────────────────────────────────────────
#  P14 — Score ponderado por indicador
# ─────────────────────────────────────────────────────────────────

def _weighted_signal(
    ema9: float, ema21: float, macd_hist: float, mom: float,
    adx_val: float, rsi_val: float, price: float, indicators: dict, prices: list,
    candle_alerts: list = None,
) -> SignalResult:
    """
    Gera sinal via score contínuo ponderado (USE_WEIGHTED_SIGNAL=True).

    Scores contínuos (-1 a +1) ao invés de binários:
      EMA cross 30% | MACD 25% | preço vs EMA 20% | momentum 15% | ADX 10%
    """
    eps = 1e-12

    # Score EMA: distância relativa normalizada
    ema_score = max(-1.0, min(1.0, (ema9 - ema21) / (ema21 * 0.001 + eps)))
    # Score preço vs EMA9
    price_score = max(-1.0, min(1.0, (price - ema9) / (ema9 * 0.001 + eps)))
    # Score MACD histograma
    macd_score = max(-1.0, min(1.0, macd_hist / (abs(macd_hist) + abs(ema21 * 0.0001) + eps)))
    # Score momentum
    mom_score = max(-1.0, min(1.0, mom / (price * 0.0005 + eps)))
    # Score RSI contínuo
    rsi_score = (rsi_val - 50.0) / 50.0
    # ADX normalizado (0-1)
    adx_norm = min(adx_val / 40.0, 1.0)

    # Score composto (positivo = bullish, negativo = bearish)
    composite = (
        ema_score * 0.25
        + macd_score * 0.25
        + price_score * 0.15
        + mom_score * 0.15
        + rsi_score * 0.10
    ) * adx_norm  # ADX modula a confiança

    indicators["tech_score"] = round(composite, 4)

    if composite >= SIGNAL_SCORE_MIN:
        return _apply_ai_filter("BUY", prices, indicators, candle_alerts)
    if composite <= -SIGNAL_SCORE_MIN:
        return _apply_ai_filter("SELL", prices, indicators, candle_alerts)
    return None, indicators


# ─────────────────────────────────────────────────────────────────
#  Filtro de padrões de vela (soft conflict filter)
# ─────────────────────────────────────────────────────────────────

def _candle_pattern_filter(signal: str, candle_alerts: list) -> str | None:
    """
    Verifica se o padrão de vela mais recente contradiz o sinal técnico.

    Retorna motivo de bloqueio (str) se conflito direto detectado, ou None.
    Lógica suave: só bloqueia conflito direto — BUY+bearish ou SELL+bullish.
    Padrões neutros e padrões alinhados com o sinal não bloqueiam.
    """
    now = time.time()
    for alert in reversed(list(candle_alerts)):
        if now - alert.get("timestamp", 0) > CANDLE_PATTERN_MAX_AGE_SEC:
            continue  # padrão expirado
        if alert.get("strength", 0.0) < CANDLE_PATTERN_MIN_STRENGTH:
            continue  # padrão fraco demais
        direction = alert.get("direction", "neutral")
        if direction == "neutral":
            continue  # neutro não bloqueia
        if signal == "BUY" and direction == "bearish":
            return f"candle_conflict:{alert['name']}(bearish,str={alert['strength']:.2f})"
        if signal == "SELL" and direction == "bullish":
            return f"candle_conflict:{alert['name']}(bullish,str={alert['strength']:.2f})"
        break  # padrão alinhado com o sinal — não bloqueia
    return None


# ─────────────────────────────────────────────────────────────────
#  P4 — Filtro de IA como peso ponderado (não gate duro)
# ─────────────────────────────────────────────────────────────────

def _apply_ai_filter(
    signal: str,
    prices: list,
    indicators: dict,
    candle_alerts: list = None,
) -> SignalResult:
    """
    Pondera o sinal dos indicadores técnicos com o modelo de IA e filtro PA.
    """
    # -- Filtro de padrões de vela: bloqueia conflito direto padrão vs. sinal --
    if CANDLE_PATTERN_FILTER and candle_alerts:
        block_reason = _candle_pattern_filter(signal, candle_alerts)
        if block_reason:
            indicators["candle_pattern_block"] = block_reason
            print(f"[VELA] ⛔ Sinal {signal} bloqueado: {block_reason}")
            return None, indicators

    # -- Filtro Price Action: bloquear sinais contra estrutura de mercado --
    candles = ind.ticks_to_candles(prices, CANDLE_SIZE)
    pa = ind.price_action_features(candles, PA_SR_TOLERANCE)
    if pa is not None:
        indicators["pa_sr_position"] = round(pa["pa_sr_position"], 4)
        indicators["pa_market_structure"] = round(pa["pa_market_structure"], 4)

        # Bloquear BUY perto de resistência forte sem break of structure
        if signal == "BUY" and pa["pa_sr_position"] > 0.8 and pa["pa_bos_strength"] <= 0:
            return None, indicators
        # Bloquear SELL perto de suporte forte sem break of structure
        if signal == "SELL" and pa["pa_sr_position"] < -0.8 and pa["pa_bos_strength"] >= 0:
            return None, indicators

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
