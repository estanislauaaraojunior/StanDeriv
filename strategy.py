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
import config
from config import (
    EMA_FAST, EMA_SLOW,
    RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    ADX_PERIOD, ADX_MIN,
    BB_PERIOD, BB_STD,
    USE_AI_MODEL, AI_CONFIDENCE_MIN,
    USE_WEIGHTED_SIGNAL, SIGNAL_SCORE_MIN,
    AI_TECH_WEIGHT, AI_MODEL_WEIGHT, AI_SCORE_MIN,
    CANDLE_SIZE, PA_SR_TOLERANCE, PRICE_ACTION_FILTER,
    CANDLE_PATTERN_FILTER, CANDLE_PATTERN_MIN_STRENGTH, CANDLE_PATTERN_MAX_AGE_SEC,
)
import time
import math
from model_contract import FEATURE_WINDOW
import ai_predictor

# Tipo de retorno: (sinal, dict com valores dos indicadores)
SignalResult = Tuple[Optional[str], dict]


# ─────────────────────────────────────────────────────────────────
#  Modo de entrada
# ─────────────────────────────────────────────────────────────────

def _entry_mode() -> str:
    """Modo de entrada configurado (`ENTRY_MODE`), normalizado; padrão 'balanced'."""
    return str(getattr(config, "ENTRY_MODE", "balanced") or "balanced").strip().lower()

def _opportunity_mode() -> bool:
    """Indica se o modo de entrada é 'opportunity'."""
    return _entry_mode() == "opportunity"


def _execution_signal_mode() -> str:
    """Modo do sinal de execução configurado (`EXECUTION_SIGNAL_MODE`), validado contra os modos permitidos."""
    mode = str(
        getattr(config, "EXECUTION_SIGNAL_MODE", "trained_model") or "trained_model"
    ).strip().lower()
    valid = getattr(
        config,
        "VALID_EXECUTION_SIGNAL_MODES",
        {"trained_model"},
    )
    return mode if mode in valid else "trained_model"


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

def get_signal(
    prices: list,
    adx_min: float = ADX_MIN,
    adx_history: list = None,
    candle_alerts: list = None,
    use_ai_model: Optional[bool] = None,
    evaluation_time: Optional[float] = None,
    candles: list = None,
) -> SignalResult:
    """
    Avalia o estado do mercado e retorna ("BUY" | "SELL" | None, indicadores).

    Args:
        prices:      lista de floats com histórico de preços
        adx_min:     limiar mínimo de ADX (P10: pode ser adaptativo)
        adx_history: histórico recente de ADX para filtro de tendência crescente
        use_ai_model: sobrescreve USE_AI_MODEL quando informado. O backtester usa
                      False por padrão para evitar avaliar um modelo nos mesmos
                      dados usados em seu treinamento.
        evaluation_time: timestamp de referência para filtros temporais. Quando
                         omitido, usa o relógio atual (comportamento ao vivo).

    O dict `indicadores` está sempre populado quando há dados suficientes,
    mesmo quando o sinal é None — útil para exibição e logging.
    """
    prices = list(prices[-FEATURE_WINDOW:])
    if not prices or any(not math.isfinite(float(p)) or p <= 0 for p in prices):
        return None, {}
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

    weighted_enabled = bool(USE_WEIGHTED_SIGNAL)
    model_enabled = bool(USE_AI_MODEL if use_ai_model is None else use_ai_model)
    indicators["enabled_decision_modules"] = [
        name for enabled, name in (
            (weighted_enabled, "weighted_score"),
            (model_enabled, "trained_model"),
            (PRICE_ACTION_FILTER, "price_action"),
            (CANDLE_PATTERN_FILTER, "candle_pattern"),
        ) if enabled
    ]
    adx_is_falling = False
    if adx_history and len(adx_history) >= 10:
        adx_avg_recent = sum(adx_history[-10:]) / 10
        adx_is_falling = adx_val < adx_avg_recent * 0.75
        if adx_is_falling:
            indicators["adx_falling"] = True
    if not indicators["enabled_decision_modules"]:
        indicators["decision_block"] = "no_decision_modules_enabled"
        return None, indicators

    weighted_signal = None
    weighted_indicators = {}
    if weighted_enabled:
        if adx_val < adx_min:
            indicators["adx_below_min"] = True
            if not _opportunity_mode():
                weighted_indicators["decision_block"] = "adx_below_min"
        if adx_is_falling and not _opportunity_mode():
            weighted_indicators["decision_block"] = "adx_falling"
        if not weighted_indicators:
            weighted_signal, weighted_indicators = _weighted_signal(
                ema9, ema21, macd_hist, mom, adx_val, rsi_val, last_price,
                dict(indicators), prices, candle_alerts, use_ai_model=False,
                evaluation_time=evaluation_time, candles=candles,
                apply_aux_filters=False,
            )

    model_signal = None
    model_indicators = {}
    if model_enabled:
        model_signal, model_indicators = _trained_model_signal(prices, candles=candles)

    primary_count = int(weighted_enabled) + int(model_enabled)
    if primary_count == 2:
        if weighted_signal not in ("BUY", "SELL") or model_signal not in ("BUY", "SELL"):
            indicators.update(weighted_indicators)
            indicators.update(model_indicators)
            indicators["decision_block"] = "primary_module_abstention"
            return None, indicators
        if weighted_signal != model_signal:
            indicators.update(weighted_indicators)
            indicators.update(model_indicators)
            indicators["decision_block"] = "primary_module_disagreement"
            return None, indicators
        indicators.update(weighted_indicators)
        indicators.update(model_indicators)
        indicators["decision_modules"] = ["weighted_score", "trained_model"]
        indicators["decision_source"] = "weighted_score+trained_model"
        return weighted_signal, indicators

    if primary_count == 1:
        if weighted_enabled:
            primary_signal, primary_indicators = weighted_signal, weighted_indicators
            primary_module = "weighted_score"
        else:
            primary_signal, primary_indicators = model_signal, model_indicators
            primary_module = "trained_model"
        if primary_signal in ("BUY", "SELL"):
            indicators.update(primary_indicators)
            indicators["decision_modules"] = [primary_module]
            indicators["decision_source"] = primary_module
            return primary_signal, indicators

    decision_candles = list(candles[-FEATURE_WINDOW:]) if candles is not None else ind.ticks_to_candles(prices, CANDLE_SIZE)
    pa = ind.price_action_features(decision_candles, PA_SR_TOLERANCE)
    if pa is not None:
        indicators["pa_sr_position"] = round(pa["pa_sr_position"], 4)
        indicators["pa_market_structure"] = round(pa["pa_market_structure"], 4)

    fallback_signals = []
    if PRICE_ACTION_FILTER:
        pa_signal, pa_score = _price_action_signal(pa)
        indicators["pa_score"] = round(pa_score, 4)
        if pa_signal:
            fallback_signals.append(("price_action", pa_signal))
    if CANDLE_PATTERN_FILTER:
        pattern_signal, pattern_reason = _candle_pattern_signal(candle_alerts, evaluation_time)
        if pattern_reason:
            indicators["candle_pattern_block"] = pattern_reason
        if pattern_signal:
            fallback_signals.append(("candle_pattern", pattern_signal))

    if len(fallback_signals) == 2 and fallback_signals[0][1] != fallback_signals[1][1]:
        indicators["decision_block"] = "fallback_module_disagreement"
        return None, indicators
    if fallback_signals:
        signal = fallback_signals[0][1]
        indicators["decision_modules"] = [name for name, _ in fallback_signals]
        indicators["decision_source"] = "+".join(indicators["decision_modules"])
        indicators.setdefault("win_probability", None)
        return signal, indicators

    indicators.update(weighted_indicators)
    indicators.update(model_indicators)
    if primary_count:
        indicators["decision_block"] = "primary_module_no_signal"
    return None, indicators


def _trained_model_signal(prices: list, *, candles: list = None) -> SignalResult:
    """Usa uma previsão válida do modelo como gerador direcional independente."""
    try:
        direction, confidence = ai_predictor.predict(prices, candles=candles)
        model_id = ai_predictor.prediction_model_id()
        confidence = float(confidence)
    except (Exception, TypeError, ValueError):
        return None, {"ai_block": "model_unavailable"}

    values = {
        "ai_direction": direction,
        "ai_confidence": confidence if math.isfinite(confidence) else None,
        "model_id": model_id,
        "execution_signal_mode": _execution_signal_mode(),
    }
    if direction not in ("BUY", "SELL"):
        values["ai_block"] = "model_abstention"
        return None, values
    if not model_id or not math.isfinite(confidence) or not AI_CONFIDENCE_MIN < confidence <= 1:
        values["ai_block"] = "invalid_model_confidence_or_id"
        return None, values
    values["win_probability"] = confidence
    return direction, values


def _price_action_signal(pa: Optional[dict]) -> tuple[Optional[str], float]:
    """Combina as features de price action em um score e retorna (BUY/SELL/None, score) pelo limiar."""
    if not pa:
        return None, 0.0
    score = (
        0.30 * pa.get("pa_market_structure", 0.0)
        + 0.25 * pa.get("pa_trend_consistency", 0.0)
        + 0.20 * pa.get("pa_bos_strength", 0.0)
        + 0.10 * pa.get("pa_candle_at_sr", 0.0)
        + 0.075 * (pa.get("pa_demand_zone", 0.0) - pa.get("pa_supply_zone", 0.0))
        + 0.075 * (pa.get("pa_fvg_bullish", 0.0) - pa.get("pa_fvg_bearish", 0.0))
    )
    threshold = max(float(SIGNAL_SCORE_MIN), 0.20)
    if score >= threshold:
        return "BUY", score
    if score <= -threshold:
        return "SELL", score
    return None, score


def _candle_pattern_signal(
    candle_alerts: list,
    evaluation_time: Optional[float] = None,
) -> tuple[Optional[str], Optional[str]]:
    """Direção dos padrões de candle recentes e fortes mais novos; (None, 'candle_pattern_conflict') se divergirem."""
    now = time.time() if evaluation_time is None else evaluation_time
    valid = [
        alert for alert in (candle_alerts or [])
        if 0 <= now - alert.get("timestamp", 0) <= CANDLE_PATTERN_MAX_AGE_SEC
        and alert.get("strength", 0.0) >= CANDLE_PATTERN_MIN_STRENGTH
        and alert.get("direction") in ("bullish", "bearish")
    ]
    if not valid:
        return None, None
    latest_epoch = max(alert.get("timestamp", 0) for alert in valid)
    latest = [alert for alert in valid if alert.get("timestamp", 0) == latest_epoch]
    directions = {alert["direction"] for alert in latest}
    if len(directions) != 1:
        return None, "candle_pattern_conflict"
    direction = "BUY" if directions.pop() == "bullish" else "SELL"
    return direction, None


# ─────────────────────────────────────────────────────────────────
#  P14 — Score ponderado por indicador
# ─────────────────────────────────────────────────────────────────

def _weighted_signal(
    ema9: float, ema21: float, macd_hist: float, mom: float,
    adx_val: float, rsi_val: float, price: float, indicators: dict, prices: list,
    candle_alerts: list = None,
    use_ai_model: Optional[bool] = None,
    evaluation_time: Optional[float] = None,
    candles: list = None,
    apply_aux_filters: bool = True,
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
    # ADX normalizado (0-1). No modo opportunity, ADX fraco penaliza mas não zera.
    adx_norm = min(adx_val / 40.0, 1.0)
    if _opportunity_mode():
        floor = float(getattr(config, "OPPORTUNITY_ADX_WEIGHT_FLOOR", 0.35) or 0.0)
        adx_norm = max(adx_norm, max(0.0, min(floor, 1.0)))

    # Score composto (positivo = bullish, negativo = bearish)
    composite = (
        ema_score * 0.25
        + macd_score * 0.25
        + price_score * 0.15
        + mom_score * 0.15
        + rsi_score * 0.10
    ) * adx_norm  # ADX modula a confiança

    indicators["tech_score"] = round(composite, 4)

    threshold = SIGNAL_SCORE_MIN
    if _opportunity_mode():
        threshold = float(getattr(config, "OPPORTUNITY_SIGNAL_SCORE_MIN", SIGNAL_SCORE_MIN) or SIGNAL_SCORE_MIN)

    if composite >= threshold:
        return _apply_ai_filter(
            "BUY",
            prices,
            indicators,
            candle_alerts,
            use_ai_model,
            evaluation_time,
            candles,
            apply_aux_filters=apply_aux_filters,
        )
    if composite <= -threshold:
        return _apply_ai_filter(
            "SELL",
            prices,
            indicators,
            candle_alerts,
            use_ai_model,
            evaluation_time,
            candles,
            apply_aux_filters=apply_aux_filters,
        )
    return None, indicators


# ─────────────────────────────────────────────────────────────────
#  Filtro de padrões de vela (soft conflict filter)
# ─────────────────────────────────────────────────────────────────

def _candle_pattern_filter(
    signal: str,
    candle_alerts: list,
    evaluation_time: Optional[float] = None,
) -> str | None:
    """
    Verifica se o padrão de vela mais recente contradiz o sinal técnico.

    Retorna motivo de bloqueio (str) se conflito direto detectado, ou None.
    Lógica suave: só bloqueia conflito direto — BUY+bearish ou SELL+bullish.
    Padrões neutros e padrões alinhados com o sinal não bloqueiam.
    """
    now = time.time() if evaluation_time is None else evaluation_time
    for alert in reversed(list(candle_alerts)):
        if not 0 <= now - alert.get("timestamp", 0) <= CANDLE_PATTERN_MAX_AGE_SEC:
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
    use_ai_model: Optional[bool] = None,
    evaluation_time: Optional[float] = None,
    candles: list = None,
    apply_aux_filters: bool = True,
) -> SignalResult:
    """
    Pondera o sinal dos indicadores técnicos com o modelo de IA e filtro PA.
    """
    # -- Filtro de padrões de vela: bloqueia conflito direto padrão vs. sinal --
    if apply_aux_filters and CANDLE_PATTERN_FILTER and candle_alerts:
        block_reason = _candle_pattern_filter(
            signal,
            candle_alerts,
            evaluation_time,
        )
        if block_reason:
            indicators["candle_pattern_block"] = block_reason
            print(f"[VELA] ⛔ Sinal {signal} bloqueado: {block_reason}")
            return None, indicators

    # -- Price Action: calcula telemetria e bloqueia apenas quando habilitado --
    if apply_aux_filters:
        if candles is None:
            candles = ind.ticks_to_candles(prices, CANDLE_SIZE)
        else:
            candles = candles[-FEATURE_WINDOW:]
        pa = ind.price_action_features(candles, PA_SR_TOLERANCE)
        if pa is not None:
            indicators["pa_sr_position"] = round(pa["pa_sr_position"], 4)
            indicators["pa_market_structure"] = round(pa["pa_market_structure"], 4)

            if PRICE_ACTION_FILTER:
                if signal == "BUY" and pa["pa_sr_position"] > 0.8 and pa["pa_bos_strength"] <= 0:
                    return None, indicators
                if signal == "SELL" and pa["pa_sr_position"] < -0.8 and pa["pa_bos_strength"] >= 0:
                    return None, indicators

    ai_enabled = USE_AI_MODEL if use_ai_model is None else use_ai_model
    if not ai_enabled:
        indicators["ai_confidence"] = None
        indicators["ai_score"]      = None
        return signal, indicators

    ai_direction, ai_conf = ai_predictor.predict(prices, candles=candles)
    indicators["ai_direction"] = ai_direction
    indicators["ai_confidence"] = ai_conf if math.isfinite(ai_conf) else None
    indicators["model_id"] = ai_predictor.prediction_model_id()
    indicators["execution_signal_mode"] = _execution_signal_mode()

    if not math.isfinite(ai_conf) or not 0 < ai_conf <= 1:
        indicators["ai_score"] = None
        indicators["ai_block"] = "invalid_ai_confidence"
        return None, indicators

    direct_confidence_min = max(float(AI_CONFIDENCE_MIN), 0.55)
    if ai_direction == signal and ai_conf < direct_confidence_min:
        indicators["ai_score"] = None
        indicators["ai_block"] = "low_direct_ai_confidence"
        return None, indicators

    directional_probability = ai_conf if ai_direction == signal else None
    if (
        directional_probability is None
        and _execution_signal_mode() == "trained_model"
        and ai_direction in ("BUY", "SELL")
    ):
        directional_probability = 1.0 - ai_conf
        indicators["ai_direction_override"] = "trained_model"

    if directional_probability is None or not AI_CONFIDENCE_MIN < directional_probability <= 1:
        indicators["ai_score"] = None
        indicators["ai_block"] = "abstention_or_disagreement"
        return None, indicators

    technical = min(1.0, abs(indicators.get("tech_score", 1.0)))
    score = AI_TECH_WEIGHT * technical + AI_MODEL_WEIGHT * directional_probability
    indicators["ai_score"] = round(score, 4)
    indicators["win_probability"] = directional_probability
    if score >= AI_SCORE_MIN:
        return signal, indicators
    return None, indicators
