"""
feature_engine.py — única fonte de verdade para extração de features.

Consolida a lógica de _extract_features / _compute_feature_map que estava
duplicada em dataset_builder.py e ai_predictor.py.

Uso:
    from feature_engine import compute_feature_map, extract_feature_vector

    fm = compute_feature_map(prices)           # -> dict | None
    vec = extract_feature_vector(prices, FEATURES)  # -> list[float] | None
"""

from typing import Optional

import indicators as ind
from config import (
    EMA_FAST, EMA_SLOW,
    RSI_PERIOD,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    ADX_PERIOD,
    BB_PERIOD, BB_STD,
    CANDLE_SIZE, PA_SR_TOLERANCE,
)

# ─── Lista canônica de features — mantida aqui como fonte única ────────────────
FEATURES = [
    "ema9", "ema21", "ema_cross",
    "rsi",
    "macd_line", "macd_hist",
    "adx",
    "bb_width", "bb_position",
    "momentum3",
    "return_1", "return_3", "return_5",
    "volatility_10", "volatility_20",
    "high_low_5",
    # Price Action features
    "pa_market_structure", "pa_bos_strength", "pa_trend_consistency",
    "pa_sr_distance", "pa_sr_touch_count", "pa_sr_position",
    "pa_demand_zone", "pa_supply_zone",
    "pa_fvg_bullish", "pa_fvg_bearish", "pa_candle_at_sr",
]

_PA_DEFAULTS = {
    "pa_market_structure": 0.0, "pa_bos_strength": 0.0, "pa_trend_consistency": 0.0,
    "pa_sr_distance": 0.0, "pa_sr_touch_count": 0.0, "pa_sr_position": 0.0,
    "pa_demand_zone": 0.0, "pa_supply_zone": 0.0,
    "pa_fvg_bullish": 0.0, "pa_fvg_bearish": 0.0, "pa_candle_at_sr": 0.0,
}


def compute_feature_map(prices: list) -> Optional[dict]:
    """
    Calcula todos os indicadores para uma janela de preços.

    Retorna dict com 27 features ou None se dados insuficientes.
    """
    ema9  = ind.ema(prices, EMA_FAST)
    ema21 = ind.ema(prices, EMA_SLOW)
    rsi_v = ind.rsi(prices, RSI_PERIOD)
    macd_ = ind.macd(prices, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    adx_v = ind.adx(prices, ADX_PERIOD)
    bb_   = ind.bollinger(prices, BB_PERIOD, BB_STD)
    mom3  = ind.momentum(prices, 3)

    if any(v is None for v in [ema9, ema21, rsi_v, macd_, adx_v, mom3]):
        return None

    macd_line, _sig, macd_hist = macd_
    price = prices[-1]

    ema_cross = (ema9 - ema21) / price if price != 0 else 0.0

    bb_width    = 0.0
    bb_position = 0.0
    if bb_ is not None:
        bb_upper, bb_mid, bb_lower = bb_
        band_range = bb_upper - bb_lower
        if band_range > 0:
            bb_width    = band_range / bb_mid if bb_mid != 0 else 0.0
            bb_position = (price - bb_lower) / band_range

    def _ret(n: int) -> float:
        if len(prices) <= n:
            return 0.0
        p_prev = prices[-1 - n]
        return (price - p_prev) / p_prev if p_prev != 0 else 0.0

    def _vol(n: int) -> float:
        slice_ = prices[-n - 1:]
        if len(slice_) < 2:
            return 0.0
        rets = [(slice_[i] - slice_[i - 1]) / slice_[i - 1]
                for i in range(1, len(slice_))
                if slice_[i - 1] != 0]
        if not rets:
            return 0.0
        mean_ = sum(rets) / len(rets)
        var_  = sum((r - mean_) ** 2 for r in rets) / len(rets)
        return var_ ** 0.5

    last5 = prices[-5:]
    high5 = max(last5)
    low5  = min(last5)
    hl5   = (high5 - low5) / price if price != 0 else 0.0

    candles = ind.ticks_to_candles(prices, CANDLE_SIZE)
    pa = ind.price_action_features(candles, PA_SR_TOLERANCE)
    pa_features = pa if pa is not None else _PA_DEFAULTS

    return {
        "ema9":          ema9,
        "ema21":         ema21,
        "ema_cross":     ema_cross,
        "rsi":           rsi_v,
        "macd_line":     macd_line,
        "macd_hist":     macd_hist,
        "adx":           adx_v,
        "bb_width":      bb_width,
        "bb_position":   bb_position,
        "momentum3":     mom3,
        "return_1":      _ret(1),
        "return_3":      _ret(3),
        "return_5":      _ret(5),
        "volatility_10": _vol(10),
        "volatility_20": _vol(20),
        "high_low_5":    hl5,
        **pa_features,
    }


def extract_feature_vector(prices: list, feature_order: list) -> Optional[list]:
    """
    Extrai features de uma lista de preços como vetor ordenado.

    Retorna list[float] na ordem de feature_order, ou None se dados insuficientes.
    """
    fm = compute_feature_map(prices)
    if fm is None:
        return None
    return [fm.get(f, 0.0) for f in feature_order]
