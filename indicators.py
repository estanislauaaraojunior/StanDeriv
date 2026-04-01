"""
indicators.py — funções puras de análise técnica.

Todas as funções:
  - Recebem uma lista simples de floats (preços de fechamento / ticks)
  - Retornam um valor ou None quando dados insuficientes
  - Sem estado interno — seguras para backtesting e uso em modelos de IA
"""

import math
from typing import Optional, Tuple, List, Dict


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


# ─────────────────────────────────────────────────────────────
#  Price Action — funções de vela e estrutura de mercado
# ─────────────────────────────────────────────────────────────

def ticks_to_candles(prices: list, candle_size: int = 10) -> List[Dict[str, float]]:
    """
    Converte lista de ticks em velas OHLC sintéticas.
    Cada vela agrupa `candle_size` ticks consecutivos.
    Retorna lista de dicts com keys: open, high, low, close.
    """
    candles = []
    for i in range(0, len(prices) - candle_size + 1, candle_size):
        chunk = prices[i : i + candle_size]
        candles.append({
            "open": chunk[0],
            "high": max(chunk),
            "low": min(chunk),
            "close": chunk[-1],
        })
    return candles


def _find_swing_points(
    candles: List[Dict[str, float]], lookback: int = 3
) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    """
    Identifica swing highs e swing lows nas velas.
    Um swing high é uma vela cujo high é maior que os `lookback` vizinhos.
    Retorna (swing_highs, swing_lows) como listas de (índice, preço).
    """
    swing_highs: List[Tuple[int, float]] = []
    swing_lows: List[Tuple[int, float]] = []

    for i in range(lookback, len(candles) - lookback):
        hi = candles[i]["high"]
        lo = candles[i]["low"]

        is_high = all(hi >= candles[i + d]["high"] for d in range(-lookback, lookback + 1) if d != 0)
        is_low = all(lo <= candles[i + d]["low"] for d in range(-lookback, lookback + 1) if d != 0)

        if is_high:
            swing_highs.append((i, hi))
        if is_low:
            swing_lows.append((i, lo))

    return swing_highs, swing_lows


def _cluster_sr_levels(
    swing_highs: List[Tuple[int, float]],
    swing_lows: List[Tuple[int, float]],
    tolerance: float = 0.001,
) -> List[Tuple[float, int]]:
    """
    Agrupa swing points próximos em níveis S/R.
    Retorna lista de (nível, contagem_toques) ordenada por contagem decrescente.
    """
    all_levels = [p for _, p in swing_highs] + [p for _, p in swing_lows]
    if not all_levels:
        return []

    all_levels.sort()
    clusters: List[List[float]] = [[all_levels[0]]]

    for level in all_levels[1:]:
        if abs(level - clusters[-1][-1]) / (clusters[-1][-1] + 1e-12) <= tolerance:
            clusters[-1].append(level)
        else:
            clusters.append([level])

    sr_levels = [(sum(c) / len(c), len(c)) for c in clusters]
    sr_levels.sort(key=lambda x: x[1], reverse=True)
    return sr_levels


def _find_supply_demand_zones(
    candles: List[Dict[str, float]], lookback: int = 3
) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float, float]]]:
    """
    Identifica zonas de demand (consolidação antes de alta forte) e
    supply (consolidação antes de queda forte).
    Retorna (demand_zones, supply_zones) como listas de (low, high, strength).
    """
    demand_zones: List[Tuple[float, float, float]] = []
    supply_zones: List[Tuple[float, float, float]] = []

    for i in range(lookback, len(candles) - 1):
        body = abs(candles[i + 0]["close"] - candles[i + 0]["open"])
        rng = candles[i]["high"] - candles[i]["low"]
        if rng == 0:
            continue

        # Vela de impulso: corpo > 60% do range
        if body / rng < 0.6:
            continue

        # Verificar consolidação antes (corpos pequenos)
        avg_body_before = sum(
            abs(candles[i - j]["close"] - candles[i - j]["open"])
            for j in range(1, min(lookback + 1, i + 1))
        ) / lookback

        if avg_body_before == 0:
            continue

        impulse_ratio = body / avg_body_before
        if impulse_ratio < 2.0:
            continue

        strength = min(impulse_ratio / 5.0, 1.0)

        if candles[i]["close"] > candles[i]["open"]:
            # Impulso bullish → demand zone na consolidação
            zone_lo = min(candles[i - j]["low"] for j in range(1, min(lookback + 1, i + 1)))
            zone_hi = max(candles[i - j]["high"] for j in range(1, min(lookback + 1, i + 1)))
            demand_zones.append((zone_lo, zone_hi, strength))
        else:
            # Impulso bearish → supply zone na consolidação
            zone_lo = min(candles[i - j]["low"] for j in range(1, min(lookback + 1, i + 1)))
            zone_hi = max(candles[i - j]["high"] for j in range(1, min(lookback + 1, i + 1)))
            supply_zones.append((zone_lo, zone_hi, strength))

    return demand_zones, supply_zones


def _find_fvg(
    candles: List[Dict[str, float]],
) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float, float]]]:
    """
    Fair Value Gaps: gap entre vela i-1 e vela i+1 (vela i é a de impulso).
    Bullish FVG: low[i+1] > high[i-1]
    Bearish FVG: high[i+1] < low[i-1]
    Retorna (bullish_fvgs, bearish_fvgs) como (gap_lo, gap_hi, size).
    """
    bullish: List[Tuple[float, float, float]] = []
    bearish: List[Tuple[float, float, float]] = []

    for i in range(1, len(candles) - 1):
        prev_hi = candles[i - 1]["high"]
        next_lo = candles[i + 1]["low"]
        prev_lo = candles[i - 1]["low"]
        next_hi = candles[i + 1]["high"]

        if next_lo > prev_hi:
            gap_size = next_lo - prev_hi
            bullish.append((prev_hi, next_lo, gap_size))
        elif next_hi < prev_lo:
            gap_size = prev_lo - next_hi
            bearish.append((next_hi, prev_lo, gap_size))

    return bullish, bearish


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def price_action_features(
    candles: List[Dict[str, float]],
    sr_tolerance: float = 0.001,
) -> Optional[Dict[str, float]]:
    """
    Calcula 11 features numéricas de Price Action a partir de velas OHLC.
    Retorna None se < 15 velas (dados insuficientes).
    
    Features retornadas (todas normalizadas):
      pa_market_structure  (-1 a +1): HH/HL vs LH/LL
      pa_bos_strength      (-1 a +1): força do break of structure
      pa_trend_consistency  (-1 a +1): sequência de swings alinhados
      pa_sr_distance        (0 a 1): distância ao S/R mais próximo
      pa_sr_touch_count     (0 a 1): força do nível S/R
      pa_sr_position       (-1 a +1): perto de suporte vs resistência
      pa_demand_zone        (0 a 1): proximidade de demand zone
      pa_supply_zone        (0 a 1): proximidade de supply zone
      pa_fvg_bullish        (0 a 1): FVG bullish por preencher
      pa_fvg_bearish        (0 a 1): FVG bearish por preencher
      pa_candle_at_sr      (-1 a +1): padrão de rejeição em zona S/R
    """
    if len(candles) < 15:
        return None

    current_price = candles[-1]["close"]
    current_range = candles[-1]["high"] - candles[-1]["low"]
    eps = 1e-12

    # --- Swing points e S/R ---
    swing_highs, swing_lows = _find_swing_points(candles)
    sr_levels = _cluster_sr_levels(swing_highs, swing_lows, sr_tolerance)

    # --- Market Structure ---
    pa_market_structure = 0.0
    pa_bos_strength = 0.0
    pa_trend_consistency = 0.0

    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        last_highs = [h for _, h in swing_highs[-4:]]
        last_lows = [l for _, l in swing_lows[-4:]]

        # HH/HL count vs LH/LL count
        hh_count = sum(1 for i in range(1, len(last_highs)) if last_highs[i] > last_highs[i - 1])
        lh_count = sum(1 for i in range(1, len(last_highs)) if last_highs[i] < last_highs[i - 1])
        hl_count = sum(1 for i in range(1, len(last_lows)) if last_lows[i] > last_lows[i - 1])
        ll_count = sum(1 for i in range(1, len(last_lows)) if last_lows[i] < last_lows[i - 1])

        bullish_pts = hh_count + hl_count
        bearish_pts = lh_count + ll_count
        total = bullish_pts + bearish_pts + eps

        pa_market_structure = _clamp((bullish_pts - bearish_pts) / total, -1.0, 1.0)

        # Break of Structure
        prev_swing_hi = swing_highs[-2][1] if len(swing_highs) >= 2 else current_price
        prev_swing_lo = swing_lows[-2][1] if len(swing_lows) >= 2 else current_price
        price_range = prev_swing_hi - prev_swing_lo + eps

        if current_price > prev_swing_hi:
            pa_bos_strength = _clamp((current_price - prev_swing_hi) / price_range, 0.0, 1.0)
        elif current_price < prev_swing_lo:
            pa_bos_strength = _clamp(-(prev_swing_lo - current_price) / price_range, -1.0, 0.0)

        # Trend consistency
        consistency = 0
        for i in range(1, min(len(last_highs), len(last_lows))):
            if last_highs[i] > last_highs[i - 1] and last_lows[i] > last_lows[i - 1]:
                consistency += 1
            elif last_highs[i] < last_highs[i - 1] and last_lows[i] < last_lows[i - 1]:
                consistency -= 1
        max_consistency = max(min(len(last_highs), len(last_lows)) - 1, 1)
        pa_trend_consistency = _clamp(consistency / max_consistency, -1.0, 1.0)

    # --- S/R features ---
    pa_sr_distance = 1.0
    pa_sr_touch_count = 0.0
    pa_sr_position = 0.0

    if sr_levels:
        nearest_sr = min(sr_levels, key=lambda x: abs(x[0] - current_price))
        dist = abs(nearest_sr[0] - current_price) / (current_price + eps)
        pa_sr_distance = _clamp(dist / 0.005, 0.0, 1.0)  # normalizar: 0.5% = 1.0
        pa_sr_touch_count = _clamp(nearest_sr[1] / 5.0, 0.0, 1.0)  # 5+ toques = 1.0

        # Posição relativa ao S/R mais próximo
        sr_above = [s for s, _ in sr_levels if s > current_price]
        sr_below = [s for s, _ in sr_levels if s < current_price]

        if sr_above and sr_below:
            nearest_above = min(sr_above)
            nearest_below = max(sr_below)
            total_range = nearest_above - nearest_below + eps
            pa_sr_position = _clamp(
                2.0 * (current_price - nearest_below) / total_range - 1.0, -1.0, 1.0
            )
        elif sr_above:
            pa_sr_position = -0.8
        elif sr_below:
            pa_sr_position = 0.8

    # --- Supply/Demand zones ---
    pa_demand_zone = 0.0
    pa_supply_zone = 0.0
    demand_zones, supply_zones = _find_supply_demand_zones(candles)

    for zone_lo, zone_hi, strength in demand_zones:
        if zone_lo <= current_price <= zone_hi:
            pa_demand_zone = max(pa_demand_zone, strength)
        else:
            dist_to_zone = min(abs(current_price - zone_lo), abs(current_price - zone_hi))
            proximity = 1.0 - _clamp(dist_to_zone / (current_price * 0.003 + eps), 0.0, 1.0)
            pa_demand_zone = max(pa_demand_zone, proximity * strength)

    for zone_lo, zone_hi, strength in supply_zones:
        if zone_lo <= current_price <= zone_hi:
            pa_supply_zone = max(pa_supply_zone, strength)
        else:
            dist_to_zone = min(abs(current_price - zone_lo), abs(current_price - zone_hi))
            proximity = 1.0 - _clamp(dist_to_zone / (current_price * 0.003 + eps), 0.0, 1.0)
            pa_supply_zone = max(pa_supply_zone, proximity * strength)

    # --- FVG ---
    pa_fvg_bullish = 0.0
    pa_fvg_bearish = 0.0
    bullish_fvgs, bearish_fvgs = _find_fvg(candles)

    for gap_lo, gap_hi, gap_size in bullish_fvgs[-3:]:
        if gap_lo <= current_price <= gap_hi:
            pa_fvg_bullish = max(pa_fvg_bullish, _clamp(gap_size / (current_price * 0.002 + eps), 0.0, 1.0))
        elif current_price < gap_lo:
            dist = gap_lo - current_price
            proximity = 1.0 - _clamp(dist / (current_price * 0.003 + eps), 0.0, 1.0)
            pa_fvg_bullish = max(pa_fvg_bullish, proximity * 0.5)

    for gap_lo, gap_hi, gap_size in bearish_fvgs[-3:]:
        if gap_lo <= current_price <= gap_hi:
            pa_fvg_bearish = max(pa_fvg_bearish, _clamp(gap_size / (current_price * 0.002 + eps), 0.0, 1.0))
        elif current_price > gap_hi:
            dist = current_price - gap_hi
            proximity = 1.0 - _clamp(dist / (current_price * 0.003 + eps), 0.0, 1.0)
            pa_fvg_bearish = max(pa_fvg_bearish, proximity * 0.5)

    # --- Candle at S/R ---
    pa_candle_at_sr = 0.0
    if pa_sr_distance < 0.3:
        upper_wick = candles[-1]["high"] - max(candles[-1]["open"], candles[-1]["close"])
        lower_wick = min(candles[-1]["open"], candles[-1]["close"]) - candles[-1]["low"]
        body = abs(candles[-1]["close"] - candles[-1]["open"])

        if current_range > 0:
            if lower_wick > body * 2:
                pa_candle_at_sr = _clamp(lower_wick / current_range, 0.0, 1.0)
            elif upper_wick > body * 2:
                pa_candle_at_sr = -_clamp(upper_wick / current_range, 0.0, 1.0)

    return {
        "pa_market_structure": round(pa_market_structure, 6),
        "pa_bos_strength": round(pa_bos_strength, 6),
        "pa_trend_consistency": round(pa_trend_consistency, 6),
        "pa_sr_distance": round(pa_sr_distance, 6),
        "pa_sr_touch_count": round(pa_sr_touch_count, 6),
        "pa_sr_position": round(pa_sr_position, 6),
        "pa_demand_zone": round(pa_demand_zone, 6),
        "pa_supply_zone": round(pa_supply_zone, 6),
        "pa_fvg_bullish": round(pa_fvg_bullish, 6),
        "pa_fvg_bearish": round(pa_fvg_bearish, 6),
        "pa_candle_at_sr": round(pa_candle_at_sr, 6),
    }


def detect_candle_patterns(
    candles: List[Dict[str, float]],
) -> List[Dict[str, object]]:
    """
    Detecta os 6 padrões de vela mais relevantes:
      1. Bullish Engulfing
      2. Bearish Engulfing
      3. Hammer
      4. Shooting Star
      5. Doji
      6. Three White Soldiers / Three Black Crows

    Retorna lista de dicts: {name, direction, strength}
    strength é float 0-1 (força do padrão).
    """
    patterns: List[Dict[str, object]] = []
    if len(candles) < 3:
        return patterns

    c = candles[-1]
    p = candles[-2]
    eps = 1e-12

    c_body = c["close"] - c["open"]
    c_abs_body = abs(c_body)
    c_range = c["high"] - c["low"] + eps
    c_upper = c["high"] - max(c["open"], c["close"])
    c_lower = min(c["open"], c["close"]) - c["low"]

    p_body = p["close"] - p["open"]
    p_abs_body = abs(p_body)
    p_range = p["high"] - p["low"] + eps

    # 1. Bullish Engulfing
    if p_body < 0 and c_body > 0 and c_abs_body > p_abs_body:
        engulf_ratio = c_abs_body / (p_abs_body + eps)
        strength = _clamp((engulf_ratio - 1.0) / 2.0, 0.1, 1.0)
        patterns.append({"name": "Bullish Engulfing", "direction": "bullish", "strength": round(strength, 2)})

    # 2. Bearish Engulfing
    if p_body > 0 and c_body < 0 and c_abs_body > p_abs_body:
        engulf_ratio = c_abs_body / (p_abs_body + eps)
        strength = _clamp((engulf_ratio - 1.0) / 2.0, 0.1, 1.0)
        patterns.append({"name": "Bearish Engulfing", "direction": "bearish", "strength": round(strength, 2)})

    # 3. Hammer (reversão de alta)
    if c_lower >= c_abs_body * 2 and c_upper <= c_abs_body * 0.5 and c_abs_body > 0:
        strength = _clamp(c_lower / (c_range * 0.7), 0.1, 1.0)
        patterns.append({"name": "Hammer", "direction": "bullish", "strength": round(strength, 2)})

    # 4. Shooting Star (reversão de baixa)
    if c_upper >= c_abs_body * 2 and c_lower <= c_abs_body * 0.5 and c_abs_body > 0:
        strength = _clamp(c_upper / (c_range * 0.7), 0.1, 1.0)
        patterns.append({"name": "Shooting Star", "direction": "bearish", "strength": round(strength, 2)})

    # 5. Doji (indecisão)
    if c_abs_body <= c_range * 0.1:
        strength = _clamp(1.0 - c_abs_body / (c_range * 0.1 + eps), 0.1, 1.0)
        patterns.append({"name": "Doji", "direction": "neutral", "strength": round(strength, 2)})

    # 6. Three White Soldiers / Three Black Crows
    if len(candles) >= 4:
        c1 = candles[-3]
        c2 = candles[-2]
        c3 = candles[-1]

        b1 = c1["close"] - c1["open"]
        b2 = c2["close"] - c2["open"]
        b3 = c3["close"] - c3["open"]

        # Three White Soldiers
        if b1 > 0 and b2 > 0 and b3 > 0:
            if c2["close"] > c1["close"] and c3["close"] > c2["close"]:
                bodies = [abs(b1), abs(b2), abs(b3)]
                growing = all(bodies[i] >= bodies[i - 1] * 0.8 for i in range(1, 3))
                if growing:
                    avg_body = sum(bodies) / 3
                    avg_range = sum(
                        candles[-3 + j]["high"] - candles[-3 + j]["low"]
                        for j in range(3)
                    ) / 3 + eps
                    strength = _clamp(avg_body / avg_range, 0.1, 1.0)
                    patterns.append({"name": "Three White Soldiers", "direction": "bullish", "strength": round(strength, 2)})

        # Three Black Crows
        if b1 < 0 and b2 < 0 and b3 < 0:
            if c2["close"] < c1["close"] and c3["close"] < c2["close"]:
                bodies = [abs(b1), abs(b2), abs(b3)]
                growing = all(bodies[i] >= bodies[i - 1] * 0.8 for i in range(1, 3))
                if growing:
                    avg_body = sum(bodies) / 3
                    avg_range = sum(
                        candles[-3 + j]["high"] - candles[-3 + j]["low"]
                        for j in range(3)
                    ) / 3 + eps
                    strength = _clamp(avg_body / avg_range, 0.1, 1.0)
                    patterns.append({"name": "Three Black Crows", "direction": "bearish", "strength": round(strength, 2)})

    return patterns
