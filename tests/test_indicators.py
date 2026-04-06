"""
tests/test_indicators.py — Testes unitários para indicators.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import indicators as ind


# ─── Dados sintéticos ──────────────────────────────────────────

def _make_prices(n=200, base=6500.0, step=0.1):
    """Gera lista de preços com tendência suave de alta."""
    import math
    return [base + i * step + math.sin(i / 10) * 2 for i in range(n)]


def _make_down_prices(n=200, base=6600.0, step=-0.1):
    """Gera lista de preços com tendência de baixa."""
    import math
    return [base + i * step + math.sin(i / 10) * 2 for i in range(n)]


# ─── Testes de indicadores existentes ──────────────────────────

class TestExistingIndicators:

    def test_ema_basic(self):
        prices = _make_prices(50)
        result = ind.ema(prices, 9)
        assert result is not None
        assert isinstance(result, float)

    def test_ema_insufficient(self):
        assert ind.ema([1.0, 2.0], 9) is None

    def test_rsi_range(self):
        prices = _make_prices(100)
        result = ind.rsi(prices, 14)
        assert result is not None
        assert 0.0 <= result <= 100.0

    def test_macd_returns_tuple(self):
        prices = _make_prices(100)
        result = ind.macd(prices, 12, 26, 9)
        assert result is not None
        assert len(result) == 3

    def test_adx_positive(self):
        prices = _make_prices(100)
        result = ind.adx(prices, 14)
        assert result is not None
        assert result >= 0.0

    def test_bollinger_order(self):
        prices = _make_prices(50)
        result = ind.bollinger(prices, 20, 2.0)
        assert result is not None
        upper, mid, lower = result
        assert upper >= mid >= lower

    def test_momentum_sign(self):
        prices = _make_prices(20)
        result = ind.momentum(prices, 3)
        assert result is not None


# ─── Testes de ticks_to_candles ─────────────────────────────────

class TestTicksToCandles:

    def test_basic_conversion(self):
        prices = list(range(1, 51))  # 50 ticks
        candles = ind.ticks_to_candles(prices, candle_size=10)
        assert len(candles) == 5

    def test_candle_ohlc(self):
        prices = [10.0, 12.0, 8.0, 11.0, 9.0, 13.0, 7.0, 14.0, 10.0, 15.0]
        candles = ind.ticks_to_candles(prices, candle_size=10)
        assert len(candles) == 1
        c = candles[0]
        assert c["open"] == 10.0
        assert c["close"] == 15.0
        assert c["high"] == 15.0
        assert c["low"] == 7.0

    def test_partial_ignored(self):
        prices = list(range(1, 16))  # 15 ticks, candle_size=10
        candles = ind.ticks_to_candles(prices, candle_size=10)
        assert len(candles) == 1  # apenas 1 vela completa

    def test_empty(self):
        candles = ind.ticks_to_candles([], candle_size=10)
        assert candles == []


# ─── Testes de Price Action features ───────────────────────────

class TestPriceActionFeatures:

    def test_insufficient_candles_returns_none(self):
        candles = [{"open": 1, "high": 2, "low": 0.5, "close": 1.5} for _ in range(10)]
        result = ind.price_action_features(candles)
        assert result is None

    def test_sufficient_candles_returns_dict(self):
        prices = _make_prices(300)
        candles = ind.ticks_to_candles(prices, 10)
        assert len(candles) >= 15
        result = ind.price_action_features(candles)
        assert result is not None
        assert len(result) == 11

    def test_all_features_in_range(self):
        prices = _make_prices(300)
        candles = ind.ticks_to_candles(prices, 10)
        result = ind.price_action_features(candles)
        assert result is not None

        bounded_neg1_pos1 = [
            "pa_market_structure", "pa_bos_strength", "pa_trend_consistency",
            "pa_sr_position", "pa_candle_at_sr",
        ]
        bounded_0_1 = [
            "pa_sr_distance", "pa_sr_touch_count",
            "pa_demand_zone", "pa_supply_zone",
            "pa_fvg_bullish", "pa_fvg_bearish",
        ]

        for key in bounded_neg1_pos1:
            assert -1.0 <= result[key] <= 1.0, f"{key}={result[key]} fora de [-1, 1]"

        for key in bounded_0_1:
            assert 0.0 <= result[key] <= 1.0, f"{key}={result[key]} fora de [0, 1]"

    def test_feature_keys(self):
        prices = _make_prices(300)
        candles = ind.ticks_to_candles(prices, 10)
        result = ind.price_action_features(candles)
        expected_keys = {
            "pa_market_structure", "pa_bos_strength", "pa_trend_consistency",
            "pa_sr_distance", "pa_sr_touch_count", "pa_sr_position",
            "pa_demand_zone", "pa_supply_zone",
            "pa_fvg_bullish", "pa_fvg_bearish", "pa_candle_at_sr",
        }
        assert set(result.keys()) == expected_keys


# ─── Testes de detecção de padrões de vela ─────────────────────

class TestDetectCandlePatterns:

    def test_empty_candles(self):
        result = ind.detect_candle_patterns([])
        assert result == []

    def test_insufficient_candles(self):
        candles = [{"open": 1, "high": 2, "low": 0.5, "close": 1.5}]
        result = ind.detect_candle_patterns(candles)
        assert result == []

    def test_bullish_engulfing(self):
        candles = [
            {"open": 100, "high": 101, "low": 99, "close": 100},  # filler
            {"open": 100, "high": 101, "low": 99, "close": 100},  # filler
            {"open": 102, "high": 102, "low": 98, "close": 99},   # bearish
            {"open": 97, "high": 105, "low": 97, "close": 104},   # bullish engulfs
        ]
        result = ind.detect_candle_patterns(candles)
        names = [p["name"] for p in result]
        assert "Bullish Engulfing" in names

    def test_bearish_engulfing(self):
        candles = [
            {"open": 100, "high": 101, "low": 99, "close": 100},
            {"open": 100, "high": 101, "low": 99, "close": 100},
            {"open": 98, "high": 102, "low": 98, "close": 101},   # bullish
            {"open": 103, "high": 103, "low": 96, "close": 97},   # bearish engulfs
        ]
        result = ind.detect_candle_patterns(candles)
        names = [p["name"] for p in result]
        assert "Bearish Engulfing" in names

    def test_hammer(self):
        candles = [
            {"open": 100, "high": 101, "low": 99, "close": 100},
            {"open": 100, "high": 101, "low": 99, "close": 100},
            {"open": 100, "high": 101, "low": 99, "close": 100},
            # Hammer: sombra inferior longa, corpo visível, sombra superior mínima
            {"open": 100, "high": 101, "low": 94, "close": 101},
        ]
        result = ind.detect_candle_patterns(candles)
        names = [p["name"] for p in result]
        assert "Hammer" in names

    def test_shooting_star(self):
        candles = [
            {"open": 100, "high": 101, "low": 99, "close": 100},
            {"open": 100, "high": 101, "low": 99, "close": 100},
            {"open": 100, "high": 101, "low": 99, "close": 100},
            # Shooting star: sombra superior longa, corpo visível
            {"open": 101, "high": 107, "low": 100.8, "close": 100},
        ]
        result = ind.detect_candle_patterns(candles)
        names = [p["name"] for p in result]
        assert "Shooting Star" in names

    def test_doji(self):
        candles = [
            {"open": 100, "high": 101, "low": 99, "close": 100},
            {"open": 100, "high": 101, "low": 99, "close": 100},
            {"open": 100, "high": 101, "low": 99, "close": 100},
            # Doji: corpo mínimo
            {"open": 100.0, "high": 102, "low": 98, "close": 100.1},
        ]
        result = ind.detect_candle_patterns(candles)
        names = [p["name"] for p in result]
        assert "Doji" in names

    def test_three_white_soldiers(self):
        candles = [
            {"open": 100, "high": 101, "low": 99, "close": 100},  # filler
            {"open": 100, "high": 103, "low": 99.5, "close": 102},
            {"open": 102, "high": 106, "low": 101.5, "close": 105},
            {"open": 105, "high": 110, "low": 104.5, "close": 109},
        ]
        result = ind.detect_candle_patterns(candles)
        names = [p["name"] for p in result]
        assert "Three White Soldiers" in names

    def test_three_black_crows(self):
        candles = [
            {"open": 100, "high": 101, "low": 99, "close": 100},  # filler
            {"open": 102, "high": 102.5, "low": 99, "close": 100},
            {"open": 100, "high": 100.5, "low": 96, "close": 97},
            {"open": 97, "high": 97.5, "low": 92, "close": 93},
        ]
        result = ind.detect_candle_patterns(candles)
        names = [p["name"] for p in result]
        assert "Three Black Crows" in names

    def test_pattern_has_strength(self):
        candles = [
            {"open": 100, "high": 101, "low": 99, "close": 100},
            {"open": 100, "high": 101, "low": 99, "close": 100},
            {"open": 102, "high": 102, "low": 98, "close": 99},
            {"open": 97, "high": 105, "low": 97, "close": 104},
        ]
        result = ind.detect_candle_patterns(candles)
        for p in result:
            assert "strength" in p
            assert 0.0 <= p["strength"] <= 1.0
            assert p["direction"] in ("bullish", "bearish", "neutral")
