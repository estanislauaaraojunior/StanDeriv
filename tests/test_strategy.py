"""
tests/test_strategy.py — testes unitários para strategy.get_signal().
"""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("DERIV_TOKEN", "test_token")

import strategy
import config


# ──────────────────────────────────────────────────────────────────
#  Geradores de preços sintéticos
# ──────────────────────────────────────────────────────────────────

def _uptrend(n=200, start=100.0, step=0.05):
    """Série estritamente crescente (gera BUY)."""
    return [start + i * step for i in range(n)]


def _downtrend(n=200, start=115.0, step=0.05):
    """Série estritamente decrescente (gera SELL)."""
    return [start - i * step for i in range(n)]


def _lateral(n=200, center=100.0, amp=0.02):
    """Oscilação pequena em torno de um centro (ADX baixo → sem sinal)."""
    import math
    return [center + amp * math.sin(2 * math.pi * i / 20) for i in range(n)]


# ──────────────────────────────────────────────────────────────────
#  Helper: desabilita filtro AI para testar só a lógica técnica
# ──────────────────────────────────────────────────────────────────

def _no_ai():
    """Retorna patch que desliga USE_AI_MODEL na estratégia."""
    return patch.object(strategy, "USE_AI_MODEL", False)


def _no_weighted():
    """Retorna patch que desliga USE_WEIGHTED_SIGNAL na estratégia."""
    return patch.object(strategy, "USE_WEIGHTED_SIGNAL", False)


# ──────────────────────────────────────────────────────────────────
#  Testes
# ──────────────────────────────────────────────────────────────────

class TestGetSignalBasic(unittest.TestCase):
    """Verifica o formato básico do retorno de get_signal."""

    def test_returns_tuple(self):
        """Garante que get_signal retorna uma tupla de dois elementos."""
        prices = _uptrend()
        result = strategy.get_signal(prices)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_returns_none_too_short(self):
        """Série muito curta → dados insuficientes → None."""
        sig, ind = strategy.get_signal([100.0] * 5)
        self.assertIsNone(sig)

    def test_indicators_populated_on_valid_data(self):
        """Garante que os indicadores ema9 e ema21 são retornados com dados válidos."""
        prices = _uptrend(200)
        sig, ind = strategy.get_signal(prices)
        # indicators devem conter pelo menos ema9
        self.assertIn("ema9", ind)
        self.assertIn("ema21", ind)


class TestBuySignal(unittest.TestCase):
    """Verifica que tendências de alta não geram SELL."""

    def test_buy_on_uptrend(self):
        """Tendência de alta bem formada → BUY ou None (nunca SELL)."""
        prices = _uptrend(300, step=0.15)
        with _no_ai(), _no_weighted():
            sig, _ = strategy.get_signal(prices, adx_min=0)
        self.assertNotEqual(sig, "SELL")

    def test_no_sell_on_strong_uptrend(self):
        """Garante que alta forte nunca gera SELL."""
        prices = _uptrend(300, step=0.20)
        with _no_ai(), _no_weighted():
            sig, _ = strategy.get_signal(prices, adx_min=0)
        self.assertNotEqual(sig, "SELL")


class TestSellSignal(unittest.TestCase):
    """Verifica que tendências de baixa não geram BUY."""

    def test_sell_on_downtrend(self):
        """Tendência de baixa bem formada → SELL ou None (nunca BUY)."""
        prices = _downtrend(300, step=0.15)
        with _no_ai(), _no_weighted():
            sig, _ = strategy.get_signal(prices, adx_min=0)
        self.assertNotEqual(sig, "BUY")

    def test_no_buy_on_strong_downtrend(self):
        """Garante que baixa forte nunca gera BUY."""
        prices = _downtrend(300, step=0.20)
        with _no_ai(), _no_weighted():
            sig, _ = strategy.get_signal(prices, adx_min=0)
        self.assertNotEqual(sig, "BUY")


class TestNoSignalLateral(unittest.TestCase):
    """Verifica que mercado lateral ou ADX alto demais não produzem sinal."""

    def test_no_signal_lateral_high_adx(self):
        """Mercado lateral com ADX mínimo alto → None."""
        prices = _lateral(300)
        with _no_ai(), _no_weighted():
            sig, _ = strategy.get_signal(prices, adx_min=50)
        self.assertIsNone(sig)

    def test_adx_filter_blocks_signal(self):
        """ADX forçado a zero filtra qualquer sinal."""
        prices = _uptrend(300, step=0.20)
        with _no_ai(), _no_weighted(), patch.object(config, "ENTRY_MODE", "balanced", create=True):
            sig, ind = strategy.get_signal(prices, adx_min=9999)
        self.assertIsNone(sig)


class TestAdaptiveADX(unittest.TestCase):
    """Verifica o ADX mínimo adaptativo."""

    def test_adaptive_adx_returns_float(self):
        """Garante que get_adaptive_adx_min retorna float para um histórico válido."""
        history = [15.0, 20.0, 25.0, 30.0, 35.0]
        val = strategy.get_adaptive_adx_min(history)
        self.assertIsInstance(val, float)

    def test_adaptive_adx_empty_history_returns_adx_min(self):
        """Garante que histórico vazio devolve o ADX_MIN configurado."""
        val = strategy.get_adaptive_adx_min([])
        self.assertEqual(val, config.ADX_MIN)

    def test_adaptive_adx_min_floor(self):
        """Nunca abaixo de 15."""
        with patch.object(config, "ADX_ADAPTIVE", True), \
             patch.object(config, "ADX_ADAPTIVE_PERCENTILE", 1):
            val = strategy.get_adaptive_adx_min([1.0, 2.0, 3.0])
        self.assertGreaterEqual(val, 15.0)


class TestWeightedSignal(unittest.TestCase):
    """Verifica o modo de sinal ponderado."""

    def test_weighted_mode_runs(self):
        """Modo ponderado não deve lançar exceção."""
        prices = _uptrend(300, step=0.10)
        with _no_ai(), patch.object(strategy, "USE_WEIGHTED_SIGNAL", True):
            result = strategy.get_signal(prices, adx_min=0)
        self.assertIsInstance(result, tuple)

    def test_weighted_result_valid_signal(self):
        """Garante que o modo ponderado retorna apenas BUY, SELL ou None."""
        prices = _uptrend(300, step=0.10)
        with _no_ai(), patch.object(strategy, "USE_WEIGHTED_SIGNAL", True):
            sig, _ = strategy.get_signal(prices, adx_min=0)
        self.assertIn(sig, ("BUY", "SELL", None))


class TestADXHistory(unittest.TestCase):
    """Verifica o filtro que usa o histórico de ADX."""

    def test_falling_adx_blocks_signal(self):
        """ADX caindo → sinal bloqueado por filtro de tendência perdendo força."""
        prices = _uptrend(300, step=0.15)
        # Histórico de ADX muito alto para que o valor atual pareça cair
        adx_history = [1000.0] * 15
        with _no_ai(), _no_weighted(), patch.object(config, "ENTRY_MODE", "balanced", create=True):
            sig, ind = strategy.get_signal(prices, adx_min=0, adx_history=adx_history)
        self.assertIsNone(sig)
        self.assertTrue(ind.get("adx_falling", False))


if __name__ == "__main__":
    unittest.main()
