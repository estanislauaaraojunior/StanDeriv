"""
tests/test_backtester.py — testes do motor de backtesting offline (backtester.py).
"""

import csv
import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import backtester
from backtester import (
    BacktestResult, Trade, _duration_to_candles, _load_candles, run_backtest,
)


def _write_synthetic_ticks(path: str, n_candles: int = 250, ticks_per_candle: int = 6,
                            symbol: str = "R_TEST", trend: float = 0.0006,
                            noise: float = 0.0004, seed: int = 42) -> None:
    """Gera um ticks.csv sintético com leve tendência de alta + ruído determinístico."""
    import random
    rnd = random.Random(seed)
    price = 100.0
    epoch = 1_700_000_000
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "datetime", "symbol", "price"])
        for _c in range(n_candles):
            for _t in range(ticks_per_candle):
                price *= (1 + trend + rnd.uniform(-noise, noise))
                writer.writerow([epoch, "2024-01-01T00:00:00", symbol, round(price, 5)])
                epoch += 10  # ~10s por tick, 6 ticks/min = velas de 60s


class TestDurationConversion(unittest.TestCase):
    def test_minutes_to_candles(self):
        self.assertEqual(_duration_to_candles(15, "m"), 15)  # candle de 60s = 1min

    def test_seconds_to_candles(self):
        self.assertEqual(_duration_to_candles(120, "s"), 2)

    def test_ticks_fallback_is_one_to_one(self):
        self.assertEqual(_duration_to_candles(5, "t"), 5)

    def test_minimum_one_candle(self):
        self.assertEqual(_duration_to_candles(0, "s"), 1)


class TestLoadCandles(unittest.TestCase):
    def test_aggregates_ticks_into_candles(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ticks.csv")
            _write_synthetic_ticks(path, n_candles=50, ticks_per_candle=6)
            closes, epochs, symbol = _load_candles(path, symbol="R_TEST")
            self.assertEqual(symbol, "R_TEST")
            self.assertGreater(len(closes), 0)
            self.assertEqual(len(closes), len(epochs))

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            _load_candles("/tmp/__does_not_exist__.csv")


class TestRunBacktest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ticks_path = os.path.join(self._tmp.name, "ticks.csv")
        _write_synthetic_ticks(self.ticks_path, n_candles=400, ticks_per_candle=6)

    def tearDown(self):
        self._tmp.cleanup()

    def test_returns_backtest_result(self):
        result = run_backtest(
            self.ticks_path, initial_balance=1000.0, symbol="R_TEST", use_ai_model=False,
        )
        self.assertIsInstance(result, BacktestResult)
        self.assertEqual(result.symbol, "R_TEST")
        self.assertEqual(result.initial_balance, 1000.0)
        self.assertGreaterEqual(result.total_trades, 0)

    def test_metrics_are_consistent(self):
        result = run_backtest(
            self.ticks_path, initial_balance=1000.0, symbol="R_TEST", use_ai_model=False,
        )
        self.assertEqual(result.wins + result.losses, result.total_trades)
        if result.total_trades:
            expected_wr = round(result.wins / result.total_trades * 100, 2)
            self.assertAlmostEqual(result.win_rate_pct, expected_wr, places=2)
        expected_return = round(
            (result.final_balance - result.initial_balance) / result.initial_balance * 100, 2,
        )
        self.assertAlmostEqual(result.total_return_pct, expected_return, places=2)
        self.assertGreaterEqual(result.max_drawdown_pct, 0.0)

    def test_does_not_touch_live_risk_files(self):
        """O backtest não deve escrever no operations_log/risk_state ao vivo."""
        import risk_manager as rm_mod
        original_log = rm_mod.OPERATIONS_LOG
        run_backtest(
            self.ticks_path, initial_balance=1000.0, symbol="R_TEST", use_ai_model=False,
        )
        # OPERATIONS_LOG deve ser restaurado ao valor original após o backtest
        self.assertEqual(rm_mod.OPERATIONS_LOG, original_log)

    def test_restores_use_ai_model_flag(self):
        import strategy as strat_mod
        original = strat_mod.USE_AI_MODEL
        run_backtest(
            self.ticks_path, initial_balance=1000.0, symbol="R_TEST", use_ai_model=not original,
        )
        self.assertEqual(strat_mod.USE_AI_MODEL, original)

    def test_insufficient_candles_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            small_path = os.path.join(tmp, "ticks.csv")
            _write_synthetic_ticks(small_path, n_candles=5, ticks_per_candle=6)
            with self.assertRaises(ValueError):
                run_backtest(small_path, initial_balance=1000.0, symbol="R_TEST")

    def test_to_dict_serializable(self):
        import json
        result = run_backtest(
            self.ticks_path, initial_balance=1000.0, symbol="R_TEST", use_ai_model=False,
        )
        payload = result.to_dict()
        json.dumps(payload)  # não deve levantar exceção
        self.assertIn("win_rate_pct", payload)
        self.assertIn("trades", payload)


class TestCompareAiFilter(unittest.TestCase):
    def test_compare_returns_both_variants(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ticks.csv")
            _write_synthetic_ticks(path, n_candles=300, ticks_per_candle=6)
            results = backtester.compare_ai_filter(path, initial_balance=1000.0, symbol="R_TEST")
            self.assertIn("without_ai_model", results)
            self.assertIn("with_ai_model", results)
            self.assertFalse(results["without_ai_model"].use_ai_model)
            self.assertTrue(results["with_ai_model"].use_ai_model)


if __name__ == "__main__":
    unittest.main()
