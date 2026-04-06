"""
tests/test_risk_manager.py — testes unitários para o RiskManager.
"""

import os
import sys
import json
import time
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("DERIV_TOKEN", "test_token")

import risk_manager as rm_mod
from risk_manager import RiskManager
import config


class _TmpLog:
    def __init__(self, balance: float = 1000.0):
        self._balance = balance

    def __enter__(self) -> "RiskManager":
        self._tmpdir = tempfile.mkdtemp()
        self._log = os.path.join(self._tmpdir, "ops.csv")
        self._patch = patch.object(rm_mod, "OPERATIONS_LOG", self._log)
        self._patch.start()
        self.rm = RiskManager(self._balance)
        return self.rm

    def __exit__(self, *_):
        self._patch.stop()


def _loss(rm, profit=-1.0):
    rm.record_result("R_100", "BUY", 1.0, 5, profit, {})


def _win(rm, profit=1.0):
    rm.record_result("R_100", "BUY", 1.0, 5, profit, {})


class TestGetStake(unittest.TestCase):
    def test_stake_one_percent(self):
        with _TmpLog(1000.0) as rm:
            self.assertAlmostEqual(rm.get_stake(), 10.0, places=2)

    def test_stake_minimum_035(self):
        with _TmpLog(20.0) as rm:
            self.assertGreaterEqual(rm.get_stake(), 0.35)

    def test_stake_proportional(self):
        with _TmpLog(500.0) as rm:
            self.assertAlmostEqual(rm.get_stake(), 5.0, places=2)


class TestCanTrade(unittest.TestCase):
    def test_can_trade_by_default(self):
        with _TmpLog() as rm:
            self.assertTrue(rm.can_trade())

    def test_stops_on_daily_stop_loss(self):
        with _TmpLog(1000.0) as rm:
            rm._daily_profit = -260.0
            self.assertFalse(rm.can_trade())

    def test_stops_on_daily_take_profit(self):
        with _TmpLog(1000.0) as rm:
            rm._daily_profit = 510.0
            self.assertFalse(rm.can_trade())

    def test_allows_below_stop(self):
        with _TmpLog(1000.0) as rm:
            rm._daily_profit = -240.0
            self.assertTrue(rm.can_trade())

    def test_paused_after_max_consec_losses(self):
        with _TmpLog() as rm:
            for _ in range(config.MAX_CONSEC_LOSSES):
                _loss(rm)
            self.assertFalse(rm.can_trade())

    def test_not_paused_before_max_losses(self):
        with _TmpLog() as rm:
            for _ in range(config.MAX_CONSEC_LOSSES - 1):
                _loss(rm)
            self.assertTrue(rm.can_trade())


class TestPauseScaling(unittest.TestCase):
    def test_pause_on_max_losses(self):
        with _TmpLog() as rm:
            for _ in range(config.MAX_CONSEC_LOSSES):
                _loss(rm)
            remaining = rm._pause_until - time.time()
            self.assertGreater(remaining, config.PAUSE_BASE_SEC - 5)

    def test_pause_scales_on_extra_loss(self):
        with _TmpLog() as rm:
            for _ in range(config.MAX_CONSEC_LOSSES + 1):
                _loss(rm)
            expected = config.PAUSE_BASE_SEC * config.PAUSE_SCALE_FACTOR
            remaining = rm._pause_until - time.time()
            self.assertGreater(remaining, expected - 5)

    def test_win_clears_pause(self):
        if not config.RESUME_ON_WIN:
            self.skipTest("RESUME_ON_WIN desativado")
        with _TmpLog() as rm:
            for _ in range(config.MAX_CONSEC_LOSSES):
                _loss(rm)
            self.assertTrue(rm.is_paused())
            _win(rm)
            self.assertEqual(rm._pause_until, 0.0)


class TestRecordResult(unittest.TestCase):
    def test_balance_increases_on_win(self):
        with _TmpLog(1000.0) as rm:
            _win(rm, 5.0)
            self.assertAlmostEqual(rm.balance, 1005.0, places=2)

    def test_balance_decreases_on_loss(self):
        with _TmpLog(1000.0) as rm:
            _loss(rm, -5.0)
            self.assertAlmostEqual(rm.balance, 995.0, places=2)

    def test_consec_losses_resets_on_win(self):
        with _TmpLog() as rm:
            _loss(rm)
            _loss(rm)
            _win(rm)
            self.assertEqual(rm._consec_losses, 0)

    def test_consec_losses_increment(self):
        with _TmpLog() as rm:
            _loss(rm)
            _loss(rm)
            self.assertEqual(rm._consec_losses, 2)

    def test_csv_written(self):
        tmpdir = tempfile.mkdtemp()
        log_path = os.path.join(tmpdir, "ops.csv")
        with patch.object(rm_mod, "OPERATIONS_LOG", log_path):
            rm = RiskManager(1000.0)
            _win(rm, 1.0)
        with open(log_path) as f:
            lines = f.readlines()
        self.assertGreater(len(lines), 1)

    def test_daily_profit_tracked(self):
        with _TmpLog(1000.0) as rm:
            _win(rm, 5.0)
            _win(rm, 3.0)
            self.assertAlmostEqual(rm._daily_profit, 8.0, places=2)


class TestStateDict(unittest.TestCase):
    def test_has_required_keys(self):
        with _TmpLog() as rm:
            d = rm.to_state_dict()
        for key in ("is_paused", "daily_pnl", "consec_losses", "win_rate_recent",
                    "drift_detected", "date", "balance", "recent_results"):
            self.assertIn(key, d)

    def test_is_paused_false_by_default(self):
        with _TmpLog() as rm:
            self.assertFalse(rm.to_state_dict()["is_paused"])

    def test_balance_matches(self):
        with _TmpLog(850.0) as rm:
            self.assertAlmostEqual(rm.to_state_dict()["balance"], 850.0, places=2)


class TestPersistence(unittest.TestCase):
    def test_state_saved_and_restored(self):
        tmpdir = tempfile.mkdtemp()
        log_path = os.path.join(tmpdir, "ops.csv")
        with patch.object(rm_mod, "OPERATIONS_LOG", log_path):
            rm1 = RiskManager(1000.0)
            _loss(rm1)
            _loss(rm1)
            consec_before = rm1._consec_losses
        with patch.object(rm_mod, "OPERATIONS_LOG", log_path):
            rm2 = RiskManager(1000.0)
        self.assertEqual(rm2._consec_losses, consec_before)

    def test_state_not_restored_next_day(self):
        tmpdir = tempfile.mkdtemp()
        log_path = os.path.join(tmpdir, "ops.csv")
        state_path = os.path.join(tmpdir, "risk_state.json")
        fake_state = {
            "date": "2000-01-01",
            "consec_losses": 5,
            "balance": 2000.0,
            "pause_until_epoch": 0.0,
            "recent_results": [],
            "daily_pnl": -100.0,
        }
        with open(state_path, "w") as f:
            json.dump(fake_state, f)
        with patch.object(rm_mod, "OPERATIONS_LOG", log_path):
            rm = RiskManager(1000.0)
        self.assertEqual(rm._consec_losses, 0)


class TestDriftDetection(unittest.TestCase):
    def test_no_drift_before_window(self):
        with _TmpLog() as rm:
            self.assertFalse(rm.to_state_dict()["drift_detected"])

    def test_drift_after_all_losses(self):
        with _TmpLog(10000.0) as rm:
            for _ in range(config.DRIFT_WINDOW):
                rm.record_result("R_100", "BUY", 0.35, 5, -0.01, {})
        self.assertTrue(rm.to_state_dict()["drift_detected"])


if __name__ == "__main__":
    unittest.main()
