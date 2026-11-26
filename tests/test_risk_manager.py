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
    """Context manager que cria um RiskManager com log de operações em diretório temporário."""
    def __init__(self, balance: float = 1000.0):
        """Guarda o saldo inicial a ser usado pelo RiskManager."""
        self._balance = balance

    def __enter__(self) -> "RiskManager":
        """Redireciona OPERATIONS_LOG para um arquivo temporário e retorna o RiskManager criado."""
        self._tmpdir = tempfile.mkdtemp()
        self._log = os.path.join(self._tmpdir, "ops.csv")
        self._patch = patch.object(rm_mod, "OPERATIONS_LOG", self._log)
        self._patch.start()
        self.rm = RiskManager(self._balance)
        return self.rm

    def __exit__(self, *_):
        """Restaura o OPERATIONS_LOG original."""
        self._patch.stop()


def _loss(rm, profit=-1.0):
    """Registra uma operação perdedora no RiskManager."""
    rm.record_result("R_100", "BUY", 1.0, 5, profit, {})


def _win(rm, profit=1.0):
    """Registra uma operação vencedora no RiskManager."""
    rm.record_result("R_100", "BUY", 1.0, 5, profit, {})


class TestGetStake(unittest.TestCase):
    """Verifica o cálculo do valor de entrada."""
    def test_stake_one_percent(self):
        """Garante que o stake é 1% do saldo."""
        with _TmpLog(1000.0) as rm:
            self.assertAlmostEqual(rm.get_stake(), 10.0, places=2)

    def test_stake_minimum_035(self):
        """Garante que o stake nunca fica abaixo de 0.35."""
        with _TmpLog(20.0) as rm:
            self.assertGreaterEqual(rm.get_stake(), 0.35)

    def test_stake_proportional(self):
        """Garante que o stake acompanha proporcionalmente o saldo."""
        with _TmpLog(500.0) as rm:
            self.assertAlmostEqual(rm.get_stake(), 5.0, places=2)


class TestCanTrade(unittest.TestCase):
    """Verifica as condições que liberam ou bloqueiam novas operações."""
    def test_can_trade_by_default(self):
        """Garante que um RiskManager novo permite operar."""
        with _TmpLog() as rm:
            self.assertTrue(rm.can_trade())

    def test_stops_on_daily_stop_loss(self):
        """Garante que can_trade bloqueia ao atingir o stop loss diário."""
        with _TmpLog(1000.0) as rm:
            rm._daily_profit = -260.0
            self.assertFalse(rm.can_trade())

    def test_stops_on_daily_take_profit(self):
        """Garante que can_trade bloqueia ao atingir o take profit diário."""
        with _TmpLog(1000.0) as rm:
            rm._daily_profit = 510.0
            self.assertFalse(rm.can_trade())

    def test_allows_below_stop(self):
        """Garante que can_trade permite operar com perda diária abaixo do stop."""
        with _TmpLog(1000.0) as rm:
            rm._daily_profit = -240.0
            self.assertTrue(rm.can_trade())

    def test_paused_after_max_consec_losses(self):
        """Garante que can_trade bloqueia após MAX_CONSEC_LOSSES perdas seguidas."""
        with _TmpLog() as rm:
            for _ in range(config.MAX_CONSEC_LOSSES):
                _loss(rm)
            self.assertFalse(rm.can_trade())

    def test_not_paused_before_max_losses(self):
        """Garante que can_trade segue liberado com uma perda a menos que o limite."""
        with _TmpLog() as rm:
            for _ in range(config.MAX_CONSEC_LOSSES - 1):
                _loss(rm)
            self.assertTrue(rm.can_trade())


class TestPauseScaling(unittest.TestCase):
    """Verifica a duração e a limpeza da pausa por perdas consecutivas."""
    def test_pause_on_max_losses(self):
        """Garante que o limite de perdas aplica pausa de ao menos PAUSE_BASE_SEC."""
        with _TmpLog() as rm:
            for _ in range(config.MAX_CONSEC_LOSSES):
                _loss(rm)
            remaining = rm._pause_until - time.time()
            self.assertGreater(remaining, config.PAUSE_BASE_SEC - 5)

    def test_pause_scales_on_extra_loss(self):
        """Garante que uma perda extra escala a pausa por PAUSE_SCALE_FACTOR."""
        with _TmpLog() as rm:
            for _ in range(config.MAX_CONSEC_LOSSES + 1):
                _loss(rm)
            expected = config.PAUSE_BASE_SEC * config.PAUSE_SCALE_FACTOR
            remaining = rm._pause_until - time.time()
            self.assertGreater(remaining, expected - 5)

    def test_win_clears_pause(self):
        """Garante que uma vitória encerra a pausa quando RESUME_ON_WIN está ativo."""
        if not config.RESUME_ON_WIN:
            self.skipTest("RESUME_ON_WIN desativado")
        with _TmpLog() as rm:
            for _ in range(config.MAX_CONSEC_LOSSES):
                _loss(rm)
            self.assertTrue(rm.is_paused())
            _win(rm)
            self.assertEqual(rm._pause_until, 0.0)


class TestRecordResult(unittest.TestCase):
    """Verifica o efeito de record_result sobre saldo, contadores e log."""
    def test_balance_increases_on_win(self):
        """Garante que o saldo aumenta com o lucro de uma vitória."""
        with _TmpLog(1000.0) as rm:
            _win(rm, 5.0)
            self.assertAlmostEqual(rm.balance, 1005.0, places=2)

    def test_balance_decreases_on_loss(self):
        """Garante que o saldo diminui com o prejuízo de uma derrota."""
        with _TmpLog(1000.0) as rm:
            _loss(rm, -5.0)
            self.assertAlmostEqual(rm.balance, 995.0, places=2)

    def test_consec_losses_resets_on_win(self):
        """Garante que uma vitória zera o contador de perdas consecutivas."""
        with _TmpLog() as rm:
            _loss(rm)
            _loss(rm)
            _win(rm)
            self.assertEqual(rm._consec_losses, 0)

    def test_consec_losses_increment(self):
        """Garante que cada derrota incrementa o contador de perdas consecutivas."""
        with _TmpLog() as rm:
            _loss(rm)
            _loss(rm)
            self.assertEqual(rm._consec_losses, 2)

    def test_csv_written(self):
        """Garante que record_result grava linhas no CSV de operações."""
        tmpdir = tempfile.mkdtemp()
        log_path = os.path.join(tmpdir, "ops.csv")
        with patch.object(rm_mod, "OPERATIONS_LOG", log_path):
            rm = RiskManager(1000.0)
            _win(rm, 1.0)
        with open(log_path) as f:
            lines = f.readlines()
        self.assertGreater(len(lines), 1)

    def test_daily_profit_tracked(self):
        """Garante que o lucro diário acumula os resultados registrados."""
        with _TmpLog(1000.0) as rm:
            _win(rm, 5.0)
            _win(rm, 3.0)
            self.assertAlmostEqual(rm._daily_profit, 8.0, places=2)


class TestStateDict(unittest.TestCase):
    """Verifica o dicionário de estado exportado por to_state_dict."""
    def test_has_required_keys(self):
        """Garante que to_state_dict contém todas as chaves obrigatórias."""
        with _TmpLog() as rm:
            d = rm.to_state_dict()
        for key in ("is_paused", "daily_pnl", "consec_losses", "win_rate_recent",
                    "drift_detected", "date", "balance", "recent_results"):
            self.assertIn(key, d)

    def test_is_paused_false_by_default(self):
        """Garante que is_paused é falso em um RiskManager novo."""
        with _TmpLog() as rm:
            self.assertFalse(rm.to_state_dict()["is_paused"])

    def test_balance_matches(self):
        """Garante que o saldo do estado coincide com o saldo inicial."""
        with _TmpLog(850.0) as rm:
            self.assertAlmostEqual(rm.to_state_dict()["balance"], 850.0, places=2)


class TestPersistence(unittest.TestCase):
    """Verifica a persistência do estado entre instâncias do RiskManager."""
    def test_state_saved_and_restored(self):
        """Garante que as perdas consecutivas são restauradas por uma nova instância no mesmo dia."""
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
        """Garante que um estado salvo em data anterior não é restaurado."""
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
    """Verifica a detecção de drift pelo desempenho recente."""
    def test_no_drift_before_window(self):
        """Garante que não há drift detectado antes de haver resultados na janela."""
        with _TmpLog() as rm:
            self.assertFalse(rm.to_state_dict()["drift_detected"])

    def test_drift_after_all_losses(self):
        """Garante que uma janela inteira de derrotas marca drift detectado."""
        with _TmpLog(10000.0) as rm:
            for _ in range(config.DRIFT_WINDOW):
                rm.record_result("R_100", "BUY", 0.35, 5, -0.01, {})
        self.assertTrue(rm.to_state_dict()["drift_detected"])


if __name__ == "__main__":
    unittest.main()
