"""
tests/test_executor_trade_flow.py — Testes do fluxo de entrada e fechamento
de operações do DerivBot (sem conexão WebSocket real).
"""
import os
import sys
import json
import time
import unittest
from unittest.mock import MagicMock, patch, call
from collections import deque

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("DERIV_TOKEN", "test_token")

from executor import DerivBot
from risk_manager import RiskManager


# ─────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────

def _make_bot() -> DerivBot:
    rm = RiskManager(initial_balance=1000.0)
    bot = DerivBot(risk_manager=rm, demo=True)
    return bot


def _proposal_msg(proposal_id: str = "prop_001") -> dict:
    return {
        "msg_type": "proposal",
        "proposal": {"id": proposal_id},
    }


def _buy_msg(contract_id: int = 12345) -> dict:
    return {
        "msg_type": "buy",
        "buy": {"contract_id": contract_id},
    }


def _contract_update_msg(contract_id: int = 12345, profit: float = 1.50, is_sold: bool = True) -> dict:
    return {
        "msg_type": "proposal_open_contract",
        "proposal_open_contract": {
            "contract_id": contract_id,
            "is_sold": 1 if is_sold else 0,
            "profit": profit,
        },
    }


# ─────────────────────────────────────────────────────────────────
#  Testes — Envio de proposal (entrada)
# ─────────────────────────────────────────────────────────────────

class TestSendProposal(unittest.TestCase):
    """Verifica que _send_proposal monta e envia a proposal corretamente."""

    def setUp(self):
        self.bot = _make_bot()
        self.ws = MagicMock()
        # Preenche candles mínimos
        self.bot._candle_closes = deque([100.0 + i * 0.01 for i in range(100)], maxlen=500)

    @patch("ai_predictor.predict_duration", return_value=5)
    def test_proposal_enviada_para_buy(self, _mock_dur):
        self.bot._send_proposal(self.ws, "BUY", {"adx": 30.0, "rsi": 60.0})

        self.assertTrue(self.ws.send.called, "ws.send não foi chamado após BUY")
        payload = json.loads(self.ws.send.call_args[0][0])
        self.assertEqual(payload["contract_type"], "CALL")
        self.assertEqual(payload["proposal"], 1)
        self.assertEqual(self.bot._pending_direction, "BUY")

    @patch("ai_predictor.predict_duration", return_value=5)
    def test_proposal_enviada_para_sell(self, _mock_dur):
        self.bot._send_proposal(self.ws, "SELL", {"adx": 30.0, "rsi": 40.0})

        payload = json.loads(self.ws.send.call_args[0][0])
        self.assertEqual(payload["contract_type"], "PUT")
        self.assertEqual(self.bot._pending_direction, "SELL")

    @patch("ai_predictor.predict_duration", return_value=5)
    def test_pending_timestamp_marcado(self, _mock_dur):
        before = time.time()
        self.bot._send_proposal(self.ws, "BUY", {})
        self.assertGreaterEqual(self.bot._pending_timestamp, before)


# ─────────────────────────────────────────────────────────────────
#  Testes — Confirmação de compra (_handle_buy)
# ─────────────────────────────────────────────────────────────────

class TestHandleBuy(unittest.TestCase):
    """Verifica que _handle_buy registra o contrato aberto."""

    def setUp(self):
        self.bot = _make_bot()
        self.ws = MagicMock()

    @patch("executor.DerivBot._set_active_contract_state")
    def test_contrato_registrado(self, _mock_state):
        # _in_trade é marcado em _handle_proposal; aqui verificamos apenas o contract_id e timestamps
        self.bot._handle_buy(self.ws, {"contract_id": 99999})

        self.assertEqual(self.bot._open_contract_id, "99999")
        self.assertGreater(self.bot._buy_timestamp, 0)

    @patch("executor.DerivBot._set_active_contract_state")
    def test_subscribe_enviado_apos_buy(self, _mock_state):
        self.bot._handle_buy(self.ws, {"contract_id": 11111})

        calls_json = [json.loads(c[0][0]) for c in self.ws.send.call_args_list]
        subscribe_calls = [p for p in calls_json if "proposal_open_contract" in p]
        self.assertTrue(subscribe_calls, "Subscrição do contrato não foi enviada")
        self.assertEqual(subscribe_calls[0]["contract_id"], 11111)
        self.assertEqual(subscribe_calls[0]["subscribe"], 1)


# ─────────────────────────────────────────────────────────────────
#  Testes — Fechamento de contrato (_handle_contract_update)
# ─────────────────────────────────────────────────────────────────

class TestHandleContractUpdate(unittest.TestCase):
    """Verifica que _handle_contract_update liquida o contrato corretamente."""

    def _open_trade(self, bot: DerivBot, contract_id: str = "12345") -> None:
        """Simula estado de trade aberto sem passar pelo WebSocket."""
        bot._in_trade = True
        bot._open_contract_id = contract_id
        bot._pending_direction = "BUY"
        bot._pending_stake = 10.0
        bot._pending_duration = 5
        bot._pending_indicators = {}
        bot._buy_timestamp = time.time()

    @patch("executor.DerivBot._set_active_contract_state")
    def test_trade_fechado_com_lucro(self, _mock_state):
        bot = _make_bot()
        self._open_trade(bot)

        msg = _contract_update_msg(contract_id=12345, profit=2.5)
        bot._handle_contract_update(msg["proposal_open_contract"])

        self.assertFalse(bot._in_trade)
        self.assertIsNone(bot._open_contract_id)
        self.assertEqual(bot._buy_timestamp, 0.0)

    @patch("executor.DerivBot._set_active_contract_state")
    def test_trade_fechado_com_prejuizo(self, _mock_state):
        bot = _make_bot()
        self._open_trade(bot)

        msg = _contract_update_msg(contract_id=12345, profit=-10.0)
        bot._handle_contract_update(msg["proposal_open_contract"])

        self.assertFalse(bot._in_trade)

    @patch("executor.DerivBot._set_active_contract_state")
    def test_atualiza_risk_manager(self, _mock_state):
        bot = _make_bot()
        self._open_trade(bot)
        bot.risk_manager.record_result = MagicMock()

        msg = _contract_update_msg(contract_id=12345, profit=3.0)
        bot._handle_contract_update(msg["proposal_open_contract"])

        bot.risk_manager.record_result.assert_called_once()
        kwargs = bot.risk_manager.record_result.call_args[1]
        self.assertEqual(kwargs["profit"], 3.0)

    @patch("executor.DerivBot._set_active_contract_state")
    def test_ignora_update_de_outro_contrato(self, _mock_state):
        """Atualizações de contratos diferentes do aberto devem ser ignoradas."""
        bot = _make_bot()
        self._open_trade(bot, contract_id="12345")
        bot.risk_manager.record_result = MagicMock()

        outro_msg = _contract_update_msg(contract_id=99999, profit=5.0)
        bot._handle_contract_update(outro_msg["proposal_open_contract"])

        bot.risk_manager.record_result.assert_not_called()
        self.assertTrue(bot._in_trade, "Trade foi fechado por contrato errado")

    @patch("executor.DerivBot._set_active_contract_state")
    def test_contrato_aberto_nao_vendido_ignorado(self, _mock_state):
        """Se is_sold=0, contrato ainda está aberto — não deve liquidar."""
        bot = _make_bot()
        self._open_trade(bot)
        bot.risk_manager.record_result = MagicMock()

        msg = _contract_update_msg(contract_id=12345, profit=0.0, is_sold=False)
        bot._handle_contract_update(msg["proposal_open_contract"])

        self.assertTrue(bot._in_trade)
        bot.risk_manager.record_result.assert_not_called()


# ─────────────────────────────────────────────────────────────────
#  Testes — Fluxo completo (proposta → buy → fechamento)
# ─────────────────────────────────────────────────────────────────

class TestFullTradeFlow(unittest.TestCase):
    """Simula o fluxo completo de uma operação sem WebSocket real."""

    def setUp(self):
        self.bot = _make_bot()
        self.ws = MagicMock()
        self.bot._candle_closes = deque([100.0 + i * 0.01 for i in range(100)], maxlen=500)
        self.bot._last_tick_price = 100.5
        self.bot._last_tick_epoch = int(time.time())

    @patch("executor.DerivBot._set_active_contract_state")
    @patch("ai_predictor.predict_duration", return_value=5)
    def test_fluxo_buy_completo(self, _mock_dur, _mock_state):
        # 1) Sinal → proposal
        self.bot._send_proposal(self.ws, "BUY", {"adx": 28.0, "rsi": 62.0})
        self.assertEqual(self.bot._pending_direction, "BUY")

        # 2) proposal confirmada → _handle_proposal → ws.send(buy)
        self.bot._handle_proposal(self.ws, {"id": "prop_x"})
        buy_calls = [json.loads(c[0][0]) for c in self.ws.send.call_args_list]
        buy_req = next((p for p in buy_calls if "buy" in p), None)
        self.assertIsNotNone(buy_req, "Mensagem de buy não enviada")

        # 3) API confirma compra
        self.bot._handle_buy(self.ws, {"contract_id": 555})
        self.assertTrue(self.bot._in_trade)
        self.assertEqual(self.bot._open_contract_id, "555")

        # 4) Contrato fecha com lucro
        self.bot.risk_manager.record_result = MagicMock()
        self.bot._handle_contract_update({
            "contract_id": 555,
            "is_sold": 1,
            "profit": 4.50,
        })
        self.assertFalse(self.bot._in_trade)
        self.bot.risk_manager.record_result.assert_called_once()

    @patch("executor.DerivBot._set_active_contract_state")
    @patch("ai_predictor.predict_duration", return_value=5)
    def test_fluxo_sell_completo(self, _mock_dur, _mock_state):
        self.bot._send_proposal(self.ws, "SELL", {"adx": 35.0, "rsi": 38.0})
        self.assertEqual(self.bot._pending_direction, "SELL")

        self.bot._handle_proposal(self.ws, {"id": "prop_y"})
        self.bot._handle_buy(self.ws, {"contract_id": 777})
        self.assertTrue(self.bot._in_trade)

        self.bot.risk_manager.record_result = MagicMock()
        self.bot._handle_contract_update({
            "contract_id": 777,
            "is_sold": 1,
            "profit": -9.0,
        })
        self.assertFalse(self.bot._in_trade)
        call_kwargs = self.bot.risk_manager.record_result.call_args[1]
        self.assertEqual(call_kwargs["profit"], -9.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
