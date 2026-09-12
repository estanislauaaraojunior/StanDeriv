"""Ordens com requisições correlacionadas e liquidação idempotente, offline."""
import csv
import json
import time
from unittest.mock import MagicMock

import pytest
import config
import executor
from executor import DerivBot
from risk_manager import RiskManager


@pytest.fixture
def bot():
    risk = RiskManager(1000.)
    risk.bind_account('test-account', 1000.)
    b = DerivBot(risk)
    b._auth_verified = True
    b._last_tick_epoch = int(time.time())
    b._last_tick_price = 100.
    return b


def proposal(bot, ws, direction = 'BUY'):
    bot._send_proposal(ws, direction, {'win_probability': .8, 'ai_confidence': .8, 'model_id': 'offline'})
    return bot._proposal_req_id


def buy(bot, ws, direction = 'BUY'):
    rid = proposal(bot, ws, direction)
    bot._handle_proposal(ws, {'id': 'proposal-test', 'ask_price': '10', 'payout': '18.5'}, req_id = rid)
    bot._handle_buy(ws, {'contract_id': 12345}, req_id = bot._buy_req_id)


@pytest.mark.parametrize('direction,contract_type', [('BUY', 'CALL'), ('SELL', 'PUT')])
def test_request_uses_same_symbol_duration_and_direction(bot, direction, contract_type):
    ws = MagicMock()
    proposal(bot, ws, direction)
    sent = json.loads(ws.send.call_args.args[0])
    assert sent['contract_type'] == contract_type
    assert sent['symbol'] == bot.symbol
    assert sent['duration'] == config.DURATION == 15
    assert sent['duration_unit'] == config.DURATION_UNIT
    assert bot._in_trade and bot._phase == 'proposal'
    assert sent['req_id'] == bot._proposal_req_id


def test_forex_symbol_can_open_contract_after_valid_signal(bot):
    bot.symbol = 'frxEURAUD'
    ws = MagicMock()
    rid = proposal(bot, ws, 'SELL')
    proposed = json.loads(ws.send.call_args.args[0])
    assert proposed['symbol'] == 'frxEURAUD'
    assert proposed['contract_type'] == 'PUT'
    bot._handle_proposal(ws, {'id': 'proposal-forex', 'ask_price': '10', 'payout': '18.5'}, req_id=rid)
    bot._handle_buy(ws, {'contract_id': 67890}, req_id=bot._buy_req_id)
    assert bot._open_contract_id == '67890'
    assert bot._phase == 'open'


def test_no_second_proposal_while_first_pending(bot):
    ws = MagicMock()
    proposal(bot, ws)
    proposal(bot, ws)
    assert ws.send.call_count == 1


@pytest.mark.parametrize('rid', [None, - 1])
def test_unsolicited_proposal_is_ignored(bot, rid):
    ws = MagicMock()
    proposal(bot, ws)
    ws.reset_mock()
    bot._handle_proposal(ws, {'id': 'other', 'ask_price': 10, 'payout': 18.5}, req_id = rid)
    ws.send.assert_not_called()


def test_duplicate_proposal_cannot_buy_twice(bot):
    ws = MagicMock()
    rid = proposal(bot, ws)
    ws.reset_mock()
    p = {'id': 'p', 'ask_price': 10, 'payout': 18.5}
    bot._handle_proposal(ws, p, req_id = rid)
    bot._handle_proposal(ws, p, req_id = rid)
    assert ws.send.call_count == 1
    assert bot._phase == 'buy'


@pytest.mark.parametrize('price,payout', [(10, 10.01), (10, None), (10, float('nan')), (11, 25), (0, 20)])
def test_invalid_or_unprofitable_proposal_not_bought(bot, price, payout):
    ws = MagicMock()
    rid = proposal(bot, ws)
    ws.reset_mock()
    bot._handle_proposal(ws, {'id': 'p', 'ask_price': price, 'payout': payout}, req_id = rid)
    ws.send.assert_not_called()
    assert not bot._in_trade


def test_expired_proposal_is_rejected(bot):
    ws = MagicMock()
    rid = proposal(bot, ws)
    ws.reset_mock()
    bot._pending_timestamp = time.time() - 30
    bot._handle_proposal(ws, {'id': 'p', 'ask_price': 10, 'payout': 18.5}, req_id = rid)
    ws.send.assert_not_called()
    assert bot._phase == 'idle'


@pytest.mark.parametrize('probability,payout,accepted', [
    (.56, 18.5, True),   # Ganho esperado positivo.
    (.60, 16.8, True),   # 0,8% positivo, abaixo da margem antiga de 2%.
    (.625, 16.0, True),  # Equilíbrio exato.
    (.6249, 16.0, False),
    (.56, 17.0, False),  # Confiança >55% não basta para este payout.
])
def test_buy_requires_nonnegative_expected_return(bot, probability, payout, accepted):
    ws = MagicMock()
    bot._send_proposal(ws, 'BUY', {'win_probability': probability,
                                 'ai_confidence': probability, 'model_id': 'offline'})
    rid = bot._proposal_req_id
    ws.reset_mock()
    bot._handle_proposal(ws, {'id': 'p', 'ask_price': 10, 'payout': payout}, req_id=rid)
    assert (bot._phase == 'buy') is accepted
    assert ws.send.call_count == int(accepted)


@pytest.mark.parametrize('probability,accepted', [(.55, False), (.550001, True)])
def test_confidence_must_be_strictly_above_fifty_five(bot, probability, accepted):
    from model_contract import profitable_proposal
    ws = MagicMock()
    bot._send_proposal(ws, 'BUY', {'win_probability': probability,
                                 'ai_confidence': probability, 'model_id': 'offline'})
    assert ws.send.call_count == int(accepted)
    assert profitable_proposal(10, 20, probability) is accepted


def test_delayed_buy_response_reconciles_only_matching_request(bot):
    ws = MagicMock()
    rid = proposal(bot, ws)
    bot._handle_proposal(ws, {'id': 'p', 'ask_price': 10, 'payout': 18.5}, req_id = rid)
    bot._phase = 'reconciling'
    bot._handle_buy(ws, {'contract_id': 555}, req_id = - 1)
    assert bot._open_contract_id is None
    bot._handle_buy(ws, {'contract_id': 12345}, req_id = bot._buy_req_id)
    assert bot._open_contract_id == '12345'
    assert bot._entry_price == 0  # não inventa preço de entrada a partir do tick local


@pytest.mark.parametrize('profit,result', [(2.5, 'WIN'), (-10, 'LOSS'), (0, 'VOID')])
def test_settlement_updates_balance_once(bot, profit, result):
    ws = MagicMock()
    buy(bot, ws)
    event = {'contract_id': 12345, 'is_sold': 1, 'profit': str(profit), 'entry_spot': '100',
           'exit_spot': '101', 'date_start': 1000, 'date_expiry': 1900}
    bot._handle_contract_update(event)
    bot._handle_contract_update(event)
    assert bot.risk_manager.balance == 1000 + profit
    assert not bot._in_trade
    with (bot.risk_manager._path.parent / 'operations.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1 and rows[0]['result'] == result
    assert rows[0]['entry_price'] == '100' and rows[0]['contract_id'] == '12345'


@pytest.mark.parametrize('event', [
    {'contract_id': 99999, 'is_sold': 1, 'profit': 10},
    {'contract_id': 12345, 'is_sold': 0, 'profit': 10},
    {'contract_id': 12345, 'is_sold': 1},
    {'contract_id': 12345, 'is_sold': 1, 'profit': 'nan'},
])
def test_invalid_or_unrelated_settlement_not_counted(bot, event):
    ws = MagicMock()
    buy(bot, ws)
    bot._handle_contract_update(event)
    assert bot._in_trade and bot.risk_manager.balance == 1000


def test_restart_restores_open_contract_and_prevents_new_order(bot):
    ws = MagicMock()
    buy(bot, ws)
    restored = DerivBot(RiskManager(9000.))
    assert restored._open_contract_id == '12345'
    assert restored._pending_stake == 10 and restored._in_trade
    restored._send_proposal(ws, 'BUY', {'win_probability': .9})
    assert restored._phase == 'open'


def test_restart_preserves_uncertain_buy(bot):
    ws = MagicMock()
    rid = proposal(bot, ws)
    bot._handle_proposal(ws, {'id': 'p', 'ask_price': 10, 'payout': 18.5}, req_id = rid)
    restored = DerivBot(RiskManager(1000.))
    assert restored._in_trade and restored._phase == 'buy' and restored._open_contract_id is None


def test_wrong_account_mode_refused(bot):
    ws = MagicMock()
    with pytest.raises(ValueError):
        bot._handle_authorize(ws, {'is_virtual': 0, 'loginid': 'test', 'balance': 1000})
    ws.close.assert_called_once()
    assert not bot._auth_verified


def test_history_is_merged_without_duplicates(bot):
    candles = [dict(epoch = i * 60, open = 100 + i, high = 101 + i, low = 99 + i, close = 100 + i) for i in range(200)]
    bot._merge_candles(candles)
    bot._merge_candles(candles[-99:])
    assert len(bot._candles) == 200 and len(set(bot._candle_epochs)) == 200
    assert list(bot._candle_epochs) == sorted(bot._candle_epochs)


def test_wrong_symbol_tick_does_not_contaminate_buffer(bot):
    before = list(bot._candle_closes)
    bot._handle_tick(MagicMock(), {'symbol': 'wrong', 'epoch': int(time.time()) + 1, 'quote': 50})
    assert list(bot._candle_closes) == before and bot._last_tick_price == 100


def test_reconnect_balance_does_not_double_count_settlement(bot):
    ws = MagicMock()
    buy(bot, ws)
    # A API já inclui +8; estado local ainda aguarda a liquidação.
    bot._verify_account('test-account', 1008.)
    bot._handle_contract_update({'contract_id': 12345, 'is_sold': 1, 'profit': 8})
    assert bot.risk_manager.balance == 1008


def test_buy_error_does_not_authorize_retry(bot):
    ws = MagicMock()
    rid = proposal(bot, ws)
    bot._handle_proposal(ws, {'id': 'p', 'ask_price': 10, 'payout': 18.5}, req_id = rid)
    bot._on_message(ws, json.dumps({'error': {'code': 'InternalError'}, 'req_id': bot._buy_req_id}))
    assert bot._in_trade and bot._phase == 'reconciling'


def test_reconnection_refetches_bearer_account_balance(bot, monkeypatch):
    import hashlib
    bot.auth_mode = 'bearer'
    bot.account_id = 'virtual-test'
    bot.risk_manager.bind_account(hashlib.sha256(b'virtual-test').hexdigest(), 1000)
    bot.risk_manager.record_result('R_100', 'BUY', 10, 15, - 10, {})
    monkeypatch.setattr(executor.deriv_session, 'get_options_accounts', lambda ** k: [{'account_id': 'virtual-test', 'account_type': 'demo', 'balance': 990}])
    ws = MagicMock()
    bot._on_close(ws, 1000, '')
    bot._on_open(ws)
    assert bot.risk_manager.balance == 990 and bot.risk_manager._daily_profit == - 10


def test_expired_open_contract_remains_tracked_on_tick(bot, monkeypatch):
    ws = MagicMock()
    buy(bot, ws)
    bot._buy_timestamp = time.time() - 10000
    bot._contract_poll_timer = None
    monkeypatch.setattr(bot, '_poll_contract_result', MagicMock())
    bot._handle_tick(ws, {'symbol': bot.symbol, 'epoch': bot._last_tick_epoch + 1, 'quote': 100})
    assert bot._in_trade and bot._open_contract_id == '12345'
    bot._poll_contract_result.assert_called_once_with('12345')


def test_market_switch_blocks_new_proposals(bot):
    ws = MagicMock()
    bot.market_switch_requested = lambda: True
    proposal(bot, ws)
    ws.send.assert_not_called()
    assert bot._market_switch_ready()


def test_market_switch_cancels_unpurchased_proposal(bot):
    ws = MagicMock()
    rid = proposal(bot, ws)
    bot.market_switch_requested = lambda: True
    ws.reset_mock()
    bot._handle_proposal(ws, {'id': 'quote', 'ask_price': 10, 'payout': 18.5}, req_id=rid)
    ws.send.assert_not_called()
    assert bot._market_switch_ready()


def test_market_switch_preserves_open_contract_until_settlement(bot):
    ws = MagicMock()
    buy(bot, ws)
    bot.market_switch_requested = lambda: True
    assert not bot._market_switch_ready()
    bot._on_message(ws, json.dumps({'msg_type': 'proposal_open_contract',
        'proposal_open_contract': {'contract_id': 12345, 'is_sold': 1, 'profit': 8.5}}))
    assert bot._market_switch_ready()
    assert bot.risk_manager.balance == 1008.5


def test_market_switch_keeps_uncertain_purchase_for_reconciliation(bot):
    ws = MagicMock()
    rid = proposal(bot, ws)
    bot._handle_proposal(ws, {'id': 'quote', 'ask_price': 10, 'payout': 18.5}, req_id=rid)
    bot.market_switch_requested = lambda: True
    bot._on_message(ws, json.dumps({'req_id': bot._buy_req_id, 'error': {'code': 'Disconnected'}}))
    assert bot._phase == 'reconciling'
    assert not bot._market_switch_ready()


@pytest.mark.parametrize('demo',[True,False])
@pytest.mark.parametrize('payout,accepted',[(15.99,False),(16,True),(18.5,True)])
def test_payout_floor_applies_to_demo_and_real(bot,demo,payout,accepted):
    bot.demo=demo
    ws=MagicMock()
    rid=proposal(bot,ws)
    ws.reset_mock()
    bot._handle_proposal(ws,{'id':'quote','ask_price':10,'payout':payout},req_id=rid)
    purchases=[json.loads(c.args[0]) for c in ws.send.call_args_list if 'buy' in json.loads(c.args[0])]
    assert bool(purchases) is accepted


@pytest.mark.parametrize('price,payout',[(.9,1.44),(10,16),(.5,.8)])
def test_exact_sixty_percent_payout_not_lost_to_float_rounding(price,payout):
    from model_contract import profitable_proposal
    assert profitable_proposal(price,payout,.9)
    assert not profitable_proposal(price,payout-.00001,.9)
