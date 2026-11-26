"""Regressões de abstenção, troca por candle e lacunas ao vivo, sem rede."""
import json
from unittest.mock import Mock

import pytest
import config
import executor
import strategy
from risk_manager import RiskManager


def candle(epoch):
    return dict(epoch=epoch, open=1.5, high=2, low=1, close=1.5)


@pytest.mark.parametrize('reason', ['technical_score_below_min', 'adx_below_min',
                                  'technical_component_no_majority'])
def test_single_module_preserves_rejection(monkeypatch, reason):
    monkeypatch.setattr(strategy, 'USE_WEIGHTED_SIGNAL', True)
    monkeypatch.setattr(strategy, 'USE_AI_MODEL', False)
    monkeypatch.setattr(strategy, 'PRICE_ACTION_FILTER', False)
    monkeypatch.setattr(strategy, 'CANDLE_PATTERN_FILTER', False)
    monkeypatch.setattr(strategy, '_weighted_signal',
                        lambda *a, **kw: (None, {'decision_block': reason}))
    signal, details = strategy.get_signal([100+i*.05 for i in range(200)], adx_min=0)
    assert signal is None
    assert details['decision_block'] == reason
    assert reason in config.UNFAVORABLE_SYMBOL_SWITCH_REASONS


def test_switch_counts_closed_candles_once(monkeypatch):
    bot = executor.DerivBot(RiskManager(1000))
    bot.on_market_unfavorable = Mock()
    monkeypatch.setattr(config, 'UNFAVORABLE_SYMBOL_SWITCH_DECISIONS', 2)
    reason = 'technical_score_below_min'
    bot._record_decision(reason=reason, tick_epoch=900)
    assert bot._unfavorable_decision_streak == 0
    bot._new_candle_closed = True
    for epoch in (900, 901, 902):
        bot._record_decision(reason=reason, tick_epoch=epoch)
    assert bot._unfavorable_decision_streak == 1
    bot.on_market_unfavorable.assert_not_called()
    bot._record_decision(reason=reason, tick_epoch=1800)
    bot.on_market_unfavorable.assert_called_once()


@pytest.mark.parametrize('response', ['complete', 'partial', 'error', 'timeout'])
def test_live_gap_blocks_orders_and_recovers_without_erasing_history(monkeypatch, response):
    bot = executor.DerivBot(RiskManager(1000))
    bot._auth_verified = True
    bot._merge_candles([candle(e) for e in range(900, 151*900, 900)])
    bot._last_tick_epoch = 150*900
    bot._last_tick_price = 1.5
    bot._reset_watchdog = Mock()
    bot._send_proposal = Mock()
    bot._preload_candles_from_csv = Mock()
    now = [152*900+1]
    monkeypatch.setattr(executor.time, 'time', lambda: now[0])
    ws = Mock()
    bot._handle_tick(ws, dict(symbol=bot.symbol, quote=1.5, epoch=now[0]))
    assert len(bot._candles) == 150
    assert len(bot._history_candles) == 150
    request = json.loads(ws.send.call_args.args[0])
    assert request['start'] == 151*900
    assert request['count'] == 1
    if response == 'timeout':
        now[0] += 61
        bot._handle_tick(ws, dict(symbol=bot.symbol, quote=1.5, epoch=now[0]))
        assert bot._history_request[0] != request['req_id']
    elif response == 'error':
        bot._on_message(ws, json.dumps(dict(req_id=request['req_id'],
                                           error={'message': 'offline rejection'})))
    else:
        bot._handle_api_candles(ws, [candle(151*900)] if response == 'complete' else [],
                               req_id=request['req_id'])
    assert len(bot._history_candles) >= 150
    assert not any('subscribe' in json.loads(c.args[0]) for c in ws.send.call_args_list)
    bot._send_proposal.assert_not_called()
    if response == 'complete':
        assert len(bot._candles) == 151
        decision = Mock(return_value=(None, {'decision_block': 'technical_score_below_min'}))
        monkeypatch.setattr(executor, 'get_signal', decision)
        bot._risk_allows_trade = Mock(return_value=True)
        monkeypatch.setattr(config, 'ENFORCE_ENTRY_CADENCE', False)
        bot._handle_tick(ws, dict(symbol=bot.symbol, quote=1.5, epoch=now[0]+1))
        decision.assert_called_once()
    elif response != 'timeout':
        # Sem resposta completa, o próximo tick continua bloqueado sem repetir a API.
        calls = ws.send.call_count
        bot._handle_tick(ws, dict(symbol=bot.symbol, quote=1.5, epoch=now[0]+1))
        assert ws.send.call_count == calls
        assert bot._decision_history[-1]['reason'] == 'history_recovery'


def test_internal_gap_after_new_candle_triggers_incremental_backfill(monkeypatch):
    bot = executor.DerivBot(RiskManager(1000))
    bot._auth_verified = True
    bot._merge_candles([candle(e) for e in range(900, 151*900, 900)])
    bot._cur_candle_epoch = 152*900
    bot._cur_candle_open = bot._cur_candle_close = 1.5
    bot._cur_candle_high, bot._cur_candle_low = 2, 1
    bot._last_tick_epoch = 153*900-1
    bot._last_tick_price = 1.5
    bot._reset_watchdog = Mock()
    bot._check_candle_patterns = Mock()
    monkeypatch.setattr(executor.time, 'time', lambda: 153*900)
    ws = Mock()
    bot._handle_tick(ws, dict(symbol=bot.symbol, quote=1.5, epoch=153*900))
    assert len(bot._candles) == 1
    assert len(bot._history_candles) == 151
    request = json.loads(ws.send.call_args.args[0])
    assert request['start'] == 151*900 and request['count'] == 1
    bot._handle_api_candles(ws, [candle(151*900)], req_id=request['req_id'])
    assert len(bot._candles) == 152
