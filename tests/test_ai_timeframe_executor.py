"""IA isolada usa OHLC privado de 5min, sem mudar Score nem contratos de 15min."""
import json
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import config
import executor
from executor import DerivBot
from model_contract import FEATURE_WINDOW
from risk_manager import RiskManager
from training_candles import TrainingCandleBuffer


BASE = 1_800_000_000
ENTRY = BASE + FEATURE_WINDOW * 300


def candle(epoch, close=100.):
    return dict(epoch=epoch, open=close, high=close, low=close, close=close)


def cached_candles(*, timeframe=300, count=FEATURE_WINDOW):
    return [dict(epoch=BASE + i * timeframe, symbol='R_100', kind='candle',
                 timeframe=timeframe, price=100., open=100., high=100., low=100.)
            for i in range(count)]


@pytest.fixture
def model_only(monkeypatch):
    monkeypatch.setattr(config, 'USE_AI_MODEL', True)
    for flag in ('USE_WEIGHTED_SIGNAL', 'PRICE_ACTION_FILTER', 'CANDLE_PATTERN_FILTER'):
        monkeypatch.setattr(config, flag, False)
    monkeypatch.setattr(config, 'ENFORCE_ENTRY_CADENCE', False)
    monkeypatch.setattr(executor.time, 'time', lambda: ENTRY)


@pytest.fixture
def private_buffer(tmp_path, monkeypatch):
    buffer = TrainingCandleBuffer(tmp_path / 'private-history', 'R_100', now=lambda: ENTRY)
    monkeypatch.setattr(buffer.history, 'cached_records',
                        lambda _symbol: (cached_candles(), {'excluded_epochs': []}))
    assert buffer.refresh()
    return buffer


@pytest.fixture
def bot(model_only, private_buffer, tmp_path, monkeypatch):
    risk = RiskManager(1000.)
    risk.bind_account('offline-account', 1000.)
    bot = DerivBot(risk)
    bot._auth_verified = True
    bot._training_candles = private_buffer
    bot._shadow_monitor = SimpleNamespace(approved=True)
    bot._shadow_model_id = 'offline'
    bot._shadow_symbol = bot.symbol
    monkeypatch.setattr(executor, '_SHADOW_STATE_JSON', tmp_path / 'shadow.json')
    monkeypatch.setattr(executor.ai_predictor, 'model_id', lambda: '')
    monkeypatch.setattr(bot, '_capture_shadow_prediction', MagicMock())
    return bot


def test_private_closed_candles_do_not_require_operational_warmup(bot, monkeypatch):
    signal = MagicMock(return_value=('BUY', {'model_id': 'offline', 'win_probability': .8}))
    monkeypatch.setattr(executor, 'get_signal', signal)
    bot._history_recovery_incomplete = True
    bot._history_request = (123, BASE, ENTRY)
    ws = MagicMock()

    bot._handle_tick(ws, dict(symbol=bot.symbol, epoch=ENTRY, quote=100.))

    assert len(bot._candles) == 0
    assert len(bot._candle_closes) == 0
    assert signal.call_args.kwargs['candles'] == [candle(BASE + i * 300) for i in range(50)]
    proposal = json.loads(ws.send.call_args.args[0])
    assert proposal['proposal'] == 1
    assert proposal['duration'] == 15 and proposal['duration_unit'] == 'm'
    assert bot._phase == 'proposal'
    bot._capture_shadow_prediction.assert_called_once()


@pytest.mark.parametrize('blocked', ['active_contract', 'risk', 'shadow', 'real'])
def test_private_timeframe_preserves_entry_gates(bot, monkeypatch, blocked):
    signal = MagicMock(return_value=('BUY', {'model_id': 'offline', 'win_probability': .8}))
    monkeypatch.setattr(executor, 'get_signal', signal)
    if blocked == 'active_contract':
        bot._in_trade = True
    elif blocked == 'risk':
        monkeypatch.setattr(bot, '_risk_allows_trade', lambda: False)
    elif blocked == 'shadow':
        bot._shadow_monitor.approved = False
    else:
        bot.demo = False
    ws = MagicMock()

    bot._handle_tick(ws, dict(symbol=bot.symbol, epoch=ENTRY, quote=100.))

    ws.send.assert_not_called()


def test_private_start_subscribes_without_150_fifteen_minute_candles(bot, monkeypatch):
    monkeypatch.setattr(bot, '_request_training_history_refresh', MagicMock())
    monkeypatch.setattr(bot, '_preload_candles_from_csv',
                        lambda: pytest.fail('leitura pesada no callback'))
    ws = MagicMock()

    bot._start_authenticated_stream(ws)

    assert json.loads(ws.send.call_args.args[0]) == {'ticks': bot.symbol, 'subscribe': 1}
    bot._request_training_history_refresh.assert_called_once()


def test_missing_private_candle_blocks_inference_without_touching_score_buffer(bot, monkeypatch):
    del bot._training_candles._candles[BASE + 25 * 300]
    bot._candles.append(candle(BASE))
    monkeypatch.setattr(bot, '_request_training_history_refresh', MagicMock())
    signal = MagicMock()
    monkeypatch.setattr(executor, 'get_signal', signal)

    bot._handle_tick(MagicMock(), dict(symbol=bot.symbol, epoch=ENTRY, quote=100.))

    signal.assert_not_called()
    bot._request_training_history_refresh.assert_called_once()
    assert list(bot._candles) == [candle(BASE)]
    assert bot._decision_history[-1]['reason'] == 'training_history_recovery'


def test_score_and_mixed_modes_keep_fifteen_minute_warmup(bot, monkeypatch):
    monkeypatch.setattr(config, 'USE_WEIGHTED_SIGNAL', True)
    monkeypatch.setattr(executor.time, 'time', lambda: ENTRY)
    bot._preload_candles_from_csv = MagicMock()
    ws = MagicMock()

    bot._request_missing_history_or_subscribe(ws)

    request = json.loads(ws.send.call_args.args[0])
    assert request['style'] == 'candles'
    assert request['granularity'] == 900
    total = request['count'] + sum((end - start) // 900 + 1
                                   for start, end in bot._history_ranges)
    assert total == config.MIN_CANDLES == 150


def test_private_shadow_identity_has_five_minute_features_and_fifteen_minute_duration(bot):
    monitor = bot._get_shadow_monitor('new-offline-model')
    assert monitor.state['timeframe_sec'] == 300
    assert monitor.state['duration_sec'] == 900
    assert monitor.minimum_results == config.SHADOW_MIN_RESULTS == 30


def test_shadow_intermediate_settlement_is_published_and_callback_timing_is_measured(bot, tmp_path, monkeypatch):
    import executor
    from shadow_monitor import ShadowMonitor
    monitor = ShadowMonitor(tmp_path / 'shadow.json', symbol=bot.symbol, model_id='offline',
                            timeframe_sec=300, duration_sec=900)
    assert monitor.observe_prediction(entry_epoch=ENTRY - 900, entry_price=99,
        direction=1, confidence=.8, net_payout=.73) == 'pending'
    bot._shadow_monitor = monitor
    bot._shadow_timeframe = 300
    monkeypatch.setattr(executor.ai_predictor, 'model_id', lambda: 'offline')
    monkeypatch.setattr(bot, '_handle_training_tick', MagicMock())
    readings = iter([10., 10.25])
    monkeypatch.setattr(executor.time, 'perf_counter', lambda: next(readings))
    bot._handle_tick(MagicMock(), dict(symbol=bot.symbol, epoch=ENTRY, quote=100.))
    status = json.loads(executor._BOT_STATUS_JSON.read_text())
    assert status['shadow']['result_count'] == 1
    assert status['shadow']['pending'] is False and status['shadow']['reason'] == 'settled'
    bot._write_runtime_status(force=True)
    timing = json.loads(executor._BOT_STATUS_JSON.read_text())['training_callback_timing']
    assert timing['samples'] == 1 and timing['max_seconds'] == .25


def test_private_history_worker_runs_outside_callback_and_obeys_cooldown(bot, monkeypatch):
    caller = threading.get_ident()
    identities = []
    callback = MagicMock(side_effect=lambda _symbol: identities.append(threading.get_ident()))
    bot.on_training_history_needed = callback
    monkeypatch.setattr(bot._training_candles, 'refresh', MagicMock(return_value=True))

    bot._request_training_history_refresh()
    bot._training_history_worker.join(timeout=5)
    bot._request_training_history_refresh()

    callback.assert_called_once_with(bot.symbol)
    assert identities and identities[0] != caller
    bot._training_candles.refresh.assert_called_once()


def test_private_buffer_rejects_open_stale_and_wrong_timeframe_candles(private_buffer, monkeypatch):
    assert len(private_buffer.window(ENTRY)) == FEATURE_WINDOW
    assert not private_buffer.window(ENTRY - 300)
    assert not private_buffer.window(ENTRY + 300)
    wrong = cached_candles(timeframe=900)
    fresh = TrainingCandleBuffer(private_buffer.history.root, 'R_100', now=lambda: ENTRY)
    monkeypatch.setattr(fresh.history, 'cached_records', lambda _symbol: (wrong, {'excluded_epochs': []}))
    assert fresh.refresh()
    assert fresh.window(ENTRY) == []


def test_private_buffer_preserves_memory_when_cache_busy(private_buffer, monkeypatch):
    def busy(_symbol):
        raise BlockingIOError()
    monkeypatch.setattr(private_buffer.history, 'cached_records', busy)
    assert not private_buffer.refresh()
    assert len(private_buffer.window(ENTRY)) == FEATURE_WINDOW


def test_private_buffer_quarantine_vetoes_whole_feature_window(private_buffer, monkeypatch):
    monkeypatch.setattr(private_buffer.history, 'cached_records',
                        lambda _symbol: (cached_candles(), {'excluded_epochs': [BASE + 900]}))
    assert private_buffer.refresh()
    assert private_buffer.window(ENTRY) == []


def test_private_buffer_does_not_create_partial_candle_and_rejects_tick_gap(private_buffer):
    assert private_buffer.feed(ENTRY + 100, 100.) is False
    assert private_buffer._current is None
    private_buffer.feed(ENTRY + 300, 100.)
    assert ENTRY not in private_buffer._candles
    assert private_buffer.window(ENTRY + 300) == []


def test_private_live_candle_closes_after_continuous_observed_ticks(private_buffer):
    assert private_buffer.feed(ENTRY, 100.)
    for offset in range(5, 300, 5):
        private_buffer.feed(ENTRY + offset, 100.)
    assert private_buffer.feed(ENTRY + 300, 100.)
    assert private_buffer.window(ENTRY + 300)[-1] == candle(ENTRY)


def test_private_live_conflict_cannot_replace_confirmed_ohlc(private_buffer):
    private_buffer.feed(ENTRY, 100.)
    for offset in range(5, 300, 5):
        private_buffer.feed(ENTRY + offset, 100.)
    private_buffer._candles[ENTRY] = candle(ENTRY, close=101.)

    assert not private_buffer.feed(ENTRY + 300, 100.)
    assert private_buffer._candles[ENTRY] == candle(ENTRY, close=101.)
    assert ENTRY in private_buffer._excluded
    assert private_buffer.window(ENTRY + 300) == []


def test_private_snapshot_aggregates_only_fully_observed_tick_candles(tmp_path, monkeypatch):
    buffer = TrainingCandleBuffer(tmp_path, 'R_100', now=lambda: ENTRY)
    ticks = [dict(epoch=BASE + i, symbol='R_100', kind='tick', timeframe=0,
                  price=100., open='', high='', low='') for i in range(0, 15001, 5)]
    monkeypatch.setattr(buffer.history, 'cached_records',
                        lambda _symbol: (ticks, {'excluded_epochs': []}))
    assert buffer.refresh()
    assert len(buffer.window(ENTRY)) == FEATURE_WINDOW
    ticks[:] = [row for row in ticks if not BASE + 400 <= row['epoch'] <= BASE + 425]
    other = TrainingCandleBuffer(tmp_path, 'R_100', now=lambda: ENTRY)
    monkeypatch.setattr(other.history, 'cached_records',
                        lambda _symbol: (ticks, {'excluded_epochs': []}))
    assert other.refresh()
    assert other.window(ENTRY) == []
