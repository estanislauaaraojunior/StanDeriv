"""Market failover without purchases, model mixing or risk resets."""
import json
import threading
from pathlib import Path
from unittest.mock import Mock

import pytest
import pipeline
from payout_quotes import ProposalRecorder


@pytest.mark.parametrize('code,message,expected', [
    ('ContractBuyValidationError', 'Trading is not available from 20:50:00 to 22:00:00.', True),
    ('ContractBuyValidationError', 'Trading on forex contracts with duration less than 5 hours is not available', True),
    ('MarketIsClosed', 'Market closed', True),
    ('ContractBuyValidationError', 'Minimum stake is 5', False),
    ('AuthorizationRequired', 'Authorize first', False),
])
def test_recorder_only_requests_switch_for_market_restriction(tmp_path, code, message, expected):
    callback, ws = Mock(), Mock()
    recorder = ProposalRecorder(tmp_path/'quotes.csv', 'R_100', 15, 'm', on_unavailable=callback)
    recorder.sample(ws, bearer=True, now=100)
    request = json.loads(ws.send.call_args.args[0])
    recorder.handle({'req_id': request['req_id'], 'error': {'code': code, 'message': message}}, now=101)
    assert callback.called is expected
    assert not (tmp_path/'quotes.csv').exists()
    assert all('buy' not in json.loads(c.args[0]) for c in ws.send.call_args_list)


def test_stale_error_cannot_switch_market(tmp_path):
    callback, ws = Mock(), Mock()
    recorder = ProposalRecorder(tmp_path/'quotes.csv', 'R_100', 15, 'm', on_unavailable=callback)
    recorder.sample(ws, bearer=True, now=100)
    request = json.loads(ws.send.call_args.args[0])
    recorder.handle({'req_id': request['req_id'], 'error': {'code': 'MarketIsClosed'}}, now=200)
    callback.assert_not_called()


def test_unfavorable_market_requests_switch(monkeypatch):
    event = threading.Event()
    monkeypatch.setattr(pipeline, '_market_switch', event)
    monkeypatch.setattr(pipeline, '_runtime_symbol', lambda: 'frxGBPJPY')
    pipeline._request_market_switch_for_unfavorable(
        'frxGBPJPY',
        'no_technical_signal',
        12,
        {'adx': 12.7},
    )
    assert event.is_set()


def test_initial_wait_observes_market_change(monkeypatch):
    event = threading.Event(); event.set()
    monkeypatch.setattr(pipeline, '_market_switch', event)
    with pytest.raises(pipeline.MarketUnavailable):
        pipeline._wait_for_ticks(200)
    with pytest.raises(pipeline.MarketUnavailable):
        pipeline._wait_for_compatible_model()


def test_compatible_model_fallback_uses_live_approved_symbol(monkeypatch):
    checked = []

    def load_model(_path, symbol):
        checked.append(symbol)
        if symbol != 'frxGBPUSD':
            raise ValueError('modelo sem validação compatível')
        return {}, {'symbol': symbol, 'approved': True}

    monkeypatch.setattr(pipeline, 'load_validated_bundle', load_model)
    monkeypatch.setattr(pipeline, 'usable', lambda metadata, symbol: metadata['symbol'] == symbol)
    monkeypatch.setattr(pipeline, '_probe_symbol', lambda symbol: (symbol == 'frxGBPUSD', 'fechado'))

    assert pipeline._first_live_symbol_with_compatible_model(['frxEURUSD', 'frxGBPUSD']) == 'frxGBPUSD'
    assert checked == ['frxEURUSD', 'frxGBPUSD']


def test_compatible_model_fallback_rejects_closed_symbol(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        'load_validated_bundle',
        lambda _path, symbol: ({}, {'symbol': symbol, 'approved': True}),
    )
    monkeypatch.setattr(pipeline, 'usable', lambda _metadata, _symbol: True)
    monkeypatch.setattr(pipeline, '_probe_symbol', lambda _symbol: (False, 'mercado fechado'))

    assert pipeline._first_live_symbol_with_compatible_model(['frxGBPUSD']) == ''


def test_session_cleanup_precedes_symbol_change(monkeypatch):
    order = []
    lock = Mock()
    monkeypatch.setattr(pipeline, 'acquire_execution_lock', lambda _: lock)
    monkeypatch.setattr(pipeline, '_shutdown', threading.Event())
    monkeypatch.setattr(pipeline, '_market_switch', threading.Event())
    monkeypatch.setattr(pipeline, '_model_ready', threading.Event())
    monkeypatch.setattr(pipeline, '_retrain_state', {})
    monkeypatch.setattr(pipeline, '_reset_ai_predictor', lambda: None)
    def session(resources, symbol):
        order.append(('session', symbol))
        if symbol is None:
            resources['collector'] = Mock(stop=lambda: order.append('stop'), is_alive=Mock(side_effect=[True, False]),
                                          join=lambda timeout: order.append('joined'))
            resources['retrain'] = Mock(is_alive=Mock(side_effect=[True, False]),
                                       join=lambda timeout: order.append('train_finished'))
            raise pipeline.MarketUnavailable('OLD')
    def compatible(candidates):
        assert order[-1] == 'train_finished'
        assert 'OLD' not in candidates
        return ''
    def select(candidates):
        assert order[-1] == 'train_finished'
        assert 'OLD' not in candidates
        return 'NEW'
    monkeypatch.setattr(pipeline, '_main_session', session)
    monkeypatch.setattr(pipeline, '_scan_candidates', lambda: ['OLD', 'NEW'])
    monkeypatch.setattr(pipeline, '_first_live_symbol_with_compatible_model', compatible)
    monkeypatch.setattr(pipeline, '_first_live_symbol', select)
    pipeline.main()
    assert order == [('session', None), 'stop', 'joined', 'train_finished', ('session', 'NEW')]
    lock.close.assert_called_once()


def test_market_change_prefers_symbol_with_compatible_model(monkeypatch):
    order = []
    lock = Mock()
    monkeypatch.setattr(pipeline, 'acquire_execution_lock', lambda _: lock)
    monkeypatch.setattr(pipeline, '_shutdown', threading.Event())
    monkeypatch.setattr(pipeline, '_market_switch', threading.Event())
    monkeypatch.setattr(pipeline, '_model_ready', threading.Event())
    monkeypatch.setattr(pipeline, '_retrain_state', {})
    monkeypatch.setattr(pipeline, '_reset_ai_predictor', lambda: None)

    def session(_resources, symbol):
        order.append(('session', symbol))
        if symbol is None:
            raise pipeline.MarketUnavailable('OLD')

    monkeypatch.setattr(pipeline, '_main_session', session)
    monkeypatch.setattr(pipeline, '_scan_candidates', lambda: ['OLD', 'LIVE_ONLY', 'APPROVED'])
    monkeypatch.setattr(pipeline, '_first_live_symbol_with_compatible_model', lambda candidates: 'APPROVED')
    fallback = Mock(return_value='LIVE_ONLY')
    monkeypatch.setattr(pipeline, '_first_live_symbol', fallback)

    pipeline.main()

    assert order == [('session', None), ('session', 'APPROVED')]
    fallback.assert_not_called()
    lock.close.assert_called_once()


def test_market_switch_budget_enters_degraded_state(monkeypatch):
    lock = Mock()
    writes = []
    monkeypatch.setattr(pipeline, 'acquire_execution_lock', lambda _: lock)
    monkeypatch.setattr(pipeline, '_shutdown', threading.Event())
    monkeypatch.setattr(pipeline, '_market_switch', threading.Event())
    monkeypatch.setattr(pipeline, '_model_ready', threading.Event())
    monkeypatch.setattr(pipeline, '_retrain_state', {})
    monkeypatch.setattr(pipeline, '_reset_ai_predictor', lambda: None)
    monkeypatch.setattr(pipeline, 'MARKET_SWITCH_MAX_PER_WINDOW', 1)
    monkeypatch.setattr(pipeline, 'MARKET_SWITCH_WINDOW_SEC', 3600)
    monkeypatch.setattr(pipeline, '_scan_candidates', lambda: ['OLD', 'NEW'])
    monkeypatch.setattr(pipeline, '_write_pipeline_status', lambda **kwargs: writes.append(kwargs))

    def session(_resources, _symbol):
        raise pipeline.MarketUnavailable('OLD')

    monkeypatch.setattr(pipeline, '_main_session', session)

    pipeline.main()

    assert writes[-1]["status"] == "degraded"
    assert writes[-1]["running"] is False
    lock.close.assert_called_once()


def test_top_trending_symbols_selects_best_live_markets(monkeypatch):
    monkeypatch.setattr(pipeline, '_SCAN_SYMBOLS_FOREX', ['A', 'B', 'C'])
    monkeypatch.setattr(pipeline, '_SCAN_SYMBOLS_PRIMARY', ['D', 'E', 'F'])
    monkeypatch.setattr(pipeline, '_SCAN_SYMBOLS_SECONDARY', ['G'])
    monkeypatch.setattr(pipeline, '_ADX_TREND_MIN', 10.0)

    scores = {'A': 18.0, 'B': 40.0, 'C': 9.0, 'D': 30.0, 'E': 25.0, 'F': 12.0, 'G': 50.0}
    monkeypatch.setattr(pipeline, '_scan_group', lambda candidates: {symbol: scores[symbol] for symbol in candidates})
    monkeypatch.setattr(pipeline, '_probe_symbol', lambda symbol: (symbol != 'D', 'fechado'))

    assert pipeline._detect_top_trending_symbols(limit=5) == ['G', 'B', 'E', 'A', 'F']


def test_training_rank_prefers_eligible_expected_return():
    rejected = {'eligible': False, 'expectancy': 99, 'win_rate': 1, 'auc': 1, 'f1': 1, 'trades': 500}
    approved_low = {'eligible': True, 'expectancy': 0.02, 'win_rate': .55, 'auc': .6, 'f1': .5, 'trades': 20}
    approved_high = {'eligible': True, 'expectancy': 0.08, 'win_rate': .52, 'auc': .58, 'f1': .48, 'trades': 12}

    assert max([rejected, approved_low, approved_high], key=pipeline._training_rank_key) is approved_high


def test_initial_symbol_tournament_promotes_best_approved(monkeypatch, tmp_path):
    active = {'symbol': 'R_100'}
    started = []
    stopped = []
    trained = []
    model_path = tmp_path / 'model.pkl'

    class Collector:
        def __init__(self, symbol):
            self.symbol = symbol
            self.fatal_error = ''
        def start(self):
            started.append(self.symbol)
        def stop(self):
            stopped.append(self.symbol)
        def is_alive(self):
            return False
        def join(self, timeout=None):
            return None

    def set_symbol(symbol):
        active['symbol'] = symbol

    def run_training(*, dataset_path, model_path, **_kwargs):
        symbol = active['symbol']
        trained.append((symbol, dataset_path, model_path))
        Path(model_path).write_text(f'model-{symbol}')
        Path(model_path + '.metadata.json').write_text(f'metadata-{symbol}')
        metrics = {
            'LOW': {'eligible': True, 'expectancy': 0.01, 'win_rate': .60, 'auc': .55, 'f1': .40, 'trades': 10},
            'BEST': {'eligible': True, 'expectancy': 0.09, 'win_rate': .58, 'auc': .62, 'f1': .50, 'trades': 8},
            'BAD': {'eligible': False, 'expectancy': 1.0, 'win_rate': 1.0, 'auc': 1.0, 'f1': 1.0, 'trades': 99, 'rejection_reasons': ['reprovado']},
        }[symbol]
        return True, metrics

    monkeypatch.setattr(pipeline, 'AI_MODEL_PATH', str(model_path))
    monkeypatch.setattr(pipeline, 'DATASET_CSV', str(tmp_path / 'dataset.csv'))
    monkeypatch.setattr(pipeline, 'TRANSFORMER_MODEL_PATH', str(tmp_path / 'transformer.pkl'))
    monkeypatch.setattr(pipeline, 'DURATION_MODEL_PATH', str(tmp_path / 'duration.pkl'))
    monkeypatch.setattr(pipeline, '_runtime_symbol', lambda: active['symbol'])
    monkeypatch.setattr(pipeline, '_set_runtime_symbol', set_symbol)
    monkeypatch.setattr(pipeline, '_ensure_ticks_header', lambda: None)
    monkeypatch.setattr(pipeline, '_CollectorThread', Collector)
    monkeypatch.setattr(pipeline, '_fetch_historical_ticks', lambda _count: 0)
    monkeypatch.setattr(pipeline, '_wait_for_ticks', lambda _min_ticks, collector=None: None)
    monkeypatch.setattr(pipeline, '_run_training', run_training)
    monkeypatch.setattr(pipeline, '_reset_ai_predictor', lambda: None)

    winner, metrics = pipeline._run_initial_symbol_tournament(
        ['LOW', 'BEST', 'BAD'],
        history_count=100,
        min_ticks=10,
        resources={},
    )

    assert winner == 'BEST'
    assert metrics['expectancy'] == 0.09
    assert active['symbol'] == 'BEST'
    assert model_path.read_text() == 'model-BEST'
    assert Path(str(model_path) + '.metadata.json').read_text() == 'metadata-BEST'
    assert started == ['LOW', 'BEST', 'BAD']
    assert stopped == ['LOW', 'BEST', 'BAD']
    assert [item[0] for item in trained] == ['LOW', 'BEST', 'BAD']


def test_main_session_passes_top_symbols_limit(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(pipeline.sys, 'argv', ['pipeline.py', '--demo', '--top-symbols', '3'])
    monkeypatch.setattr(pipeline, '_validate_deriv_token', lambda demo: True)

    def detect_top(no_scan=False, limit=0):
        captured['limit'] = limit
        return ['AAA']

    monkeypatch.setattr(pipeline, '_detect_top_trending_symbols', detect_top)
    monkeypatch.setattr(pipeline, '_ensure_operable_symbol', lambda symbol: symbol)
    monkeypatch.setattr(pipeline, '_set_runtime_symbol', lambda symbol: captured.setdefault('runtime', symbol))
    monkeypatch.setattr(pipeline, '_prepare_training_start', lambda **_kwargs: None)
    monkeypatch.setattr(pipeline, '_run_initial_symbol_tournament', lambda candidates, **_kwargs: ('AAA', {'eligible': True}))
    monkeypatch.setattr(pipeline, '_banner', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline, '_ensure_ticks_header', lambda: None)
    monkeypatch.setattr(pipeline, '_fetch_historical_ticks', lambda _count: 0)
    monkeypatch.setattr(pipeline, '_count_ticks', lambda: 0)
    monkeypatch.setattr(pipeline, '_load_latest_final_metrics', lambda: {'eligible': True})
    monkeypatch.setattr(pipeline, '_uses_bearer_auth', lambda: False)

    class Collector:
        fatal_error = ''
        def __init__(self, symbol):
            captured['collector'] = symbol
        def start(self):
            return None

    class Bot:
        def __init__(self, **_kwargs):
            self.run_called = False
        def run(self):
            captured['bot_run'] = True

    monkeypatch.setattr(pipeline, '_CollectorThread', Collector)
    monkeypatch.setattr(pipeline, 'RiskManager', lambda initial_balance: Mock())
    monkeypatch.setattr(pipeline, 'DerivBot', Bot)
    monkeypatch.setattr(pipeline, '_check_market_switch', lambda: None)
    monkeypatch.setattr(pipeline, '_retrain_loop', lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline.threading, 'Thread', lambda *args, **kwargs: Mock(start=lambda: None))
    monkeypatch.setattr(pipeline, 'TICKS_CSV', str(tmp_path / 'ticks.csv'))

    pipeline._main_session({'lock': Mock()})

    assert captured['limit'] == 3
    assert captured['collector'] == 'AAA'
    assert captured['bot_run'] is True


def test_main_session_rejects_invalid_top_symbols(monkeypatch):
    monkeypatch.setattr(pipeline.sys, 'argv', ['pipeline.py', '--demo', '--top-symbols', '0'])
    with pytest.raises(SystemExit):
        pipeline._main_session({'lock': Mock()})


def test_initial_training_failure_starts_only_bootstrap_retrain(monkeypatch, tmp_path):
    started_threads = []
    captured = {'symbol': 'R_100'}

    class FakeThread:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs
            self.name = kwargs.get('name')
        def start(self):
            started_threads.append(self)

    class Collector:
        fatal_error = ''
        def __init__(self, symbol):
            self.symbol = symbol
        def start(self):
            return None
        def stop(self):
            return None
        def is_alive(self):
            return True

    class Bot:
        def __init__(self, **_kwargs):
            pass
        def run(self):
            captured['bot_run'] = True

    monkeypatch.setattr(pipeline.sys, 'argv', ['pipeline.py', '--demo', '--top-symbols', '1'])
    monkeypatch.setattr(pipeline, 'TICKS_CSV', str(tmp_path / 'ticks.csv'))
    monkeypatch.setattr(pipeline, 'AI_MODEL_PATH', str(tmp_path / 'model.pkl'))
    monkeypatch.setattr(pipeline, '_validate_deriv_token', lambda demo: True)
    monkeypatch.setattr(pipeline, '_detect_top_trending_symbols', lambda **_kwargs: ['R_100'])
    monkeypatch.setattr(pipeline, '_ensure_operable_symbol', lambda symbol: symbol)
    monkeypatch.setattr(pipeline, '_set_runtime_symbol', lambda symbol: captured.update(symbol=symbol))
    monkeypatch.setattr(pipeline, '_runtime_symbol', lambda: captured['symbol'])
    monkeypatch.setattr(pipeline, '_prepare_training_start', lambda **_kwargs: None)
    monkeypatch.setattr(pipeline, '_run_initial_symbol_tournament', lambda candidates, **_kwargs: ('', None))
    monkeypatch.setattr(pipeline, '_banner', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline, '_ensure_ticks_header', lambda: None)
    monkeypatch.setattr(pipeline, '_CollectorThread', Collector)
    monkeypatch.setattr(pipeline, '_fetch_historical_ticks', lambda _count: 0)
    monkeypatch.setattr(pipeline, '_count_ticks', lambda: 200)
    monkeypatch.setattr(pipeline, '_wait_for_ticks', lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, '_run_training', lambda **_kwargs: (True, {'eligible': False, 'rejection_reasons': ['x']}))
    monkeypatch.setattr(pipeline, '_should_promote_model', lambda current, challenger: (False, 'x'))
    monkeypatch.setattr(pipeline, '_load_latest_final_metrics', lambda: None)
    monkeypatch.setattr(pipeline, '_wait_for_compatible_model', lambda collector=None: True)
    monkeypatch.setattr(pipeline, '_uses_bearer_auth', lambda: False)
    monkeypatch.setattr(pipeline, 'RiskManager', lambda initial_balance: Mock())
    monkeypatch.setattr(pipeline, 'DerivBot', Bot)
    monkeypatch.setattr(pipeline, '_check_market_switch', lambda: None)
    monkeypatch.setattr(pipeline.threading, 'Thread', FakeThread)

    pipeline._main_session({'lock': Mock()})

    assert [thread.name for thread in started_threads] == ['RetrainBootstrap']
    assert started_threads[0].kwargs.get('kwargs') == {'bootstrap_until_ready': True}
    assert captured['bot_run'] is True


def test_bootstrap_retrain_requests_market_switch_after_repeated_rejections(monkeypatch, tmp_path):
    calls = {"training": 0, "switch": []}
    event = threading.Event()

    monkeypatch.setattr(pipeline, "_shutdown", threading.Event())
    monkeypatch.setattr(pipeline, "_market_switch", event)
    monkeypatch.setattr(pipeline, "BOOTSTRAP_MAX_FAILED_PROMOTIONS", 3)
    monkeypatch.setattr(pipeline, "RETRAIN_NEW_LABELS_TRIGGER", 1)
    monkeypatch.setattr(pipeline, "TICKS_CSV", str(tmp_path / "ticks.csv"))
    monkeypatch.setattr(pipeline, "AI_MODEL_PATH", str(tmp_path / "model.pkl"))
    monkeypatch.setattr(pipeline, "_runtime_symbol", lambda: "R_100")
    monkeypatch.setattr(pipeline, "_count_ticks", lambda: 500)
    monkeypatch.setattr(pipeline, "_count_labelable_contracts_for_training", lambda: 9999)
    monkeypatch.setattr(pipeline, "minimum_training_rows", lambda: 100)
    monkeypatch.setattr(pipeline, "_load_latest_final_metrics", lambda: None)
    monkeypatch.setattr(pipeline, "_should_promote_model", lambda current, challenger: (False, "reprovado"))
    monkeypatch.setattr(pipeline, "_request_market_switch_for_unfavorable", lambda *args: (calls["switch"].append(args), event.set()))

    def fake_sleep(_seconds):
        if event.is_set():
            pipeline._shutdown.set()

    def fake_run_training(**_kwargs):
        calls["training"] += 1
        Path(str(tmp_path / "model.pkl.new")).write_text("candidate")
        return True, {"eligible": False, "rejection_reasons": ["reprovado"]}

    monkeypatch.setattr(pipeline.time, "sleep", fake_sleep)
    monkeypatch.setattr(pipeline, "_run_training", fake_run_training)

    pipeline._retrain_state.update(last_epoch=0, last_tick_count=0, last_label_count=0, bootstrap_failed_promotions=0)
    pipeline._retrain_loop(bootstrap_until_ready=True)

    assert calls["training"] == 3
    assert calls["switch"]
    assert calls["switch"][-1][:3] == ("R_100", "bootstrap_model_not_promoted", 3)
