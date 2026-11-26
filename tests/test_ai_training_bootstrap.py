"""Complemento histórico da IA sem coleta nova, autenticação ou ordens."""
import csv
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import config
import pipeline
from market_data import FIELDS
from training_history import TrainingHistory

ORIGINAL_PROVIDER = pipeline._training_history_provider


@pytest.fixture
def model_only(monkeypatch, tmp_path):
    monkeypatch.setattr(config, 'USE_AI_MODEL', True)
    for key in ('USE_WEIGHTED_SIGNAL', 'PRICE_ACTION_FILTER', 'CANDLE_PATTERN_FILTER'):
        monkeypatch.setattr(config, key, False)
    monkeypatch.setattr(pipeline, 'TICKS_CSV', str(tmp_path / 'ticks.csv'))
    monkeypatch.setattr(pipeline, '_runtime_symbol', lambda: 'FIRST')
    monkeypatch.setattr(pipeline, '_training_survey_seen', set())
    pipeline._shutdown.clear()
    pipeline._market_switch.clear()


def tick(epoch=1_800_000_000):
    return dict(epoch=epoch, datetime='', symbol='FIRST', price=100.,
                kind='tick', timeframe=0, open='', high='', low='')


def test_provider_is_five_minutes_without_changing_collection(model_only, monkeypatch):
    monkeypatch.setattr(pipeline, '_training_history_instances', {})
    from model_contract import model_timeframe_sec
    provider = ORIGINAL_PROVIDER()
    assert provider.timeframe == 300 and provider.duration == 900
    assert model_timeframe_sec() == 300
    assert config.CANDLE_TIMEFRAME_SEC == pipeline.CANDLE_TIMEFRAME_SEC == 900
    assert pipeline._training_cache_key('FIRST')[-1] == 300


def test_existing_ready_cache_needs_no_public_query(tmp_path, model_only, monkeypatch):
    provider = TrainingHistory(tmp_path / 'history', timeframe=300, now=lambda: 1_800_001_000,
                               client_factory=Mock(side_effect=AssertionError('consulta desnecessária')))
    # Construir o cliente com orçamento zero não deve abrir socket.
    provider.client_factory = lambda **kwargs: SimpleNamespace(
        stats={'requests': 0}, close=lambda: None,
        request=Mock(side_effect=__import__('training_history').RecoveryBudgetExhausted('zero')),
    )
    provider._save(provider._folder('FIRST'), 'FIRST', [tick()],
                   dict(excluded_epochs=[], retries={}))
    monkeypatch.setattr(pipeline.dataset_builder, 'training_diagnostics_from_records',
                        lambda rows, report, symbol: {'stage': 'ready'})
    provider.enrich_incremental = Mock(side_effect=AssertionError('dados já prontos'))
    rows, report = pipeline._training_enricher(provider, 'FIRST')([], dict(symbol='FIRST', excluded_epochs=[]))
    assert rows == [tick()]
    assert report['stop_reason'] == 'ready' and report['requests'] == 0
    provider.enrich_incremental.assert_not_called()


def test_bootstrap_counts_recovered_ticks_without_waiting_primary_csv(model_only, monkeypatch):
    monkeypatch.setattr(pipeline, '_count_ticks', lambda: 0)
    key = pipeline._training_cache_key('FIRST')
    def count():
        pipeline._training_readiness_cache[key] = (0, dict(ticks=20000, ready_for_training=True))
        return 1000
    monkeypatch.setattr(pipeline, '_count_labelable_contracts_for_training', count)
    monkeypatch.setattr(pipeline, '_pipeline_heartbeat', lambda *_a, **_k: None)
    monkeypatch.setattr(pipeline.time, 'sleep', Mock(side_effect=AssertionError('não esperar coleta')))
    pipeline._wait_for_ticks(100)


def test_corrupt_source_is_preserved_and_replaced_by_independent_snapshot(model_only, tmp_path):
    source = Path(pipeline.TICKS_CSV)
    with source.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        writer.writerow([1_800_000_000, '', 'OTHER', 100, 'tick', 0, 'FIRST', '', '', '', '', '', ''])
    before = source.read_bytes()
    def public_only(rows, report):
        assert rows == [] and report['excluded_epochs'] == []
        return [tick()], {'stage': 'partial', 'requests': 1}
    snapshot, report = pipeline._prepare_model_snapshot(
        source, tmp_path / 'prepared', 'FIRST', history_path=None, enricher=public_only)
    assert source.read_bytes() == before
    assert report['blocked_source']['policy'] == 'independent_verified_public_history'
    assert snapshot.parent.name.endswith('-independent')
    assert report['records'] == 1


def test_unused_fifteen_minute_history_cannot_block_five_minute_preparation(model_only, tmp_path):
    source = Path(pipeline.TICKS_CSV)
    with source.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, FIELDS)
        writer.writeheader()
        writer.writerow(tick())
    technical_history = tmp_path / 'technical900.csv'
    technical_history.write_text('invalid unrelated OHLC900\n')
    snapshot, report = pipeline._prepare_model_snapshot(
        source, tmp_path / 'prepared', 'FIRST', history_path=technical_history,
        enricher=lambda rows, _report: (rows, {'stage': 'available'}))
    assert snapshot.exists() and report['records'] == 1
    assert 'blocked_source' not in report
    assert technical_history.read_text() == 'invalid unrelated OHLC900\n'


def test_busy_local_pass_shares_previously_verified_cache(model_only, tmp_path):
    import fcntl
    provider = TrainingHistory(tmp_path / 'history', timeframe=300, now=lambda: 1_800_001_000)
    provider._save(provider._folder('FIRST'), 'FIRST', [tick()],
                   dict(excluded_epochs=[], retries={}))
    with (provider.root / '.public-history.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows, report = pipeline._training_enricher(provider, 'FIRST')(
            [], dict(symbol='FIRST', excluded_epochs=[]))
    assert rows == [tick()]
    assert report['stage'] == 'busy' and report['cache_preserved']


def test_busy_recovery_preserves_previous_diagnostic(model_only, monkeypatch, tmp_path):
    key = pipeline._training_cache_key('FIRST')
    previous = dict(symbol='FIRST', stage='ready', readiness_stage='ready',
                    ready_for_training=True, labelable_contracts=900, ticks=10000)
    pipeline._training_readiness_cache[key] = (-1000, previous)
    monkeypatch.setattr(pipeline, '_prepare_model_snapshot', lambda *_a, **_k:
                        (tmp_path / 'unused', {'history_recovery': {'stage': 'busy', 'requests': 0}}))
    result = pipeline._diagnose_training_symbol('FIRST')
    assert result['ready_for_training'] and result['labelable_contracts'] == 900
    assert result['history_recovery']['stage'] == 'busy'


def test_survey_reaches_beyond_first_five_without_changing_scanner(model_only, monkeypatch):
    eligible = ['A', 'B', 'C', 'D', 'E', 'READY']
    visited = []
    clock = [100.]
    monkeypatch.setattr(pipeline, '_load_latest_final_metrics', lambda: None)
    monkeypatch.setattr(pipeline.time, 'monotonic', lambda: clock[0])
    def scan(*, limit, exclude_symbols):
        assert limit == 5
        return [symbol for symbol in eligible if symbol not in exclude_symbols][:limit]
    monkeypatch.setattr(pipeline, '_detect_top_trending_symbols', scan)
    monkeypatch.setattr(pipeline, '_first_live_symbol_with_compatible_model', lambda _symbols: '')
    def diagnose(symbol):
        visited.append(symbol)
        return dict(ready_for_training=symbol == 'READY', labelable_contracts=1000)
    monkeypatch.setattr(pipeline, '_diagnose_training_symbol', diagnose)
    for _ in range(3):
        pipeline._maybe_request_training_candidate()
        clock[0] += 61
    assert 'READY' in visited
    assert pipeline._training_next_symbol == 'READY'
    assert pipeline._market_switch.is_set()


def test_internal_single_class_does_not_report_ready():
    import numpy as np
    import pandas as pd
    from training_policy import readiness_capacity
    signal = 1_800_000_000 + np.arange(579) * 900
    frame = pd.DataFrame(dict(signal_epoch=signal, entry_epoch=signal, expiry_epoch=signal + 900,
                              feature_start_epoch=signal - 45000, target=np.arange(579) % 2))
    frame.loc[247:278, 'target'] = 0
    diagnostic = readiness_capacity(frame)
    assert diagnostic['evaluation_capacity'] == {'validation': 30, 'test': 116}
    assert diagnostic['stage'] == 'evaluation_capacity'
    assert 'seleção semanal' in diagnostic['reason']


def test_live_recovery_requests_recent_data_without_authentication(model_only, monkeypatch):
    provider = SimpleNamespace(enrich_incremental=Mock())
    monkeypatch.setattr(pipeline, '_training_history_provider', lambda: provider)
    monkeypatch.setattr(pipeline.time, 'time', lambda: 1_800_000_000)
    pipeline._recover_model_candles('FIRST')
    args, kwargs = provider.enrich_incremental.call_args
    assert args == ([], dict(symbol='FIRST', excluded_epochs=[], recovery_start=1_800_000_000 - 21600))
    assert kwargs == dict(rounds=2, total_requests=24, total_seconds=25)


class HistoricalComplement:
    """API falsa: disponibiliza um prefixo histórico ainda não usado no treino."""
    base, timeframe, candles = 1_800_000_000, 300, 1300

    def __init__(self):
        self.calls = []

    def price(self, epoch):
        return 100 + math.sin((epoch - self.base) / 1800)

    def records(self, first):
        rows = []
        for index in range(first, self.candles):
            epoch = self.base + index * self.timeframe
            prices = [self.price(value) for value in range(epoch, epoch + self.timeframe, 10)]
            rows.append(dict(epoch=epoch, datetime='', symbol='FIRST', kind='candle',
                             timeframe=300, price=prices[-1], open=prices[0],
                             high=max(prices), low=min(prices)))
        rows.extend(dict(tick(epoch), price=self.price(epoch)) for epoch in
                    range(self.base + first * self.timeframe,
                          self.base + self.candles * self.timeframe + 1, 10))
        return sorted(rows, key=lambda row: (row['epoch'], row['kind']))

    def client(self, *, max_requests, seconds):
        server = self
        class Client:
            stats = None
            def __init__(self):
                self.stats = dict(requests=0, timeouts=0, request_errors=0, max_latency_sec=0.)
            def request(self, request):
                from training_history import RecoveryBudgetExhausted
                if self.stats['requests'] >= max_requests:
                    raise RecoveryBudgetExhausted('orçamento simulado')
                self.stats['requests'] += 1
                server.calls.append(request)
                start, end = request['start'], int(request['end'])
                if request['style'] == 'candles':
                    rows = [row for row in server.records(-150) if row['kind'] == 'candle'
                            and start <= row['epoch'] <= end][-request['count']:]
                    return dict(candles=[dict(epoch=row['epoch'], open=row['open'], high=row['high'],
                                              low=row['low'], close=row['price']) for row in rows])
                epochs = list(range(max(start, server.base - 150 * 300),
                                    min(end, server.base + server.candles * 300) + 1, 10))[-request['count']:]
                return dict(history=dict(times=epochs, prices=[server.price(epoch) for epoch in epochs]))
            def close(self):
                pass
        return Client()


def test_rejected_bootstrap_recovers_unseen_history_before_executor_starts(model_only, monkeypatch, tmp_path):
    server = HistoricalComplement()
    provider = TrainingHistory(tmp_path / 'history', timeframe=300, duration=900,
                               now=lambda: server.base + server.candles * 300,
                               client_factory=server.client)
    records = server.records(0)
    preparation = dict(symbol='FIRST', excluded_epochs=[])
    before = pipeline.dataset_builder.training_diagnostics_from_records(records, preparation, 'FIRST')
    assert before['stage'] == 'ready'
    provider._save(provider._folder('FIRST'), 'FIRST', records,
                   dict(excluded_epochs=[], retries={}))
    progress = pipeline._training_progress()
    progress.mark_attempt('FIRST', before['_label_ids'],
                          dict(eligible=False, rejection_reasons=['taxa de acerto abaixo do mínimo']))
    original_progress = progress.path.read_bytes()
    monkeypatch.setattr(pipeline, '_training_history_provider', lambda: provider)

    # O diagnóstico já é executado no bootstrap, sem existir bot ou bundle ativo.
    result = pipeline._diagnose_training_symbol('FIRST')

    assert server.calls, 'Amostra antiga pronta não pode interromper o complemento após reprovação'
    assert result['new_samples'] >= 120 and result['ready_for_training']
    assert result['remaining_samples'] == 0
    assert result['history_recovery']['stop_reason'] == 'ready'
    assert all('buy' not in call and 'subscribe' not in call for call in server.calls)
    assert progress.path.read_bytes() == original_progress
    assert progress.observe('FIRST', before['_label_ids'])['new_samples'] == 0


@pytest.mark.parametrize('samples,reason,fresh,needed', [
    (1323, 'taxa de acerto abaixo do mínimo', 0, 120),
    (1323, 'taxa de acerto abaixo do mínimo', 119, 120),
    (1323, 'taxa de acerto abaixo do mínimo', 120, 120),
    (818, 'amostra final insuficiente (9 < 30 contratos)', 40, 41),
    (818, 'amostra final insuficiente (9 < 30 contratos)', 41, 41),
])
def test_recovery_respects_new_ids_and_retry_policy_in_constant_window(
        model_only, monkeypatch, tmp_path, samples, reason, fresh, needed):
    from training_history import RecoveryBudgetExhausted
    def labels(start):
        return [(1_800_000_000 + i * 300, 1_800_000_000 + i * 300,
                 1_800_000_900 + i * 300, i % 2) for i in range(start, start + samples)]
    progress = pipeline._training_progress()
    progress.mark_attempt('FIRST', labels(0), dict(eligible=False, rejection_reasons=[reason]))
    saved = progress.path.read_bytes()
    # Reinstanciar comprova que o complemento usa o progresso persistido, inclusive
    # quando a janela mantém a quantidade total e perde IDs antigos.
    from training_policy import TrainingProgress
    monkeypatch.setattr(pipeline, '_training_progress',
                        lambda: TrainingProgress(progress.path, identity=progress.identity))
    available = [fresh]
    monkeypatch.setattr(pipeline.dataset_builder, 'training_diagnostics_from_records',
                        lambda *_args: dict(stage='ready', labelable_contracts=samples,
                                           _label_ids=labels(available[0])))
    provider = TrainingHistory(tmp_path / 'history', timeframe=300, now=lambda: 1_800_001_000,
                               client_factory=lambda **_kwargs: SimpleNamespace(
                                   stats={'requests': 0}, close=lambda: None,
                                   request=Mock(side_effect=RecoveryBudgetExhausted('zero'))))
    provider._save(provider._folder('FIRST'), 'FIRST', [tick()], dict(excluded_epochs=[], retries={}))
    provider.enrich_incremental = Mock(return_value=([tick()], dict(stage='partial', stop_reason='no_progress')))
    pipeline._training_enricher(provider, 'FIRST')([], dict(symbol='FIRST', excluded_epochs=[]))

    assert provider.enrich_incremental.call_count == (1 if fresh < needed else 0)
    if fresh < needed:
        ready = provider.enrich_incremental.call_args.kwargs['ready']
        assert not ready([tick()], {})
        available[0] = needed
        assert ready([tick()], {})
    assert progress.path.read_bytes() == saved


def test_recovery_never_uses_another_symbols_attempt(model_only, monkeypatch, tmp_path):
    from training_policy import TrainingProgress
    progress = TrainingProgress(tmp_path / 'progress.json')
    progress.mark_attempt('OTHER', [(1_800_000_000, 1_800_000_000, 1_800_000_900, 1)],
                          dict(eligible=False, rejection_reasons=['desempenho insuficiente']))
    monkeypatch.setattr(pipeline, '_training_progress', lambda: progress)
    provider = TrainingHistory(tmp_path / 'history', timeframe=300, now=lambda: 1_800_001_000)
    provider._save(provider._folder('FIRST'), 'FIRST', [tick()], dict(excluded_epochs=[], retries={}))
    provider.enrich_incremental = Mock(side_effect=AssertionError('FIRST ainda não teve tentativa'))
    monkeypatch.setattr(pipeline.dataset_builder, 'training_diagnostics_from_records',
                        lambda *_args: dict(stage='ready', labelable_contracts=1000, _label_ids=[]))
    rows, report = pipeline._training_enricher(provider, 'FIRST')([], dict(symbol='FIRST', excluded_epochs=[]))
    assert rows and report['stop_reason'] == 'ready'
    provider.enrich_incremental.assert_not_called()
