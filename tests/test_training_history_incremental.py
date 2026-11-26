"""Bootstrap público da IA simulado, sem rede ou artefatos operacionais."""
import csv
import fcntl
import json
import sys

import pytest

import training_data
import training_history
from market_data import FIELDS
from training_data import TrainingDataError
from training_history import TrainingHistory


BASE = 1_800_000_000


def tick(epoch=BASE, price=101):
    return dict(epoch=epoch, datetime='', symbol='R_100', price=price,
                kind='tick', timeframe=0, open='', high='', low='')


def candle(epoch=BASE, timeframe=900):
    return dict(epoch=epoch, datetime='', symbol='R_100', price=101,
                kind='candle', timeframe=timeframe, open=100, high=102, low=99)


def preparation():
    return dict(symbol='R_100', excluded_epochs=[])


@pytest.fixture(autouse=True)
def no_public_network(monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail('o teste tentou abrir conexão pública')
    monkeypatch.setattr(training_history.websocket, 'create_connection', blocked)


def seed_cache(root, *, timeframe=900, rows=None, excluded=()):
    worker = TrainingHistory(root, timeframe=timeframe, now=lambda: BASE + 1800,
                             max_requests=0)
    worker._save(worker._folder('R_100'), 'R_100', rows or [tick(), candle(timeframe=timeframe)],
                 dict(excluded_epochs=list(excluded), retries={}))
    return worker


def test_five_minute_cache_reuses_only_verified_ticks_and_preserves_sibling(tmp_path):
    root = tmp_path / 'history'
    sibling = seed_cache(root, rows=[tick(), tick(BASE + 1), candle()], excluded=[BASE + 1])
    folder = sibling._folder('R_100')
    before = {path.name: path.read_bytes() for path in folder.iterdir()}
    worker = TrainingHistory(root, timeframe=300, max_requests=0, now=lambda: BASE + 1800)
    report = preparation()
    records, _ = worker.enrich([], report)
    assert records == [tick()]
    assert report['excluded_epochs'] == [BASE + 1]
    assert {path.name: path.read_bytes() for path in folder.iterdir()} == before
    cached, state = worker.cached_records('R_100')
    assert cached == records and state['timeframe'] == 300


def test_default_fifteen_minute_cache_does_not_import_five_minute_ticks(tmp_path):
    root = tmp_path / 'history'
    seed_cache(root, timeframe=300)
    worker = TrainingHistory(root, max_requests=0, now=lambda: BASE + 1800)
    records, _ = worker.enrich([], preparation())
    assert records == []


def test_reused_tick_conflict_preserves_temporal_veto(tmp_path):
    root = tmp_path / 'history'
    seed_cache(root)
    worker = TrainingHistory(root, timeframe=300, max_requests=0, now=lambda: BASE + 1800)
    report = preparation()
    records, _ = worker.enrich([tick(price=99)], report)
    assert records == []
    assert report['excluded_epochs'] == [BASE]


def test_tampered_sibling_is_not_imported(tmp_path):
    root = tmp_path / 'history'
    sibling = seed_cache(root)
    with (sibling._folder('R_100') / 'records.csv').open('a') as handle:
        handle.write('\n')
    worker = TrainingHistory(root, timeframe=300, max_requests=0)
    with pytest.raises(TrainingDataError, match='alterado'):
        worker.enrich([], preparation())
    assert not worker._folder('R_100').exists()


def test_cached_reader_never_creates_missing_root(tmp_path):
    worker = TrainingHistory(tmp_path / 'missing')
    assert worker.cached_records('R_100') == ([], dict(excluded_epochs=[], retries={}))
    assert not worker.root.exists()


def test_cached_reader_busy_is_explicit_and_does_not_change_files(tmp_path):
    worker = seed_cache(tmp_path / 'history')
    folder = worker._folder('R_100')
    before = {path.name: path.read_bytes() for path in folder.iterdir()}
    with (worker.root / '.public-history.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            worker.cached_records('R_100')
    assert {path.name: path.read_bytes() for path in folder.iterdir()} == before


def test_busy_recovery_preserves_verified_cache_and_new_temporal_veto(tmp_path):
    worker = seed_cache(tmp_path / 'history', rows=[tick(), tick(BASE + 1)], excluded=[BASE])
    # Também cobre uma instância que aqueceu o snapshot pela API pública local.
    reader = TrainingHistory(worker.root, now=lambda: BASE + 1800)
    reader.cached_records('R_100')
    report = preparation()
    with (worker.root / '.public-history.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows, status = reader.enrich([tick(BASE + 1, price=99), tick(BASE + 2)], report)
    assert rows == [tick(BASE + 2)]
    assert report['excluded_epochs'] == [BASE, BASE + 1]
    assert status['stage'] == 'busy' and status['cache_preserved']
    assert status['requests'] == 0


def test_busy_new_instance_never_reads_unlocked_cache(tmp_path):
    seeded = seed_cache(tmp_path / 'history')
    reader = TrainingHistory(seeded.root)
    with (seeded.root / '.public-history.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows, status = reader.enrich([tick()], preparation())
    assert rows == [tick()] and not status['cache_preserved']


def test_cached_reader_enforces_hash_and_temporal_veto(tmp_path):
    worker = seed_cache(tmp_path / 'history', rows=[tick(), tick(BASE + 1)], excluded=[BASE])
    records, state = worker.cached_records('R_100')
    assert records == [tick(BASE + 1)] and state['excluded_epochs'] == [BASE]
    (worker._folder('R_100') / 'records.csv').write_text('changed')
    with pytest.raises(TrainingDataError, match='alterado'):
        worker.cached_records('R_100')


def fake_rounds(monkeypatch, *, stages=(), gains=(), requests=2, cancelled=None):
    calls = []

    def enrich(worker, records, report):
        index = len(calls)
        calls.append(dict(max_requests=worker.max_requests, seconds=worker.seconds,
                          timeframe=worker.timeframe, duration=worker.duration))
        stage = stages[index] if index < len(stages) else 'partial'
        gain = gains[index] if index < len(gains) else 1
        records = [*records, *(tick(BASE + 100 + index * 10 + i) for i in range(gain))]
        if cancelled is not None:
            cancelled[0] = True
        return records, dict(stage=stage, requests=min(requests, worker.max_requests),
                             ticks_added=gain, max_latency_sec=0.1)

    monkeypatch.setattr(TrainingHistory, 'enrich', enrich)
    return calls


def test_incremental_rounds_share_request_budget_and_keep_progress(tmp_path, monkeypatch):
    calls = fake_rounds(monkeypatch)
    worker = TrainingHistory(tmp_path, timeframe=300, max_requests=2)
    records, report = worker.enrich_incremental([], preparation(), rounds=10, total_requests=5)
    assert [call['max_requests'] for call in calls] == [2, 2, 1]
    assert all(call['timeframe'] == 300 and call['duration'] == 900 for call in calls)
    assert len(records) == 3
    assert report['requests'] == 5 and report['rounds_completed'] == 3
    assert report['stop_reason'] == 'request_budget'


def test_incremental_rounds_stop_on_no_new_verified_records(tmp_path, monkeypatch):
    calls = fake_rounds(monkeypatch, gains=[1, 0, 1])
    _, report = TrainingHistory(tmp_path).enrich_incremental([], preparation(), rounds=10)
    assert len(calls) == 2 and report['stop_reason'] == 'no_progress'
    assert report['records_added'] == 1


@pytest.mark.parametrize('stage', ['busy', 'backoff', 'unavailable'])
def test_incremental_rounds_stop_on_unavailable_recovery(tmp_path, monkeypatch, stage):
    calls = fake_rounds(monkeypatch, stages=[stage])
    _, report = TrainingHistory(tmp_path).enrich_incremental([], preparation(), rounds=10)
    assert len(calls) == 1 and report['stop_reason'] == stage


def test_incremental_rounds_stop_when_readiness_is_satisfied(tmp_path, monkeypatch):
    calls = fake_rounds(monkeypatch)
    _, report = TrainingHistory(tmp_path).enrich_incremental(
        [], preparation(), rounds=10, ready=lambda rows, report: len(rows) >= 2)
    assert len(calls) == 2 and report['stop_reason'] == 'ready'


def test_incremental_rounds_cancel_before_or_between_batches(tmp_path, monkeypatch):
    cancelled = [True]
    calls = fake_rounds(monkeypatch, cancelled=cancelled)
    worker = TrainingHistory(tmp_path, cancelled=lambda: cancelled[0])
    _, report = worker.enrich_incremental([], preparation(), rounds=10)
    assert calls == [] and report['stop_reason'] == 'cancelled'
    cancelled[0] = False
    rows, report = worker.enrich_incremental([], preparation(), rounds=10)
    assert len(calls) == 1 and len(rows) == 1 and report['stop_reason'] == 'cancelled'


def test_incremental_deadline_limits_each_batch_and_stops_without_sleep(tmp_path, monkeypatch):
    calls, elapsed = [], [1]
    def batch(worker, rows, report):
        calls.append(worker.seconds)
        elapsed[0] += 5
        return [*rows, tick(BASE + len(calls))], dict(stage='partial', requests=1)
    monkeypatch.setattr(TrainingHistory, 'enrich', batch)
    worker = TrainingHistory(tmp_path, seconds=15)
    _, report = worker.enrich_incremental([], preparation(), rounds=10, total_seconds=10,
                                         clock=lambda: elapsed[0])
    assert calls == [10, 5]
    assert report['stop_reason'] == 'time_budget'


def test_incremental_deadline_accounts_for_cache_preparation_before_socket(tmp_path, monkeypatch):
    elapsed, allotted = [0], []
    def client(**kwargs):
        allotted.append(kwargs['seconds'])
    def batch(worker, rows, report):
        elapsed[0] += 8
        worker.client_factory(max_requests=worker.max_requests, seconds=worker.seconds)
        assert not worker.cancelled()
        elapsed[0] += 2
        assert worker.cancelled()
        return rows, dict(stage='partial', requests=0)
    monkeypatch.setattr(TrainingHistory, 'enrich', batch)
    _, report = TrainingHistory(tmp_path, client_factory=client).enrich_incremental(
        [], preparation(), rounds=4, total_seconds=10, clock=lambda: elapsed[0])
    assert allotted == [2] and report['stop_reason'] == 'time_budget'


def test_incremental_counts_unique_records_once_across_repeated_pages(tmp_path, monkeypatch):
    calls = []
    def batch(worker, rows, report):
        calls.append(True)
        return [tick()], dict(stage='partial', requests=1)
    monkeypatch.setattr(TrainingHistory, 'enrich', batch)
    _, report = TrainingHistory(tmp_path).enrich_incremental([], preparation(), rounds=4)
    assert len(calls) == 2
    assert report['records_added'] == 1 and report['stop_reason'] == 'no_progress'


def test_recent_recovery_hint_limits_queries_without_trimming_old_cache(tmp_path):
    now = BASE + 200 * 300
    old = tick(BASE - 10 * 86400)
    worker = TrainingHistory(tmp_path / 'history', timeframe=300, now=lambda: now, max_requests=3)
    worker._save(worker._folder('R_100'), 'R_100', [old],
                 dict(excluded_epochs=[], retries={}))
    queries = []
    class Client:
        stats = dict(requests=0, timeouts=0, request_errors=0, max_latency_sec=0)
        def request(self, request):
            if self.stats['requests'] >= worker.max_requests:
                raise training_history.RecoveryBudgetExhausted()
            self.stats['requests'] += 1
            queries.append(request)
            if request['style'] == 'candles':
                return dict(candles=[dict(epoch=e, open=100, high=102, low=99, close=101)
                    for e in range(request['start'], int(request['end']) + 1, 300)])
            return dict(history=dict(times=[], prices=[]))
        def close(self):
            pass
    worker.client_factory = lambda **kwargs: Client()
    report = dict(preparation(), recovery_start=now - 6 * 3600 + 1)
    rows, status = worker.enrich([], report)
    assert old in rows
    assert queries and all(query['start'] >= now - 6 * 3600 for query in queries)
    assert status['recovery_start'] == now - 6 * 3600
    assert old in worker.cached_records('R_100')[0]


@pytest.mark.parametrize('arguments', [dict(rounds=0), dict(total_requests=-1),
                                      dict(total_seconds=0), dict(total_seconds=float('nan'))])
def test_incremental_invalid_global_budgets_fail_before_recovery(tmp_path, monkeypatch, arguments):
    calls = fake_rounds(monkeypatch)
    with pytest.raises(ValueError):
        TrainingHistory(tmp_path).enrich_incremental([], preparation(), **arguments)
    assert calls == []


@pytest.mark.parametrize('extra, expected_method', [([], 'single'),
    (['--timeframe', '300', '--duration', '900', '--rounds', '5',
      '--total-requests', '40', '--total-seconds', '50'], 'incremental')])
def test_cli_preserves_default_single_batch_and_exposes_explicit_bootstrap(
        tmp_path, monkeypatch, capsys, extra, expected_method):
    calls = []

    class Provider:
        def __init__(self, root, **kwargs):
            calls.append(('init', kwargs))

        def enrich(self, rows, report):
            calls.append(('single', {}))
            return rows, dict(stage='partial', requests=1)

        def enrich_incremental(self, rows, report, **kwargs):
            calls.append(('incremental', kwargs))
            return rows, dict(stage='partial', requests=2)

    def prepare(source, destination, symbol, *, history_path, enricher):
        _, result = enricher([], preparation())
        return tmp_path / 'prices.csv', dict(history_recovery=result)

    monkeypatch.setattr(training_history, 'TrainingHistory', Provider)
    monkeypatch.setattr(training_data, 'prepare_snapshot', prepare)
    monkeypatch.setattr(sys, 'argv', ['training_history.py', '--input', 'unused.csv',
        '--symbol', 'R_100', '--cache-dir', str(tmp_path / 'cache'),
        '--output-dir', str(tmp_path / 'output'), *extra])
    training_history.main()
    result = json.loads(capsys.readouterr().out)
    assert set(result) == {'snapshot', 'history_recovery'}
    assert calls[1][0] == expected_method
    assert calls[0][1]['timeframe'] == (300 if extra else 900)
    if extra:
        assert calls[1][1] == dict(rounds=5, total_requests=40, total_seconds=50.0)


def test_malformed_epoch_still_blocks_preparation_before_history(tmp_path, monkeypatch):
    source = tmp_path / 'ticks.csv'
    with source.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        writer.writerow(['unsafe', '', 'R_100', 101, 'tick', 0, '', '', ''])
    calls = fake_rounds(monkeypatch)
    worker = TrainingHistory(tmp_path / 'cache', timeframe=300)
    with pytest.raises(TrainingDataError, match='sem época segura'):
        training_data.prepare_snapshot(source, tmp_path / 'out', 'R_100',
                                       enricher=worker.enrich_incremental)
    assert calls == [] and not worker.root.exists()
