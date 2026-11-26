"""Recuperação incremental da IA com API pública inteiramente simulada."""
import csv
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

import dataset_builder
from market_data import FIELDS, read_records
from training_data import TrainingDataError, prepare_snapshot
from training_history import (PUBLIC_URL, HistoryAPIError, PublicHistoryClient,
                              RecoveryBudgetExhausted, TrainingHistory,
                              _candle_page, _missing_ticks, _tick_page, feed_diagnostics)

BASE, TF = 1_800_000_000, 900


def candle(epoch):
    return dict(epoch=epoch, open=100, high=102, low=99, close=101)


def row_tick(epoch, price=101):
    return dict(epoch=epoch, datetime='', symbol='R_100', price=price,
                kind='tick', timeframe=0, open='', high='', low='')


def row_candle(epoch):
    return dict(epoch=epoch, datetime='', symbol='R_100', price=101,
                kind='candle', timeframe=TF, open=100, high=102, low=99)


def seed(tmp_path):
    """Histórico acaba às 12h30 da série; ticks começam 703s depois desse sinal."""
    source, history = tmp_path / 'ticks.csv', tmp_path / 'ohlc.csv'
    with source.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, FIELDS)
        writer.writeheader()
        writer.writerows(row_tick(e) for e in range(BASE + 50 * TF + 703, BASE + 51 * TF, 2))
    with history.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, FIELDS)
        writer.writeheader()
        writer.writerows(row_candle(BASE + i * TF) for i in range(50))
    return source, history


class Server:
    def __init__(self, *, end=BASE + 65 * TF + 100, step=10, error=None, empty=False):
        self.candles = [candle(BASE + i * TF) for i in range(65)]
        self.ticks = list(range(BASE + 50 * TF, end, step))
        self.error, self.empty = error, empty
        self.calls, self.clients = [], []

    def client(self, **kwargs):
        server = self

        class Client:
            def __init__(self):
                self.stats = dict(requests=0, timeouts=0, request_errors=0, max_latency_sec=0.0)
                self.closed = False

            def request(self, request):
                if self.stats['requests'] >= kwargs['max_requests']:
                    raise RecoveryBudgetExhausted('limite simulado')
                self.stats['requests'] += 1
                server.calls.append(request)
                if server.error:
                    raise server.error
                start, end = request['start'], int(request['end'])
                if request['style'] == 'candles':
                    return dict(candles=[] if server.empty else
                                [c for c in server.candles if start <= c['epoch'] <= end][-request['count']:])
                ticks = [] if server.empty else [e for e in server.ticks if start <= e <= end][-request['count']:]
                return dict(history=dict(times=ticks, prices=[101] * len(ticks)))

            def close(self):
                self.closed = True

        result = Client()
        self.clients.append(result)
        return result


def provider(tmp_path, server, *, requests=12, now=BASE + 65 * TF + 100):
    return TrainingHistory(tmp_path / 'ai-cache', now=lambda: now,
                           max_requests=requests, client_factory=server.client)


def test_recovery_closes_startup_gap_without_touching_operational_sources(tmp_path):
    source, history = seed(tmp_path)
    before = source.read_bytes(), history.read_bytes()
    original, _ = prepare_snapshot(source, tmp_path / 'before', 'R_100', history_path=history)
    assert dataset_builder.count_labelable_contracts(original, 'R_100') == 0
    server = Server()
    worker = provider(tmp_path, server)
    prepared, report = prepare_snapshot(source, tmp_path / 'after', 'R_100', history_path=history,
                                        enricher=worker.enrich)
    count = dataset_builder.count_labelable_contracts(prepared, 'R_100')
    assert count > 0
    assert report['history_recovery']['ticks_added'] > 0
    assert report['history_recovery']['candles_added'] > 0
    assert all('subscribe' not in request and 'buy' not in request for request in server.calls)
    assert (source.read_bytes(), history.read_bytes()) == before
    assert read_records(prepared)[1] == []
    assert all(client.closed for client in server.clients)


def test_tick_pagination_is_backward_bounded_and_resumable(tmp_path):
    source, history = seed(tmp_path)
    server = Server(step=1)
    worker = provider(tmp_path, server, requests=4)
    prepare_snapshot(source, tmp_path / 'first', 'R_100', history_path=history, enricher=worker.enrich)
    first_calls = [r for r in server.calls if r['style'] == 'ticks']
    assert first_calls
    previous_start = min(int(r['end']) for r in first_calls)
    server.calls.clear()
    # Outro objeto simula reinício: o progresso vem do cache em disco.
    worker = provider(tmp_path, server, requests=12)
    prepared, _ = prepare_snapshot(source, tmp_path / 'second', 'R_100', history_path=history,
                                   enricher=worker.enrich)
    tick_calls = [r for r in server.calls if r['style'] == 'ticks']
    assert tick_calls and int(tick_calls[0]['end']) < previous_start
    assert all(r['start'] <= int(r['end']) for r in server.calls)
    assert all(r['count'] == 5000 for r in tick_calls)
    assert dataset_builder.count_labelable_contracts(prepared, 'R_100') > 0


def test_complete_known_tick_intervals_are_not_requested_again(tmp_path):
    source, history = seed(tmp_path)
    server = Server()
    worker = provider(tmp_path, server)
    prepare_snapshot(source, tmp_path / 'one', 'R_100', history_path=history, enricher=worker.enrich)
    server.calls.clear()
    prepare_snapshot(source, tmp_path / 'two', 'R_100', history_path=history, enricher=worker.enrich)
    assert not [r for r in server.calls if r['style'] == 'ticks']


def test_empty_response_does_not_become_coverage_and_is_cooled_down(tmp_path):
    source, history = seed(tmp_path)
    server = Server(empty=True)
    worker = provider(tmp_path, server)
    prepared, report = prepare_snapshot(source, tmp_path / 'one', 'R_100', history_path=history,
                                        enricher=worker.enrich)
    assert report['history_recovery']['stage'] == 'partial'
    assert report['history_recovery']['missing_tick_ranges'] > 0
    assert dataset_builder.count_labelable_contracts(prepared, 'R_100') == 0
    server.calls.clear()
    prepare_snapshot(source, tmp_path / 'two', 'R_100', history_path=history, enricher=worker.enrich)
    assert server.calls == []


def test_rate_limit_backoff_is_shared_between_symbols(tmp_path):
    source, history = seed(tmp_path)
    server = Server(error=HistoryAPIError('RateLimit', retry_after=120))
    worker = provider(tmp_path, server)
    _, report = prepare_snapshot(source, tmp_path / 'one', 'R_100', history_path=history,
                                 enricher=worker.enrich)
    assert report['history_recovery']['stage'] == 'unavailable'
    server.calls.clear()
    rows, status = worker.enrich([], dict(symbol='R_50', excluded_epochs=[]))
    assert rows == [] and status['stage'] == 'backoff'
    assert server.calls == []


def test_malformed_page_preserves_valid_cache_and_reports_error(tmp_path):
    source, history = seed(tmp_path)
    server = Server()
    worker = provider(tmp_path, server)
    prepared, _ = prepare_snapshot(source, tmp_path / 'one', 'R_100', history_path=history,
                                   enricher=worker.enrich)
    count = dataset_builder.count_labelable_contracts(prepared, 'R_100')
    server.error = ValueError('bad response wss://private/session?otp=hidden')
    # Avança o relógio para expirar o cooldown de intervalos vazios.
    worker = provider(tmp_path, server, now=BASE + 65 * TF + 500)
    result, report = prepare_snapshot(source, tmp_path / 'two', 'R_100', history_path=history,
                                      enricher=worker.enrich)
    assert dataset_builder.count_labelable_contracts(result, 'R_100') == count
    recovery = report['history_recovery']
    assert recovery['stage'] == 'unavailable'
    assert 'hidden' not in json.dumps(recovery)


def test_cache_tampering_fails_closed(tmp_path):
    source, history = seed(tmp_path)
    server = Server()
    worker = provider(tmp_path, server)
    prepare_snapshot(source, tmp_path / 'one', 'R_100', history_path=history, enricher=worker.enrich)
    cache = next((tmp_path / 'ai-cache').glob('*/records.csv'))
    with cache.open('a') as handle:
        handle.write('\n')
    with pytest.raises(TrainingDataError, match='alterado'):
        prepare_snapshot(source, tmp_path / 'two', 'R_100', enricher=worker.enrich)


def test_quarantine_remains_a_veto_after_history_recovery(tmp_path):
    source, history = seed(tmp_path)
    with source.open('a', newline='') as handle:
        csv.writer(handle).writerow([BASE + 50 * TF + 100, '', 'R_100', 'nan', 'tick', 0, '', '', ''])
    worker = provider(tmp_path, Server())
    prepared, report = prepare_snapshot(source, tmp_path / 'one', 'R_100', history_path=history,
                                        enricher=worker.enrich)
    assert BASE + 50 * TF + 100 in report['excluded_epochs']
    series = dataset_builder._load_training_series(prepared, 'R_100')
    assert dataset_builder._excluded_window(series[-1], BASE + 50 * TF, BASE + 51 * TF)


@pytest.mark.parametrize('data', [dict(history={}), dict(history=dict(times=[BASE], prices=[])),
                                 dict(history=dict(times=[BASE], prices=['nan']))])
def test_invalid_tick_payload_is_rejected(data):
    with pytest.raises(ValueError):
        _tick_page(data, 'R_100', BASE, BASE + 100)


def test_out_of_range_and_future_ohlc_are_never_used():
    response = dict(candles=[candle(BASE - TF), candle(BASE), candle(BASE + TF)])
    rows = _candle_page(response, 'R_100', BASE, BASE + 2 * TF, TF, BASE + TF)
    assert [r['epoch'] for r in rows] == [BASE]
    assert _tick_page(dict(history=dict(times=[BASE - 1, BASE, BASE + 2], prices=[100, 101, 102])),
                      'R_100', BASE, BASE + 1) == [row_tick(BASE)]


def test_partial_page_does_not_fabricate_intermediate_ticks(tmp_path):
    source, history = seed(tmp_path)
    server = Server(step=30)
    prepared, report = prepare_snapshot(source, tmp_path / 'one', 'R_100', history_path=history,
                                        enricher=provider(tmp_path, server).enrich)
    assert report['history_recovery']['missing_tick_ranges'] > 0
    assert dataset_builder.count_labelable_contracts(prepared, 'R_100') == 0


def test_cancelled_recovery_does_not_send_requests(tmp_path):
    source, history = seed(tmp_path)
    server = Server()
    worker = provider(tmp_path, server)
    worker.cancelled = lambda: True
    prepare_snapshot(source, tmp_path / 'one', 'R_100', history_path=history, enricher=worker.enrich)
    assert server.calls == []


class Socket:
    def __init__(self, replies):
        self.replies, self.sent, self.timeouts = list(replies), [], []
        self.closed = False

    def settimeout(self, timeout):
        self.timeouts.append(timeout)

    def send(self, raw):
        self.sent.append(json.loads(raw))

    def recv(self):
        if not self.replies:
            raise TimeoutError('timeout simulado')
        return json.dumps(self.replies.pop(0))

    def close(self):
        self.closed = True


def test_public_transport_correlates_responses_and_never_authenticates():
    sock = Socket([dict(req_id=999, msg_type='history'),
                   dict(req_id=1, msg_type='history', history=dict(times=[], prices=[]))])
    connect = Mock(return_value=sock)
    client = PublicHistoryClient(connect=connect, interval=0)
    request = dict(ticks_history='R_100', start=BASE, end=str(BASE + 100), count=5000, style='ticks')
    assert client.request(request)['req_id'] == 1
    assert connect.call_args.args == (PUBLIC_URL,)
    assert sock.sent == [dict(request, req_id=1)]
    assert all(0 < timeout <= 5 for timeout in sock.timeouts)
    client.close()
    assert sock.closed


@pytest.mark.parametrize('payload', [dict(buy='contract'),
                                    dict(ticks_history='R_100', style='ticks', subscribe=1),
                                    dict(authorize='anything')])
def test_public_client_rejects_trading_authentication_and_subscriptions(payload):
    connect = Mock()
    client = PublicHistoryClient(connect=connect)
    with pytest.raises(ValueError, match='somente histórico público'):
        client.request(payload)
    connect.assert_not_called()


def test_public_transport_timeout_is_diagnosed():
    sock = Socket([])
    client = PublicHistoryClient(connect=lambda *a, **kw: sock, interval=0)
    with pytest.raises(TimeoutError):
        client.request(dict(ticks_history='R_100', start=BASE, end=str(BASE + 100), count=1, style='ticks'))
    assert client.stats['timeouts'] == 1 and client.stats['request_errors'] == 1
    client.close()


def test_public_transport_budget_prevents_unbounded_downloads():
    client = PublicHistoryClient(max_requests=0, connect=Mock())
    with pytest.raises(RecoveryBudgetExhausted):
        client.request(dict(ticks_history='R_100', style='ticks'))
    client.connect.assert_not_called()


def test_missing_ticks_do_not_include_already_covered_intervals():
    assert _missing_ticks(list(range(100, 201)), 100, 200) == []
    assert _missing_ticks([100, 110, 150, 160, 200], 100, 200) == [(111, 149), (161, 199)]


def test_feed_diagnosis_reads_only_current_session_and_preserves_log(tmp_path):
    log = tmp_path / 'pipeline.log'
    line = '[COLETOR] Erro WebSocket: horário={} causa=ping/pong timed out\n'
    log.write_text(line.format('2026-10-28T13:00:00+00:00') + line.format('2026-10-28T14:00:00+00:00'))
    before = log.read_bytes()
    result = feed_diagnostics(log, '2026-10-28T13:30:00+00:00')
    assert result['ping_timeouts'] == 1
    assert result['scope'] == 'last_256_kb'
    assert log.read_bytes() == before


def test_feed_diagnosis_separates_executor_and_collector_timeouts(tmp_path):
    log = tmp_path / 'pipeline.log'
    log.write_text('[BOT] Erro WebSocket: horário=2026-10-28T13:00:00+00:00 causa=ping/pong timed out\n'
                   '[BOT] Erro WebSocket: horário=2026-10-28T14:00:00+00:00 causa=ping/pong timed out\n'
                   '[COLETOR] Erro WebSocket: horário=2026-10-28T14:00:01+00:00 causa=ping/pong timed out\n')
    result = feed_diagnostics(log, '2026-10-28T13:30:00+00:00')
    assert result['ping_timeouts'] == 1
    assert result['executor_ping_timeouts'] == 1
    assert result['executor_last_timeout_at'] == '2026-10-28T14:00:00+00:00'


def test_pipeline_readiness_uses_recovered_history_without_trading(tmp_path, monkeypatch):
    import pipeline
    from asset_history import AssetHistoryStore
    source, history = seed(tmp_path)
    store = AssetHistoryStore(tmp_path / 'hist', TF)
    store.root.mkdir()
    store.path_for('R_100').write_bytes(history.read_bytes())
    server = Server()
    worker = provider(tmp_path, server)
    monkeypatch.setattr(pipeline, 'TICKS_CSV', str(source))
    monkeypatch.setattr(pipeline, '_runtime_symbol', lambda: 'R_100')
    monkeypatch.setattr(pipeline, '_asset_history_store', store)
    monkeypatch.setattr(pipeline, '_training_readiness_cache', {})
    monkeypatch.setattr(pipeline, '_training_history_provider', lambda: worker)
    assert pipeline._count_labelable_contracts_for_training() > 0
    status = json.loads(pipeline._BOT_STATUS_JSON.read_text())
    assert status['training']['history_recovery']['ticks_added'] > 0
    assert not (tmp_path / 'model.pkl').exists()


def test_empty_recent_gap_does_not_starve_other_ranges(tmp_path, monkeypatch):
    import training_history
    source, history = seed(tmp_path)
    server = Server(empty=True)
    recovery = provider(tmp_path, server, requests=1)
    monkeypatch.setattr(training_history, '_ranges', lambda *args: [])
    monkeypatch.setattr(training_history, '_label_ranges',
                        lambda *args: [(BASE, BASE + 100), (BASE + 1000, BASE + 1100)])
    for index in range(2):
        prepare_snapshot(source, tmp_path / f'prepared-{index}', 'R_100',
                         history_path=history, enricher=recovery.enrich)
    queries = [q for q in server.calls if q['style'] == 'ticks']
    assert len(queries) == 2
    assert queries[0]['start'] != queries[1]['start']
