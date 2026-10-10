"""Backfill paralelo de ticks, gravação em lote e preferência por mercados não sintéticos."""
import json

import pytest

from market_data import append_records, canonicalize, read_records
from asset_history import AssetHistoryStore


def test_append_records_batches_ticks_under_v2_header(tmp_path):
    """Garante cabeçalho único, linhas de tick válidas e leitura sem conflitos."""
    path = tmp_path / 'ticks.csv'
    assert append_records(path, [(10, 'R_100', 1.5), (11, 'R_100', 1.6)]) == 2
    assert append_records(path, []) == 0
    records, rejected = read_records(path)
    _unique, conflicts, duplicates = canonicalize(records)
    assert [r['epoch'] for r in records] == [10, 11]
    assert not rejected and not conflicts and not duplicates
    assert all(r['kind'] == 'tick' for r in records)


def test_append_records_refuses_legacy_schema(tmp_path):
    """Garante que arquivos de esquema legado não recebem lotes."""
    path = tmp_path / 'legacy.csv'
    path.write_text('epoch,datetime,symbol,price\n')
    with pytest.raises(ValueError):
        append_records(path, [(1, 'R_100', 1.0)])


def test_missing_ranges_ignore_gaps_within_tolerance():
    """Garante que só lacunas maiores que MAX_TICK_GAP_SEC viram intervalos a baixar."""
    import pipeline
    gap = pipeline.MAX_TICK_GAP_SEC
    assert pipeline._missing_tick_ranges([], 100, 200) == [(100, 200)]
    covered = list(range(100, 201))
    assert pipeline._missing_tick_ranges(covered, 100, 200) == []
    assert pipeline._missing_tick_ranges([100 + gap, 150, 200 - gap], 100, 200) == [(111, 149), (151, 189)]
    assert pipeline._missing_tick_ranges([150], 100, 200) == [(100, 149), (151, 200)]


def test_split_ranges_cover_everything_without_overlap(monkeypatch):
    """Garante blocos contíguos, sem sobreposição, respeitando o tamanho mínimo."""
    import pipeline
    monkeypatch.setattr(pipeline, '_TICK_MIN_CHUNK_SEC', 10)
    chunks = pipeline._split_tick_ranges([(0, 99), (200, 209)], 4)
    covered = [e for start, end in chunks for e in range(start, end + 1)]
    assert covered == list(range(0, 100)) + list(range(200, 210))
    assert pipeline._split_tick_ranges([], 4) == []


class _FakeSocket:
    """Servidor simulado de ticks_history que devolve até `count` ticks terminando em `end`."""
    def __init__(self, epochs, fail=False, rate_limits=0):
        self.epochs, self.fail, self.rate_limits = epochs, fail, rate_limits
        self.queue, self.requests = [], []

    def send(self, payload):
        request = json.loads(payload)
        self.requests.append(request)
        if self.fail:
            self.queue.append({'error': {'code': 'Boom', 'message': 'falha'}})
        elif self.rate_limits:
            self.rate_limits -= 1
            self.queue.append({'error': {'code': 'RateLimit', 'message': 'limite'}})
        else:
            selected = [e for e in self.epochs if request['start'] <= e <= request['end']][-request['count']:]
            self.queue.append({'msg_type': 'history',
                               'history': {'times': selected, 'prices': [100.0] * len(selected)}})

    def recv(self):
        return json.dumps(self.queue.pop(0))

    def close(self):
        pass


def _patch_network(monkeypatch, pipeline, make_socket):
    monkeypatch.setattr(pipeline, '_uses_bearer_auth', lambda: True)
    monkeypatch.setattr(pipeline, '_get_ws_url', lambda: 'wss://teste')
    monkeypatch.setattr(pipeline.websocket, 'create_connection', lambda url, timeout=30: make_socket())
    monkeypatch.setattr(pipeline.time, 'sleep', lambda _s: None)


def test_parallel_fetch_pages_sorts_and_deduplicates(monkeypatch):
    """Garante paginação por janela, ordenação e ausência de duplicatas entre conexões."""
    import pipeline
    epochs = list(range(1_000, 1_000 + 12_000))
    monkeypatch.setattr(pipeline, '_TICK_MIN_CHUNK_SEC', 100)
    _patch_network(monkeypatch, pipeline, lambda: _FakeSocket(epochs))
    ticks = pipeline._fetch_ticks_parallel('R_100', 1_000, 1_000 + 11_999, [], workers=4)
    assert [e for e, _p in ticks] == epochs


def test_parallel_fetch_downloads_only_missing_ranges(monkeypatch):
    """Garante que ticks já conhecidos não são pedidos novamente."""
    import pipeline
    epochs = list(range(1_000, 2_000))
    sockets = []
    def make_socket():
        sockets.append(_FakeSocket(epochs))
        return sockets[-1]
    _patch_network(monkeypatch, pipeline, make_socket)
    known = list(range(1_000, 1_800))
    ticks = pipeline._fetch_ticks_parallel('R_100', 1_000, 1_999, known, workers=2)
    assert min(e for e, _p in ticks) > 1_790
    requested = [r for s in sockets for r in s.requests]
    assert requested and all(r['start'] > 1_790 for r in requested)


def test_parallel_fetch_retries_rate_limit(monkeypatch):
    """Garante retry com backoff quando a API responde RateLimit."""
    import pipeline
    _patch_network(monkeypatch, pipeline, lambda: _FakeSocket(list(range(100, 200)), rate_limits=2))
    ticks = pipeline._fetch_ticks_parallel('R_100', 100, 199, [], workers=1)
    assert len(ticks) == 100


def test_parallel_fetch_reports_failed_connection(monkeypatch):
    """Garante que falha numa conexão levanta erro para o coletor recorrer ao modo sequencial."""
    import pipeline
    _patch_network(monkeypatch, pipeline, lambda: _FakeSocket([], fail=True))
    with pytest.raises(RuntimeError):
        pipeline._fetch_ticks_parallel('R_100', 100, 199, [], workers=2)


def test_collector_parallel_backfill_writes_batch_and_signals_ready(tmp_path, monkeypatch):
    """Garante gravação em lote sem repetir ticks ao vivo e sinalização de histórico pronto."""
    import pipeline
    from market_data import ensure_header
    from unittest.mock import Mock
    path = tmp_path / 'ticks.csv'
    ensure_header(path)
    monkeypatch.setattr(pipeline, 'TICKS_CSV', str(path))
    collector = pipeline._CollectorThread('R_100')
    collector._history_seen = {1_005}
    monkeypatch.setattr(pipeline, '_fetch_ticks_parallel',
                        lambda *a, **k: [(e, 100.0) for e in range(1_000, 1_010)])
    collector._parallel_backfill(Mock(), 1_000, 1_009)
    records, _ = read_records(path)
    assert sorted(r['epoch'] for r in records) == [e for e in range(1_000, 1_010) if e != 1_005]
    assert collector.history_ready.is_set()


def test_collector_reconnect_preserves_completed_history_state(tmp_path, monkeypatch):
    """Garante que reconexao nao zera o historico ja recuperado."""
    import pipeline
    from market_data import ensure_header
    from unittest.mock import Mock
    path = tmp_path / 'ticks.csv'
    ensure_header(path)
    append_records(path, [(1_000, 'R_100', 100.0)])
    monkeypatch.setattr(pipeline, 'TICKS_CSV', str(path))
    monkeypatch.setattr(pipeline.threading, 'Timer', lambda *_a, **_k: Mock(start=Mock()))
    collector = pipeline._CollectorThread('R_100')
    collector._history_start, collector._history_end = 900, 999
    collector.history_ready.set()
    collector._on_open(Mock())
    assert collector.history_ready.is_set()
    assert collector._history_start == 900


def test_collector_does_not_start_duplicate_backfill(monkeypatch):
    """Garante que ticks durante backfill ativo nao criam workers duplicados."""
    import pipeline
    from unittest.mock import Mock
    collector = pipeline._CollectorThread('R_100')
    collector._history_start, collector._history_end = 1_000, 1_999
    starts = []
    class _Thread:
        def __init__(self, target, args=(), daemon=None, name=None):
            starts.append((target, args, daemon, name))
        def start(self):
            pass
    monkeypatch.setattr(pipeline, '_TICK_BACKFILL_WORKERS', 2)
    monkeypatch.setattr(pipeline.threading, 'Thread', _Thread)
    ws = Mock()
    collector._start_backfill(ws)
    collector._start_backfill(ws)
    assert len(starts) == 1


def test_collector_first_tick_timeout_suppresses_closed_socket_noise(capsys):
    """Garante que fechamento pelo watchdog de primeiro tick não vira erro WebSocket ruidoso."""
    import pipeline
    from unittest.mock import Mock

    collector = pipeline._CollectorThread('R_100')
    ws = Mock()
    collector._on_first_tick_timeout(ws)
    collector._on_error(ws, 'Connection is already closed.')
    collector._on_error(ws, "'NoneType' object has no attribute 'sock'")
    collector._on_close(ws, 1000, '')

    output = capsys.readouterr().out
    assert 'Primeiro tick não recebido em 30s' in output
    assert 'Erro WebSocket' not in output
    assert 'sem tick inicial' in output


def test_collector_falls_back_to_sequential_when_parallel_fails(tmp_path, monkeypatch):
    """Garante que a falha do download paralelo pede a página sequencial pelo WebSocket ao vivo."""
    import pipeline
    from unittest.mock import Mock
    monkeypatch.setattr(pipeline, 'TICKS_CSV', str(tmp_path / 'ticks.csv'))
    collector = pipeline._CollectorThread('R_100')
    collector._history_start, collector._history_end = 1_000, 1_999
    def boom(*_a, **_k):
        raise RuntimeError('rede')
    monkeypatch.setattr(pipeline, '_fetch_ticks_parallel', boom)
    ws = Mock()
    collector._parallel_backfill(ws, 1_000, 1_999)
    assert json.loads(ws.send.call_args.args[0])['ticks_history'] == 'R_100'
    assert not collector.history_ready.is_set()


def test_rate_limit_detected_by_message_and_throttle_backs_off(monkeypatch):
    """Garante retry quando só a mensagem indica rate limit e que o throttle recua para todos."""
    import pipeline

    class _MessageOnlyLimit(_FakeSocket):
        def send(self, payload):
            if self.rate_limits:
                self.rate_limits -= 1
                self.requests.append(json.loads(payload))
                self.queue.append({'error': {'code': 'Other',
                                             'message': 'You have reached the rate limit for ticks_history.'}})
            else:
                super().send(payload)

    _patch_network(monkeypatch, pipeline, lambda: _MessageOnlyLimit(list(range(100, 200)), rate_limits=2))
    assert len(pipeline._fetch_ticks_parallel('R_100', 100, 199, [], workers=1)) == 100
    throttle = pipeline._TickThrottle(interval=0.2)
    throttle.backoff()
    assert throttle._interval == 0.4


def test_parallel_failure_keeps_partial_ticks(monkeypatch):
    """Garante que ticks baixados por conexões saudáveis sobrevivem à falha de outra conexão."""
    import pipeline
    monkeypatch.setattr(pipeline, '_TICK_MIN_CHUNK_SEC', 100)
    sockets = []
    def make_socket():
        sockets.append(_FakeSocket(list(range(1_000, 1_400)), fail=len(sockets) == 1))
        return sockets[-1]
    _patch_network(monkeypatch, pipeline, make_socket)
    with pytest.raises(pipeline._PartialTickFetch) as caught:
        pipeline._fetch_ticks_parallel('R_100', 1_000, 1_399, [], workers=2)
    assert caught.value.ticks


def test_collector_repeats_only_missing_after_partial_fetch(tmp_path, monkeypatch):
    """Garante gravação do parcial e nova passada que conclui o histórico."""
    import pipeline
    from market_data import ensure_header
    from unittest.mock import Mock
    path = tmp_path / 'ticks.csv'
    ensure_header(path)
    monkeypatch.setattr(pipeline, 'TICKS_CSV', str(path))
    monkeypatch.setattr(pipeline, '_TICK_PASS_PAUSE_SEC', 0)
    collector = pipeline._CollectorThread('R_100')
    calls = []
    def fake_fetch(_symbol, _start, _end, known):
        calls.append(set(known))
        if len(calls) == 1:
            raise pipeline._PartialTickFetch('limite', [(e, 100.0) for e in range(1_000, 1_005)])
        return [(e, 100.0) for e in range(1_005, 1_010)]
    monkeypatch.setattr(pipeline, '_fetch_ticks_parallel', fake_fetch)
    collector._parallel_backfill(Mock(), 1_000, 1_009)
    assert calls[1] == set(range(1_000, 1_005))
    assert collector.history_ready.is_set()
    records, _ = read_records(path)
    assert sorted(r['epoch'] for r in records) == list(range(1_000, 1_010))


def test_collector_sequential_rate_limit_retries_later(monkeypatch):
    """Garante que rate limit durante o backfill agenda nova tentativa em vez de fechar a conexão."""
    import pipeline
    from unittest.mock import Mock
    timers = []
    class _Timer:
        def __init__(self, delay, fn, args=()):
            timers.append((delay, fn, args))
            self.daemon = False
        def start(self):
            pass
    monkeypatch.setattr(pipeline.threading, 'Timer', _Timer)
    collector = pipeline._CollectorThread('R_100')
    collector._history_start, collector._history_end = 1_000, 1_999
    ws = Mock()
    collector._on_message(ws, json.dumps({'error': {'code': 'RateLimit',
                                                    'message': 'You have reached the rate limit for ticks_history.'}}))
    ws.close.assert_not_called()
    assert timers and timers[0][1] == collector._request_tick_history


def _candles_between(start, end, timeframe=900):
    """Cria OHLC alinhado para cada epoch inclusivo do intervalo."""
    return [dict(epoch=epoch, open=100, high=101, low=99, close=100.5)
            for epoch in range(start, end + 1, timeframe)]


def test_asset_history_is_incremental_per_symbol_and_deduplicates(tmp_path):
    """Consulta o armazenamento antes da API, mantém um CSV por ativo e não repete epochs já cobertos."""
    store = AssetHistoryStore(tmp_path / 'Histórico Ativos', 900, max_candles=336)
    now = 10 * 86400
    calls = []

    def fetch(start, end):
        calls.append((start, end))
        candles = _candles_between(start, end)
        return candles + candles[:1]

    first = store.synchronize('R_100', fetch, now_epoch=now)
    second = store.synchronize('R_100', fetch, now_epoch=now + 900)
    other = store.synchronize('R_75', fetch, now_epoch=now)

    assert first['complete'] and first['added'] == 336
    assert second['complete'] and second['requested'] == 1 and second['added'] == 1
    assert other['added'] == 336
    assert calls[1][0] == calls[1][1]
    assert len(store.read('R_100')) == 336
    assert len(store.read('R_75')) == 336
    assert store.path_for('R_100') != store.path_for('R_75')


def test_asset_history_retries_failed_page_without_losing_partial_data(tmp_path):
    """Uma falha de página preserva candles já baixados e só reconsulta os epochs faltantes."""
    store = AssetHistoryStore(tmp_path / 'Histórico Ativos', 900, max_candles=336, page_limit=200)
    now = 10 * 86400
    calls = []

    def partial_then_fail(start, end):
        calls.append((start, end))
        if len(calls) == 1:
            return _candles_between(start, end)
        raise RuntimeError('offline')

    first = store.synchronize('R_100', partial_then_fail, now_epoch=now)
    partial_count = len(store.read('R_100'))
    retry = store.synchronize(
        'R_100', lambda start, end: _candles_between(start, end), now_epoch=now
    )

    assert not first['complete']
    assert partial_count == 200
    assert retry['complete']
    assert len(store.read('R_100')) == 336
    assert calls[0][1] < calls[1][0]


@pytest.mark.parametrize('available_days', [3, 7])
def test_asset_history_retains_available_three_to_seven_day_window(tmp_path, available_days):
    """Mantém os candles disponíveis até o limite operacional configurado."""
    store = AssetHistoryStore(tmp_path / 'Histórico Ativos', 900, max_candles=336)
    now = 10 * 86400

    def fetch(start, end):
        available_start = max(start, end - available_days * 86400 + 900)
        return _candles_between(available_start, end)

    result = store.synchronize('R_100', fetch, now_epoch=now)
    rows = store.read('R_100')

    assert result['complete']
    assert rows[0]['epoch'] >= result['window_start']
    assert rows[-1]['epoch'] == result['window_end']
    assert len(rows) == min(336, available_days * 86400 // 900)


def test_asset_history_candle_gaps_request_only_missing_epochs(tmp_path):
    """Uma lacuna em candles confirmados gera apenas a consulta daquele intervalo."""
    store = AssetHistoryStore(tmp_path / 'Histórico Ativos', 900, max_candles=336)
    now = 10 * 86400
    store.synchronize('R_100', lambda start, end: _candles_between(start, end), now_epoch=now)
    rows = store.read('R_100')
    missing_epoch = rows[100]['epoch']
    rows.pop(100)
    store._write_csv_atomic(store.path_for('R_100'), rows)
    requested = []

    result = store.synchronize(
        'R_100',
        lambda start, end: requested.append((start, end)) or _candles_between(start, end),
        now_epoch=now,
    )

    assert requested == [(missing_epoch, missing_epoch)]
    assert result['added'] == 1
    assert len(store.read('R_100')) == 336


def test_asset_history_does_not_rewrite_corrupt_existing_file(tmp_path):
    """Dados inválidos locais bloqueiam sincronização sem apagar o arquivo original."""
    store = AssetHistoryStore(tmp_path / 'Histórico Ativos', 900)
    path = store.path_for('R_100')
    path.parent.mkdir(parents=True)
    path.write_text('invalid,csv\n', encoding='utf-8')
    original = path.read_bytes()

    with pytest.raises(ValueError):
        store.synchronize('R_100', lambda *_: [], now_epoch=10 * 86400)

    assert path.read_bytes() == original


def test_asset_history_preserves_file_with_wrong_symbol_or_kind(tmp_path):
    """CSV per-asset com linha de outro símbolo não é filtrado nem regravado silenciosamente."""
    store = AssetHistoryStore(tmp_path / 'Histórico Ativos', 900)
    path = store.path_for('R_100')
    path.parent.mkdir(parents=True)
    foreign = store._normalize_candle('R_75', _candles_between(900, 900)[0])
    store._write_csv_atomic(path, [foreign])
    original = path.read_bytes()

    with pytest.raises(ValueError, match='ativo ou tipo inesperado'):
        store.synchronize('R_100', lambda *_: [], now_epoch=10 * 86400)

    assert path.read_bytes() == original


def test_asset_history_rechecks_empty_epochs_after_ttl(tmp_path):
    """Resposta sem candles evita consultas a cada scan e é revalidada após uma hora."""
    store = AssetHistoryStore(tmp_path / 'Histórico Ativos', 900)
    now = 10 * 86400
    calls = []
    fetch = lambda start, end: calls.append((start, end)) or []

    store.synchronize('R_100', fetch, now_epoch=now)
    first_call_count = len(calls)
    store.synchronize('R_100', fetch, now_epoch=now)
    assert len(calls) == first_call_count

    store.synchronize('R_100', fetch, now_epoch=now + 3600)
    assert len(calls) == first_call_count * 2

def test_ranking_prefers_forex_over_synthetic_regardless_of_score(monkeypatch):
    """Garante que não sintéticos vêm antes de sintéticos mesmo com score menor."""
    import pipeline
    monkeypatch.setattr(pipeline, '_probe_symbol', lambda _s: (True, ''))
    monkeypatch.setattr(pipeline, '_scan_adx_values', {'frxEURUSD': 40, 'R_100': 60, 'frxGBPUSD': 40, 'JD50': 40})
    monkeypatch.setattr(pipeline, '_scan_adx_epochs', {'frxEURUSD': 100, 'R_100': 100,
                                                       'frxGBPUSD': 100, 'JD50': 100})
    monkeypatch.setattr(pipeline, '_adx_is_fresh', lambda _symbol: True)
    monkeypatch.setattr(pipeline, '_active_symbols_catalog', {
        'frxEURUSD': {'is_synthetic': False, 'exchange_is_open': True, 'is_trading_suspended': False},
        'frxGBPUSD': {'is_synthetic': False, 'exchange_is_open': True, 'is_trading_suspended': False},
        'R_100': {'is_synthetic': True, 'exchange_is_open': True, 'is_trading_suspended': False},
        'JD50': {'is_synthetic': True, 'exchange_is_open': True, 'is_trading_suspended': False},
    })
    results = {'R_100': 90.0, 'frxEURUSD': 20.0, 'JD50': 80.0, 'frxGBPUSD': 5.0}
    selected = [s for s, _score in pipeline._top_live_from(results, limit=5)]
    assert selected == ['frxEURUSD', 'frxGBPUSD', 'R_100', 'JD50']
