"""Diagnósticos e retry com dados temporários e sem rede."""
from unittest.mock import MagicMock

import operational_diagnostics as diagnostics
from asset_history import AssetHistoryStore


def test_diagnostic_redacts_secrets_and_reports_duration(monkeypatch):
    monkeypatch.setenv('DERIV_TOKEN', 'known-secret')
    monkeypatch.setattr(diagnostics.time, 'monotonic', lambda: 15.0)
    detail = diagnostics.connection_diagnostic(
        10.0, error=RuntimeError('lost known-secret wss://host/session?otp=abc token=xyz Bearer hidden')
    )
    assert 'duração=5.0s' in detail and 'tipo=RuntimeError' in detail
    assert 'horário=' in detail and 'lost' in detail
    for secret in ('known-secret', 'abc', 'xyz', 'hidden', 'wss://'):
        assert secret not in detail


def test_close_reason_is_safe_and_unopened_is_explicit():
    detail = diagnostics.connection_diagnostic(None, code=1006, reason='{"otp": "private"}')
    assert 'não aberta' in detail and '1006' in detail
    assert 'private' not in detail


def test_bot_and_collector_callbacks_use_safe_diagnostics(capsys, monkeypatch):
    from executor import DerivBot
    import pipeline
    monkeypatch.setenv('DERIV_TOKEN', 'callback-secret')
    bot = object.__new__(DerivBot)
    bot._connection_opened_at = None
    collector = pipeline._CollectorThread('R_100')
    collector._first_tick_timed_out = False
    monkeypatch.setattr(pipeline, '_shutdown', MagicMock(is_set=lambda: False))
    for component in (bot, collector):
        component._on_error(MagicMock(), RuntimeError('lost callback-secret'))
        component._on_close(MagicMock(), 1006, 'token=private')
    output = capsys.readouterr().out
    assert output.count('tipo=RuntimeError') == 2
    assert output.count('código=1006') == 2
    assert 'callback-secret' not in output and 'private' not in output
    assert not bot._auth_verified


def test_history_retries_only_failed_page(tmp_path, monkeypatch, capsys):
    import asset_history
    sleep = MagicMock()
    monkeypatch.setattr(asset_history.time, 'sleep', sleep)
    store = AssetHistoryStore(tmp_path, 900, max_candles=2, page_limit=1)
    calls = []

    def fetch(start, end):
        calls.append((start, end))
        if start == 8100 and calls.count((start, end)) < 3:
            raise TimeoutError('timeout token=hidden')
        return [dict(epoch=start, open=1, high=2, low=1, close=2)]

    result = store.synchronize('R_100', fetch, now_epoch=9000, retries=2)
    assert result['complete'] and result['added'] == 2
    assert calls == [(7200, 7200), (8100, 8100), (8100, 8100), (8100, 8100)]
    assert [call.args[0] for call in sleep.call_args_list] == [1.0, 2.0]
    output = capsys.readouterr().out
    assert 'TimeoutError' in output and '8100..8100' in output
    assert 'hidden' not in output


def test_history_exhausted_retry_preserves_successful_page(tmp_path, monkeypatch):
    import asset_history
    monkeypatch.setattr(asset_history.time, 'sleep', lambda _: None)
    store = AssetHistoryStore(tmp_path, 900, max_candles=2, page_limit=1)
    def fetch(start, end):
        if start == 8100:
            raise ConnectionError('offline')
        return [dict(epoch=start, open=1, high=2, low=1, close=2)]
    result = store.synchronize('R_100', fetch, now_epoch=9000, retries=2)
    assert not result['complete']
    assert result['failed_ranges'] == [(8100, 8100)]
    assert [row['epoch'] for row in store.read('R_100')] == [7200]
