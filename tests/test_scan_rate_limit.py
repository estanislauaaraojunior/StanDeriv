"""Limite de concorrência e pausa global do scanner sem API real."""
import threading
import json
from unittest.mock import Mock

import pytest
import pipeline
import request_throttle


def test_scan_runs_at_most_two_symbols(monkeypatch):
    symbols = ['A', 'B', 'C', 'D']
    monkeypatch.setattr(pipeline, '_active_symbols_catalog', {
        symbol: dict(market='forex', underlying_symbol_type='forex', exchange_is_open=True)
        for symbol in symbols
    })
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    active = peak = 0
    def fetch(symbol, count):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        barrier.wait(timeout=3)
        with lock:
            active -= 1
        return []
    monkeypatch.setattr(pipeline, '_fetch_prices_for_symbol', fetch)
    result = pipeline._scan_group(symbols)
    assert set(result) == set(symbols)
    assert peak == 2 and pipeline._SCAN_WORKERS == 2


def test_rate_limit_retries_same_request_and_shares_pause(monkeypatch):
    throttle = Mock()
    monkeypatch.setattr(pipeline, '_scan_throttle', throttle)
    throttle.backoff.return_value = 5
    error = pipeline.deriv_session.DerivAPIError('rate limit', code='RateLimit')
    error.retry_after = 12
    request = {'ticks_history': 'R_100', 'start': 900, 'end': 1800}
    fetch = Mock(side_effect=[error, {'candles': []}])
    monkeypatch.setattr(pipeline, '_websocket_request_once', fetch)
    assert pipeline._websocket_request(request) == {'candles': []}
    assert throttle.wait.call_count == 2
    throttle.backoff.assert_called_once_with(12)
    assert all(call.args[0] == request for call in fetch.call_args_list)


def test_rate_limit_retry_is_bounded_and_other_errors_are_not_retried(monkeypatch):
    throttle = Mock()
    throttle.backoff.return_value = 5
    monkeypatch.setattr(pipeline, '_scan_throttle', throttle)
    fetch = Mock(side_effect=pipeline.deriv_session.DerivAPIError('limited', status=429))
    monkeypatch.setattr(pipeline, '_websocket_request_once', fetch)
    with pytest.raises(pipeline.deriv_session.DerivAPIError):
        pipeline._websocket_request({})
    assert fetch.call_count == 3
    fetch.reset_mock(side_effect=True)
    fetch.side_effect = ValueError('invalid candle')
    with pytest.raises(ValueError):
        pipeline._websocket_request({})
    assert fetch.call_count == 1


def test_throttle_rechecks_pause_imposed_while_waiting(monkeypatch):
    clock = [0.0]
    waits = []
    throttle = request_throttle.RequestThrottle()
    monkeypatch.setattr(request_throttle.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(request_throttle.random, 'uniform', lambda *_: 0)
    def sleep(delay):
        waits.append(delay)
        clock[0] += delay
        if len(waits) == 1:
            throttle.backoff(retry_after=12)
    monkeypatch.setattr(request_throttle.time, 'sleep', sleep)
    throttle.wait()
    throttle.wait()
    assert waits == [1.0, 12.0]
    assert clock[0] == 13


def test_websocket_rate_limit_preserves_code_and_wait_hint(monkeypatch):
    class Socket:
        def __init__(self, url, **callbacks):
            self.__dict__.update(callbacks)
        def run_forever(self):
            self.on_open(self)
        def send(self, raw):
            self.on_message(self, json.dumps({'error': {
                'code': 'RateLimit', 'message': 'limit token=private', 'retry_after': 20
            }}))
        def close(self):
            self.on_close(self, None, None)
    monkeypatch.setattr(pipeline, '_get_ws_url', lambda: 'offline')
    monkeypatch.setattr(pipeline, '_uses_bearer_auth', lambda: True)
    monkeypatch.setattr(pipeline.websocket, 'WebSocketApp', Socket)
    with pytest.raises(pipeline.deriv_session.DerivAPIError) as raised:
        pipeline._websocket_request_once({'ticks_history': 'A'})
    assert raised.value.code == 'RateLimit'
    assert raised.value.retry_after == 20
    assert 'private' not in str(raised.value)
