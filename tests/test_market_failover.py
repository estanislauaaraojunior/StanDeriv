"""Market failover without purchases, model mixing or risk resets."""
import json
import threading
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


def test_initial_wait_observes_market_change(monkeypatch):
    event = threading.Event(); event.set()
    monkeypatch.setattr(pipeline, '_market_switch', event)
    with pytest.raises(pipeline.MarketUnavailable):
        pipeline._wait_for_ticks(200)
    with pytest.raises(pipeline.MarketUnavailable):
        pipeline._wait_for_compatible_model()


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
    def select(candidates):
        assert order[-1] == 'train_finished'
        assert 'OLD' not in candidates
        return 'NEW'
    monkeypatch.setattr(pipeline, '_main_session', session)
    monkeypatch.setattr(pipeline, '_scan_candidates', lambda: ['OLD', 'NEW'])
    monkeypatch.setattr(pipeline, '_first_live_symbol', select)
    pipeline.main()
    assert order == [('session', None), 'stop', 'joined', 'train_finished', ('session', 'NEW')]
    lock.close.assert_called_once()
