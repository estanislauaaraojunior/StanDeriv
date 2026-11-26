"""Sessões e status concorrente, sem estado operacional nem conexão externa."""
import json
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

from runtime_status import update_status


def test_new_session_clears_old_decisions_errors_and_ticks(tmp_path):
    path=tmp_path/'status.json'
    path.write_text(json.dumps(dict(session_id='old',pid=1,symbol='OLD',
        last_decision={'symbol':'OLD'},decision_counts={'old':3},
        last_api_error={'code':'Disconnected'},last_tick_at='old',shadow={'model_id':'offline'},
        candle_count=200,stopped_at='old')))
    state=update_status(path,dict(session_id='new',pid=2,symbol='NEW',min_candles=150))
    assert state['last_decision']=={} and state['decision_counts']=={}
    assert state['last_api_error']=={} and state['shadow']=={}
    assert state['last_tick_at'] is None and state['stopped_at'] is None
    assert state['candle_count']==0 and state['warmup_remaining_candles']==150


def test_pipeline_does_not_overwrite_executor_candle_counts(tmp_path):
    path=tmp_path/'status.json'
    update_status(path,dict(pid=1,symbol='FX',candle_source='executor',candle_count=87,
                           min_candles=150,warmup_remaining_candles=63),owner='executor')
    state=update_status(path,dict(pid=1,symbol='FX',candle_source='pipeline',
        candle_count=0,warmup_remaining_candles=150,last_seen_at='now'),owner='pipeline')
    assert state['candle_count']==87 and state['warmup_remaining_candles']==63
    assert state['last_seen_at']=='now'
    state=update_status(path,dict(pid=1,symbol='OTHER',candle_count=10,
        warmup_remaining_candles=140),owner='pipeline')
    assert state['candle_count']==10 and state['last_decision']=={}


def test_concurrent_updates_keep_independent_fields(tmp_path):
    path=tmp_path/'status.json'
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda n:update_status(path,{f'field_{n}':n}),range(20)))
    state=json.loads(path.read_text())
    assert all(state[f'field_{n}']==n for n in range(20))


def test_pipeline_preserves_private_ai_candles_until_symbol_changes(tmp_path):
    path = tmp_path / 'status.json'
    update_status(path, dict(pid=1, symbol='FX', candle_source='training_private',
                            candle_count=50, min_candles=50, warmup_remaining_candles=0))
    result = update_status(path, dict(pid=1, symbol='FX', candle_source='pipeline',
                           candle_count=0, min_candles=150, warmup_remaining_candles=150), owner='pipeline')
    assert result['candle_source'] == 'training_private'
    assert result['candle_count'] == result['min_candles'] == 50
    assert result['warmup_remaining_candles'] == 0
    result = update_status(path, dict(pid=1, symbol='OTHER', candle_source='pipeline',
                           candle_count=0, min_candles=150), owner='pipeline')
    assert result['candle_count'] == 0 and result['symbol'] == 'OTHER'


def test_executor_publishes_actual_buffer_every_time(monkeypatch):
    import executor
    from risk_manager import RiskManager
    bot=executor.DerivBot(RiskManager(1000))
    executor._BOT_STATUS_JSON.write_text(json.dumps(dict(last_decision={'symbol':'OLD'},
        last_api_error={'code':'old'},candle_count=0,warmup_remaining_candles=150)))
    bot._merge_candles([dict(epoch=i*900,open=1,high=2,low=1,close=2) for i in range(1,88)])
    bot._write_runtime_status(force=True,event='tick')
    state=json.loads(executor._BOT_STATUS_JSON.read_text())
    assert state['candle_count']==87 and state['warmup_remaining_candles']==63
    assert state['candle_source']=='executor'
    assert state['last_decision']=={} and state['last_api_error']=={}
    bot._write_runtime_status(force=True,last_api_error={'code':'current'})
    bot._write_runtime_status(force=True,event='tick')
    assert json.loads(executor._BOT_STATUS_JSON.read_text())['last_api_error']=={'code':'current'}


def test_dashboard_poll_does_not_fabricate_worker_heartbeat(monkeypatch,tmp_path):
    import dashboard.server as srv
    path=tmp_path/'status.json'
    monkeypatch.setattr(srv,'BOT_STATUS_JSON',path)
    path.write_text(json.dumps(dict(pid=123,status='running',running=True,last_seen_at='worker-time')))
    state=srv._refresh_bot_status(True,123)
    assert state['last_seen_at']=='worker-time'


def test_late_launcher_update_keeps_running_executor_status(tmp_path):
    path=tmp_path/'status.json'
    update_status(path,dict(session_id='session',pid=123,symbol='FX',status='running',
        candle_source='executor',candle_count=87,warmup_remaining_candles=63))
    state=update_status(path,dict(pid=123,status='starting',command=['pipeline.py']),owner='dashboard')
    assert state['status']=='running' and state['candle_count']==87


def test_dashboard_mode_comes_from_status(monkeypatch,tmp_path):
    import dashboard.server as srv
    monkeypatch.setattr(srv,'_bot_running',lambda:(True,123))
    monkeypatch.setattr(srv,'_refresh_bot_status',lambda *args:{'mode':'real','status':'running'})
    monkeypatch.setattr(srv,'_read_json_safe',lambda *_:{})
    monkeypatch.setattr(srv,'_PSUTIL',False)
    with srv.app.test_client() as client:
        assert client.get('/api/bot/status').get_json()['mode']=='real'
