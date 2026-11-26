"""Keepalive JSON e recuperação de lacunas, inteiramente offline."""
import json
from collections import deque
from unittest.mock import Mock

from api_keepalive import ApiKeepalive
from asset_history import AssetHistoryStore, plan_candle_backfill


def candle(epoch):
    return dict(epoch=epoch,open=1,high=2,low=1,close=2)


def test_api_ping_acknowledgement_and_stop(capsys):
    ws=Mock()
    keepalive=ApiKeepalive(ws,'TEST')
    keepalive.send_ping()
    sent=json.loads(ws.send.call_args.args[0])
    assert sent['ping']==1
    keepalive.acknowledge({'msg_type':'ping','req_id':sent['req_id']})
    keepalive.acknowledge({'msg_type':'ping','req_id':sent['req_id']})
    assert keepalive.received==1
    assert keepalive.last_latency is not None and keepalive.last_latency >= 0
    assert capsys.readouterr().out == ''
    keepalive.stop()
    keepalive.send_ping()
    assert ws.send.call_count==1
    assert 'Ping API confirmado' not in capsys.readouterr().out


def test_keepalive_loop_sends_every_thirty_seconds():
    keepalive=ApiKeepalive(Mock(),'TEST')
    keepalive._stop=Mock()
    keepalive._stop.wait.side_effect=[False,False,True]
    keepalive._stop.is_set.return_value=False
    keepalive._run()
    assert keepalive.ws.send.call_count==2
    assert all(call.args==(30,) for call in keepalive._stop.wait.call_args_list)


def test_planner_requests_internal_gap_without_repeating_recent_candles():
    epochs=list(range(900, 151*900, 900))
    known=[e for e in epochs if e not in (80*900,81*900,82*900)]
    assert plan_candle_backfill(known,900,150,150*900)==[(80*900,82*900)]


def test_planner_excludes_closed_periods_and_splits_pages():
    calendar=Mock()
    calendar.closed_candle.side_effect=lambda symbol,epoch,tf:epoch in (4*900,5*900)
    assert plan_candle_backfill([],900,6,8*900,calendar=calendar,page_limit=2)==[
        (900,1800),(2700,2700),(5400,6300),(7200,7200)]


def test_recovery_keeps_both_sides_of_gap_and_ignores_stale_reply(monkeypatch):
    import executor
    from risk_manager import RiskManager
    bot=executor.DerivBot(RiskManager(1000))
    monkeypatch.setattr(executor.time,'time',lambda:151*900)
    monkeypatch.setattr(bot,'_preload_candles_from_csv',Mock())
    known=[candle(e) for e in range(900,151*900,900) if e not in (80*900,81*900,82*900)]
    bot._merge_candles(known)
    assert len(bot._candles)==68
    ws=Mock()
    bot._request_missing_history_or_subscribe(ws)
    request=json.loads(ws.send.call_args.args[0])
    # end cobre o intervalo inteiro do último candle (82*900 .. 82*900+899).
    assert request['start']==80*900 and request['end']==str(82*900+899)
    assert request['count']==3
    assert 'adjust_start_time' not in request
    bot._preload_candles_from_csv.assert_not_called()
    bot._handle_api_candles(ws,[candle(80*900)],req_id=request['req_id']-1)
    assert len(bot._candles)==68
    persist=Mock();bot.on_history_recovered=persist
    recovered=[candle(e) for e in (80*900,81*900,82*900)]
    bot._handle_api_candles(ws,recovered+[candle(1)],req_id=request['req_id'])
    assert len(bot._candles)==150
    persist.assert_called_once_with(recovered)
    assert json.loads(ws.send.call_args.args[0])=={'ticks':bot.symbol,'subscribe':1}
    before=list(bot._candles)
    bot._handle_api_candles(ws,[candle(150*900+900)],req_id=request['req_id'])
    assert list(bot._candles)==before


def deriv_candles(request, available):
    """Semântica observada na API em 26/11/2026: últimos `count` candles abertos antes de `end`."""
    end = int(request['end'])
    eligible = [e for e in sorted(available) if e < end]
    return [candle(e) for e in eligible[-int(request['count']):]]


def test_live_recovery_fills_single_candle_gap_with_current_api_semantics(monkeypatch):
    """Reproduz a sessão de 26/11: início no meio do candle deixava a lacuna sem fechar."""
    import executor
    from risk_manager import RiskManager
    bot=executor.DerivBot(RiskManager(1000))
    now=[151*900+5]
    monkeypatch.setattr(executor.time,'time',lambda:now[0])
    bot._merge_candles([candle(e) for e in range(900,150*900,900)])  # falta o candle 150*900
    ws=Mock()
    bot._request_missing_history_or_subscribe(ws, live=True)
    request=json.loads(ws.send.call_args.args[0])
    assert (request['start'], request['count'])==(150*900, 1)
    reply=deriv_candles(request, range(900,152*900,900))
    assert [c['epoch'] for c in reply]==[150*900]
    bot._handle_api_candles(ws, reply, req_id=request['req_id'])
    assert bot._candles[-1]['epoch']==150*900
    assert not bot._history_recovery_incomplete


def test_old_end_value_reproduced_the_stuck_gap():
    """Documenta o defeito: end == início do candle devolve o anterior."""
    stale = deriv_candles({'end': str(150*900), 'count': 1}, range(900, 152*900, 900))
    assert [c['epoch'] for c in stale] == [149*900]


def test_asset_history_fetch_requests_end_of_last_candle(monkeypatch):
    import pipeline
    sent={}
    monkeypatch.setattr(pipeline,'_websocket_request',lambda request: sent.update(request) or
                        {'candles':deriv_candles(request, range(900, 20*900, 900))})
    monkeypatch.setattr(pipeline.time,'time',lambda: 20*900+10)
    rows=pipeline._fetch_candles_for_symbol('R_100', 10*900, 12*900)
    assert sent['end']==12*900+899
    assert [r['epoch'] for r in rows]==[10*900, 11*900, 12*900]


def test_keepalive_stops_on_bot_close(monkeypatch):
    from executor import DerivBot
    bot=object.__new__(DerivBot)
    bot._api_keepalive=Mock()
    bot._connection_opened_at=None
    bot._auth_verified=True
    bot._on_close(Mock(),1000,'normal')
    bot._api_keepalive.stop.assert_called_once()
    assert not bot._auth_verified


def test_recovery_preserves_valid_candles_on_first_live_tick(monkeypatch):
    import executor
    from risk_manager import RiskManager
    bot=executor.DerivBot(RiskManager(1000))
    bot._auth_verified=True
    bot._last_tick_epoch=148*900
    bot._last_tick_price=1.5
    bot._history_stream_ready=True
    bot._merge_candles([candle(e) for e in range(900,151*900,900)])
    monkeypatch.setattr(bot,'_write_runtime_status',Mock())
    monkeypatch.setattr(bot,'_reset_watchdog',Mock())
    bot._handle_tick(Mock(),{'symbol':bot.symbol,'quote':1.5,'epoch':151*900+30})
    assert len(bot._candles)==150
    assert not bot._history_stream_ready


def test_ingest_preserves_existing_candles_and_does_not_mark_other_gaps(tmp_path):
    store=AssetHistoryStore(tmp_path,900)
    store.ingest('FX',[candle(900),candle(2700)],now_epoch=3600)
    store.ingest('FX',[candle(1800)],now_epoch=3600)
    assert [row['epoch'] for row in store.read('FX')]==[900,1800,2700]
    assert not store.path_for('FX').with_suffix('.metadata.json').exists()
