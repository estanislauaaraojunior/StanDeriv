"""Calendário oficial com respostas simuladas e histórico temporário."""
from datetime import datetime, timezone
from collections import deque
from unittest.mock import Mock

import pytest
from market_calendar import MarketCalendar
from asset_history import AssetHistoryStore


def epoch(date, clock='00:00:00'):
    return int(datetime.fromisoformat(f'{date}T{clock}').replace(tzinfo=timezone.utc).timestamp())


def response(opens=None, closes=None, days=None, events=None):
    return {'trading_times': {'markets': [{'submarkets': [{'symbols': [{
        'symbol': 'FX', 'times': {'open': opens or ['00:00:00'], 'close': closes or ['23:59:59']},
        'trading_days': days if days is not None else ['Mon','Tue','Wed','Thu','Fri'],
        'events': events or [],
    }]}]}]}}


def test_weekend_closed_and_day_cache_shared_by_symbols():
    fetch = Mock(return_value=response())
    calendar = MarketCalendar(fetch)
    saturday = epoch('2026-10-10')
    assert calendar.closed('FX', saturday, saturday + 86400)
    assert not calendar.closed('UNKNOWN', saturday, saturday + 900)
    assert fetch.call_count == 1
    fetch.assert_called_with({'trading_times': '2026-10-10'})
    assert not calendar.closed('FX', epoch('2026-10-12'), epoch('2026-10-12')+900)


def test_daily_break_reopening_and_candle_partially_open():
    calendar = MarketCalendar(lambda _: response(['00:00:00','22:00:00'], ['20:59:59','23:59:59']))
    assert calendar.closed('FX',epoch('2026-10-12','21:00:00'),epoch('2026-10-12','22:00:00'))
    assert not calendar.closed('FX',epoch('2026-10-12','20:55:00'),epoch('2026-10-12','21:10:00'))
    assert not calendar.closed('FX',epoch('2026-10-12','22:00:00'),epoch('2026-10-12','22:15:00'))


def test_calendar_failure_and_special_event_do_not_hide_gaps():
    calendar = MarketCalendar(Mock(side_effect=TimeoutError('offline')))
    start = epoch('2026-10-10')
    assert not calendar.closed('FX', start, start+900)
    calendar = MarketCalendar(lambda _: response(events=[{'dates':'2026-10-10','descrip':'special session'}]))
    assert not calendar.closed('FX', start, start+900)


def test_daily_calendar_is_refreshed_after_negative_cache(monkeypatch):
    import market_calendar
    clock=[0.0]
    monkeypatch.setattr(market_calendar.time,'monotonic',lambda:clock[0])
    fetch=Mock(side_effect=[TimeoutError('offline'),response()])
    calendar=MarketCalendar(fetch)
    start=epoch('2026-10-10')
    assert not calendar.closed('FX',start,start+900)
    clock[0]=30
    assert not calendar.closed('FX',start,start+900)
    assert fetch.call_count==1
    clock[0]=61
    assert calendar.closed('FX',start,start+900)
    assert fetch.call_count==2


def test_synthetic_with_all_days_stays_open():
    calendar = MarketCalendar(lambda _: response(days=['Mon','Tue','Wed','Thu','Fri','Sat','Sun']))
    start=epoch('2026-10-10')
    assert not calendar.closed('FX',start,start+86400)


def test_closed_day_is_not_downloaded_or_stored_as_no_data(tmp_path):
    calendar = MarketCalendar(lambda _: response())
    store=AssetHistoryStore(tmp_path,900,max_candles=96)
    fetch=Mock()
    result=store.synchronize('FX',fetch,now_epoch=epoch('2026-10-11'),calendar=calendar)
    assert result['requested']==0 and result['scheduled_closed']==96
    fetch.assert_not_called()
    import json
    metadata=json.loads(store.path_for('FX').with_suffix('.metadata.json').read_text())
    assert metadata['no_data']=={}


def test_outside_adjusted_page_does_not_poison_valid_candles(tmp_path):
    store=AssetHistoryStore(tmp_path,900,max_candles=1)
    result=store.synchronize('FX',lambda *_: [
        dict(epoch=1,open=-1,high=-1,low=-1,close=-1),
        dict(epoch=900,open=1,high=2,low=1,close=2),
    ],now_epoch=1800)
    assert result['complete'] and result['added']==1
    result=store.synchronize('FX',lambda *_: [dict(epoch=1800,open=2,high=1,low=1,close=2)],now_epoch=2700)
    assert not result['complete']  # OHLC inválido dentro da faixa continua falhando


def test_technical_warmup_crosses_only_confirmed_closures(monkeypatch):
    import executor
    bot=object.__new__(executor.DerivBot)
    bot.symbol='FX';bot._candles=deque()
    bot.market_calendar=MarketCalendar(lambda _:response())
    monkeypatch.setattr(executor.config,'USE_AI_MODEL',False)
    def candle(e):return dict(epoch=e,open=1,high=2,low=1,close=2)
    friday=epoch('2026-10-09','23:45:00');monday=epoch('2026-10-12')
    bot._merge_candles([candle(friday),candle(monday)])
    assert len(bot._candles)==2
    bot._merge_candles([candle(monday+1800)])
    assert len(bot._candles)==1  # candle faltante em mercado aberto
    bot._candles.clear()
    monkeypatch.setattr(executor.config,'USE_AI_MODEL',True)
    bot._merge_candles([candle(friday),candle(monday)])
    assert len(bot._candles)==1


def test_feed_watchdog_classifies_scheduled_close(monkeypatch, capsys):
    import executor
    bot=object.__new__(executor.DerivBot)
    bot.symbol='FX'
    bot.market_calendar=MarketCalendar(lambda _:response())
    bot._write_runtime_status=Mock()
    bot._reset_watchdog=Mock()
    monkeypatch.setattr(executor.time,'time',lambda:epoch('2026-10-10'))
    bot._on_tick_timeout()
    bot._write_runtime_status.assert_called_once_with(force=True,event='market_closed')
    bot._reset_watchdog.assert_called_once()
    assert 'fechamento programado' in capsys.readouterr().out


def test_collector_does_not_reconnect_during_known_closure(monkeypatch):
    import pipeline
    calendar=MarketCalendar(lambda _:response())
    monkeypatch.setattr(pipeline,'_market_calendar',calendar)
    monkeypatch.setattr(pipeline.time,'time',lambda:epoch('2026-10-10'))
    monkeypatch.setattr(pipeline,'_shutdown',thread_event())
    collector=pipeline._CollectorThread('FX')
    ws=Mock()
    collector._on_first_tick_timeout(ws)
    ws.close.assert_not_called()
    assert not collector._first_tick_timed_out
    collector._first_tick_timer.start.assert_called_once()


def thread_event():
    import threading
    return threading.Event()


def test_trading_times_response_is_accepted_without_network(monkeypatch):
    import pipeline,json
    class Socket:
        def __init__(self,_url,**callbacks):self.__dict__.update(callbacks)
        def run_forever(self):self.on_open(self)
        def send(self,raw):
            assert json.loads(raw)=={'trading_times':'2026-10-10'}
            self.on_message(self,json.dumps(dict(response(),msg_type='trading_times')))
        def close(self):self.on_close(self,None,None)
    monkeypatch.setattr(pipeline,'_get_ws_url',lambda:'offline')
    monkeypatch.setattr(pipeline,'_uses_bearer_auth',lambda:True)
    monkeypatch.setattr(pipeline.websocket,'WebSocketApp',Socket)
    assert pipeline._websocket_request_once({'trading_times':'2026-10-10'})==dict(response(),msg_type='trading_times')
