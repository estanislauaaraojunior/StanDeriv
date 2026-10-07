"""Offline regressions for symbol admission and collector reconnection."""
import json
import threading
import config
from unittest.mock import Mock

import pytest
import pipeline
from market_data import ensure_header


@pytest.mark.parametrize('scenario,expected', [
    ('valid', True), ('invalid_symbol', False), ('stale_tick', False),
    ('wrong_symbol', False), ('no_call_put', False), ('disconnect', False),
    ('duration_closed', False), ('only_call', False), ('invalid_payout', False),
])
def test_probe_requires_live_tick_and_contracts(monkeypatch, scenario, expected):
    """Garante que _probe_symbol só aprova símbolo com tick ao vivo e cotações CALL/PUT válidas, sem comprar."""
    sent = []

    class Socket:
        """WebSocket falso que responde conforme o cenário parametrizado."""
        def __init__(self, url, **callbacks):
            """Registra os callbacks como atributos da instância."""
            self.__dict__.update(callbacks)

        def run_forever(self):
            """Simula a abertura da conexão."""
            self.on_open(self)

        def send(self, raw):
            """Registra a requisição e responde com tick, proposta ou contratos conforme o cenário."""
            request = json.loads(raw)
            sent.append(request)
            if 'ticks' in request:
                if scenario == 'disconnect':
                    self.close()
                    return
                if scenario == 'invalid_symbol':
                    reply = {'error': {'code': 'InvalidSymbol', 'message': 'Invalid symbol.'}}
                else:
                    reply = {'tick': {'symbol': 'OTHER' if scenario == 'wrong_symbol' else 'R_100',
                                      'epoch': 900 if scenario == 'stale_tick' else 1000,
                                      'quote': 100}}
            elif 'proposal' in request:
                if scenario == 'duration_closed' or scenario == 'only_call' and request['contract_type'] == 'PUT':
                    reply = {'req_id': request['req_id'], 'error': {'code': 'ContractBuyValidationError',
                             'message': 'Trading on forex contracts with duration less than 5 hours is not available'}}
                else:
                    reply = {'req_id': request['req_id'], 'proposal': {'id': 'quote', 'ask_price': 1,
                             'payout': 'nan' if scenario == 'invalid_payout' else 1.85}}
            else:
                kinds = ['DIGITMATCH'] if scenario == 'no_call_put' else ['CALL', 'PUT']
                reply = {'contracts_for': {'available': [{'contract_type': k} for k in kinds]}}
            self.on_message(self, json.dumps(reply))

        def close(self):
            """Dispara o callback de fechamento."""
            self.on_close(self, None, None)

    monkeypatch.setattr(pipeline.websocket, 'WebSocketApp', Socket)
    monkeypatch.setattr(pipeline, '_get_ws_url', lambda: 'offline')
    monkeypatch.setattr(pipeline, '_uses_bearer_auth', lambda: True)
    monkeypatch.setattr(pipeline.time, 'time', lambda: 1000)
    ok, reason = pipeline._probe_symbol('R_100', timeout=.1)
    assert ok is expected
    assert bool(reason) is not expected
    assert not any('buy' in r for r in sent)
    quotes = [r for r in sent if 'proposal' in r]
    assert {r['contract_type'] for r in quotes} == {'CALL', 'PUT'}
    assert all(r['duration'] == config.DURATION for r in quotes)


def test_scanner_skips_historical_winner_without_live_feed(monkeypatch):
    """Garante que o scanner descarta o melhor símbolo histórico sem feed ao vivo e escolhe o próximo."""
    probe = Mock(side_effect=[(False, 'Invalid symbol.'), (True, '')])
    monkeypatch.setattr(pipeline, '_probe_symbol', probe)
    monkeypatch.setattr(pipeline, '_ADX_TREND_MIN', 20)
    assert pipeline._best_live_from({'frxNZDCAD': 35, 'frxEURUSD': 30}) == ('frxEURUSD', 30)
    assert [call.args[0] for call in probe.call_args_list] == ['frxNZDCAD', 'frxEURUSD']


def test_fixed_symbol_searches_fallback_if_unavailable(monkeypatch):
    """Garante que símbolo fixo indisponível aciona a busca de um símbolo alternativo ao vivo."""
    monkeypatch.setattr(pipeline, '_probe_symbol', lambda _: (False, 'Invalid symbol.'))
    monkeypatch.setattr(pipeline, '_first_live_symbol', lambda _: 'R_50')
    assert pipeline._detect_trending_symbol(no_scan=True) == 'R_50'


@pytest.fixture
def collector(tmp_path, monkeypatch):
    """Cria um _CollectorThread para R_100 gravando em CSV temporário."""
    path = tmp_path / 'ticks.csv'
    ensure_header(path)
    monkeypatch.setattr(pipeline, 'TICKS_CSV', str(path))
    return pipeline._CollectorThread('R_100')


def test_collector_waits_for_legacy_authorization(collector, monkeypatch):
    """Garante que, sem bearer, o coletor só assina ticks após a resposta de autorização."""
    monkeypatch.setattr(pipeline, '_uses_bearer_auth', lambda: False)
    ws = Mock()
    collector._on_open(ws)
    assert len(ws.send.call_args_list) == 1
    assert 'authorize' in json.loads(ws.send.call_args.args[0])
    collector._on_message(ws, json.dumps({'msg_type': 'authorize'}))
    assert json.loads(ws.send.call_args.args[0]) == {'ticks': 'R_100', 'subscribe': 1}
    collector.stop()


def test_invalid_symbol_requests_market_change(collector, monkeypatch):
    """Garante que erro InvalidSymbol fecha o socket e faz a espera por ticks levantar MarketUnavailable."""
    monkeypatch.setattr(pipeline, '_runtime_symbol', lambda: collector._symbol)
    monkeypatch.setattr(pipeline, '_market_switch', threading.Event())
    ws = Mock()
    collector._on_message(ws, json.dumps({'error': {'code': 'InvalidSymbol', 'message': 'Invalid symbol.'}}))
    ws.close.assert_called_once()
    with pytest.raises(pipeline.MarketUnavailable):
        pipeline._wait_for_ticks(200, collector=collector)


def test_disconnect_closes_socket_for_new_session(collector):
    """Garante que erro de conexão fecha o socket para que uma nova sessão seja aberta."""
    ws = Mock()
    collector._on_error(ws, RuntimeError('connection lost'))
    ws.close.assert_called_once()


def test_reconnection_gets_new_url(collector, monkeypatch):
    """Garante que cada reconexão do coletor obtém uma nova URL de sessão e não usa reconnect automático."""
    urls = []
    monkeypatch.setattr(pipeline, '_shutdown', Mock(is_set=lambda: False))
    monkeypatch.setattr(collector, '_stopped', Mock(is_set=lambda: False, wait=lambda _: False))
    monkeypatch.setattr(pipeline, '_get_ws_url', Mock(side_effect=['session-1', 'session-2']))

    class Socket:
        """WebSocket falso que registra as URLs e encerra o coletor na segunda sessão."""
        def __init__(self, url, **callbacks):
            """Registra a URL usada na conexão."""
            urls.append(url)

        def run_forever(self, **kwargs):
            """Valida a ausência de reconnect e sinaliza erro fatal na segunda sessão."""
            assert 'reconnect' not in kwargs
            if len(urls) == 2:
                collector.fatal_error = 'end offline test'

    monkeypatch.setattr(pipeline.websocket, 'WebSocketApp', Socket)
    collector.run()
    assert urls == ['session-1', 'session-2']
