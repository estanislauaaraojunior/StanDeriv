"""Ordens com requisições correlacionadas e liquidação idempotente, offline."""
import csv
import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import config
import executor
from executor import DerivBot
from risk_manager import RiskManager


@pytest.fixture
def bot(tmp_path, monkeypatch):
    """Cria um DerivBot autenticado, com conta vinculada e tick recente."""
    risk = RiskManager(1000.)
    risk.bind_account('test-account', 1000.)
    b = DerivBot(risk)
    monkeypatch.setattr(executor, '_SHADOW_STATE_JSON', tmp_path/'shadow.json')
    b._auth_verified = True
    b._last_tick_epoch = int(time.time())
    b._last_tick_price = 100.
    b._shadow_monitor = SimpleNamespace(approved=True)
    b._shadow_model_id = 'offline'
    b._shadow_symbol = b.symbol
    return b


def proposal(bot, ws, direction = 'BUY'):
    """Envia uma proposta com probabilidade 0.8 e retorna o req_id correlacionado."""
    bot._shadow_symbol = bot.symbol
    bot._send_proposal(ws, direction, {'win_probability': .8, 'ai_confidence': .8, 'model_id': 'offline'})
    return bot._proposal_req_id


def buy(bot, ws, direction = 'BUY'):
    """Percorre proposta, aceite e compra até abrir o contrato 12345."""
    rid = proposal(bot, ws, direction)
    bot._handle_proposal(ws, {'id': 'proposal-test', 'ask_price': '10', 'payout': '18.5'}, req_id = rid)
    bot._handle_buy(ws, {'contract_id': 12345}, req_id = bot._buy_req_id)


@pytest.mark.parametrize('direction,contract_type', [('BUY', 'CALL'), ('SELL', 'PUT')])
def test_request_uses_same_symbol_duration_and_direction(bot, direction, contract_type):
    """Garante que a proposta usa o símbolo, a duração e o tipo de contrato correspondentes à direção."""
    ws = MagicMock()
    proposal(bot, ws, direction)
    sent = json.loads(ws.send.call_args.args[0])
    assert sent['contract_type'] == contract_type
    assert sent['symbol'] == bot.symbol
    assert sent['duration'] == config.DURATION == 15
    assert sent['duration_unit'] == config.DURATION_UNIT
    assert bot._in_trade and bot._phase == 'proposal'
    assert sent['req_id'] == bot._proposal_req_id


def test_forex_symbol_can_open_contract_after_valid_signal(bot):
    """Garante que um símbolo forex abre contrato PUT após proposta e compra válidas."""
    bot.symbol = 'frxEURAUD'
    ws = MagicMock()
    rid = proposal(bot, ws, 'SELL')
    proposed = json.loads(ws.send.call_args.args[0])
    assert proposed['symbol'] == 'frxEURAUD'
    assert proposed['contract_type'] == 'PUT'
    bot._handle_proposal(ws, {'id': 'proposal-forex', 'ask_price': '10', 'payout': '18.5'}, req_id=rid)
    bot._handle_buy(ws, {'contract_id': 67890}, req_id=bot._buy_req_id)
    assert bot._open_contract_id == '67890'
    assert bot._phase == 'open'


def test_non_model_decision_can_reach_buy_without_ai_metadata(bot):
    """Garante que decisão sem módulo de IA chega à compra sem metadados do modelo."""
    ws = MagicMock()
    indicators = {
        'decision_modules': ['weighted_score'],
        'decision_source': 'weighted_score',
        'tech_score': .7,
    }

    sent = bot._send_proposal(ws, 'BUY', indicators)
    proposal_req_id = bot._proposal_req_id
    bot._handle_proposal(
        ws,
        {'id': 'technical-proposal', 'ask_price': 10, 'payout': 18.5},
        req_id=proposal_req_id,
    )

    assert sent is True
    assert bot._phase == 'buy'
    assert json.loads(ws.send.call_args.args[0])['buy'] == 'technical-proposal'


def test_weighted_score_proposal_does_not_consult_ai_duration(bot, monkeypatch):
    """Garante que score ponderado puro não usa IA para decidir abertura."""
    ws = MagicMock()
    monkeypatch.setattr(
        executor.ai_predictor,
        'predict_duration',
        lambda *_args, **_kwargs: pytest.fail('score ponderado nao deve consultar IA'),
    )

    sent = bot._send_proposal(
        ws,
        'BUY',
        {'decision_modules': ['weighted_score'], 'decision_source': 'weighted_score'},
    )

    assert sent is True
    proposal = json.loads(ws.send.call_args.args[0])
    assert proposal['duration'] == config.DURATION


def test_weighted_score_proposal_does_not_consult_shadow_gate(bot):
    """Garante que score ponderado puro não depende da validação shadow do modelo."""
    ws = MagicMock()
    bot._shadow_monitor.approved = False

    sent = bot._send_proposal(
        ws,
        'BUY',
        {'decision_modules': ['weighted_score'], 'decision_source': 'weighted_score'},
    )

    assert sent is True
    assert bot._phase == 'proposal'
    assert json.loads(ws.send.call_args.args[0])['contract_type'] == 'CALL'


def test_non_model_decision_still_obeys_minimum_payout(bot):
    """Garante que decisão sem IA continua rejeitando propostas com payout abaixo do mínimo."""
    ws = MagicMock()
    indicators = {
        'decision_modules': ['price_action'],
        'decision_source': 'price_action',
    }

    sent = bot._send_proposal(ws, 'SELL', indicators)
    bot._handle_proposal(
        ws,
        {'id': 'low-payout', 'ask_price': 10, 'payout': 15.99},
        req_id=bot._proposal_req_id,
    )

    assert sent is True
    assert ws.send.call_count == 1
    assert not bot._in_trade


def test_model_decision_is_blocked_until_shadow_is_approved(bot):
    """Garante que um modelo sem validação forward não envia proposta."""
    bot._shadow_monitor.approved = False
    ws = MagicMock()

    sent = bot._send_proposal(ws, 'BUY', {'model_id': 'offline', 'win_probability': .8})

    assert sent is False
    ws.send.assert_not_called()


def test_model_decision_is_blocked_on_real_account(bot):
    """Garante que o novo fluxo shadow não libera compras em conta real."""
    bot.demo = False
    ws = MagicMock()

    sent = bot._send_proposal(ws, 'BUY', {'model_id': 'offline', 'win_probability': .8})

    assert sent is False
    ws.send.assert_not_called()


def test_executor_records_shadow_prediction_with_prior_observed_quote(bot, tmp_path, monkeypatch):
    """Garante que a previsão shadow usa payout observado antes da entrada e não compra."""
    from payout_quotes import FIELDS
    import executor as executor_module
    ticks_path = tmp_path / 'ticks.csv'
    quotes_path = tmp_path / 'payout_quotes.csv'
    entry_epoch = 1_800_000_000
    monkeypatch.setattr(executor_module, 'TICKS_CSV', str(ticks_path))
    with quotes_path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        writer.writerow([entry_epoch - 1, bot.symbol, config.DURATION, config.DURATION_UNIT,
                         'CALL', config.CURRENCY, 10, 18, .8, 'sampled_proposal'])
    monkeypatch.setattr(executor_module.ai_predictor, 'status', lambda: {'ready': True, 'model_id': 'model-shadow'})
    monkeypatch.setattr(executor_module.ai_predictor, 'predict', lambda *_args, **_kwargs: ('BUY', .8))
    monkeypatch.setattr(executor_module.ai_predictor, 'prediction_model_id', lambda: 'model-shadow')
    bot._candle_closes.extend([100.0] * 50)
    bot._candles.extend([{'epoch': i * config.CANDLE_TIMEFRAME_SEC, 'close': 100.0} for i in range(50)])

    bot._capture_shadow_prediction(100.0, entry_epoch)

    assert bot._shadow_monitor.state['pending']['direction'] == 1
    assert bot._shadow_monitor.state['pending']['net_payout'] == pytest.approx(.8)
    assert bot._shadow_monitor.result_count == 0


def test_shadow_rejection_requests_market_switch(bot, tmp_path, monkeypatch):
    """Garante que a liquidação shadow reprovada solicita troca de mercado."""
    callback = MagicMock()
    monkeypatch.setattr(executor, '_BOT_STATUS_JSON', tmp_path/'bot_status.json')
    monkeypatch.setattr(executor.ai_predictor, 'model_id', lambda: 'offline')
    bot._shadow_monitor = SimpleNamespace(settle_tick=lambda **_kwargs: 'rejected')
    bot._shadow_model_id = 'offline'
    bot._shadow_symbol = bot.symbol
    bot.on_shadow_rejected = callback

    bot._handle_tick(
        MagicMock(),
        {'symbol': bot.symbol, 'epoch': bot._last_tick_epoch + 1, 'quote': 100.0},
    )

    callback.assert_called_once_with(bot.symbol)


def test_model_proposal_requires_observed_net_payout_of_70_percent(bot):
    """Garante que o payout bruto da proposal equivale a pelo menos 70% líquido antes da compra."""
    ws = MagicMock()
    rid = proposal(bot, ws)

    bot._handle_proposal(ws, {'id': 'low-shadow-payout', 'ask_price': 10, 'payout': 16.9}, req_id=rid)

    assert ws.send.call_count == 1
    assert not bot._in_trade


def test_no_second_proposal_while_first_pending(bot):
    """Garante que não é enviada segunda proposta enquanto a primeira está pendente."""
    ws = MagicMock()
    proposal(bot, ws)
    proposal(bot, ws)
    assert ws.send.call_count == 1


def test_heartbeat_timeout_closes_stale_websocket(bot, tmp_path, monkeypatch):
    """Garante que a falha de heartbeat fecha o websocket e grava o evento heartbeat_timeout no status."""
    monkeypatch.setattr(executor, '_BOT_STATUS_JSON', tmp_path/'bot_status.json')
    ws = MagicMock()
    bot._ws = ws
    bot._last_tick_epoch = 123456

    bot._on_heartbeat_fail()

    ws.close.assert_called_once()
    status = json.loads((tmp_path/'bot_status.json').read_text())
    assert status['event'] == 'heartbeat_timeout'
    assert status['last_tick_epoch'] == 123456


def test_proposal_retries_once_with_lower_stake_when_payout_exceeds_api_cap(bot, tmp_path, monkeypatch):
    """Garante que payout acima do teto da API gera uma nova proposta com stake menor."""
    monkeypatch.setattr(executor, '_BOT_STATUS_JSON', tmp_path/'bot_status.json')
    ws = MagicMock()
    bot.risk_manager.balance = 7566.79
    bot._ws = ws
    rid = proposal(bot, ws, 'SELL')
    first = json.loads(ws.send.call_args.args[0])
    assert first['amount'] == pytest.approx(49.0)
    bot._handle_api_error({'req_id': rid, 'error': {
        'code': 'ContractBuyValidationError',
        'message': 'Minimum stake of 0.50 and maximum payout of 100.00. Current payout is 142.26.',
    }})
    assert bot._phase == 'proposal'
    assert bot._proposal_retry_count == 1
    assert ws.send.call_count == 2
    second = json.loads(ws.send.call_args.args[0])
    assert second['amount'] == pytest.approx(33.75)
    assert second['amount'] < first['amount']
    assert second['req_id'] == bot._proposal_req_id != rid
    assert bot._decision_history[-1]['reason'] == 'proposal_retry_payout_cap'


def test_initial_proposal_caps_stake_to_avoid_api_payout_limit(bot):
    """Garante que a primeira proposta já respeita o teto preventivo de payout."""
    ws = MagicMock()
    bot.risk_manager.balance = 7726.39

    proposal(bot, ws, 'BUY')

    sent = json.loads(ws.send.call_args.args[0])
    assert sent['amount'] == pytest.approx(49.0)
    assert sent['amount'] < 77.26


def test_repeated_payout_cap_rejections_request_market_switch(bot, monkeypatch):
    """Garante que rejeições repetidas por teto de payout acionam o callback de mercado desfavorável."""
    callback = MagicMock()
    monkeypatch.setattr(config, "PROPOSAL_API_ERROR_SWITCH_THRESHOLD", 3, raising=False)
    bot.on_market_unfavorable = callback
    bot.risk_manager.balance = 7566.79
    error = {
        'code': 'ContractBuyValidationError',
        'message': 'Minimum stake of 0.50 and maximum payout of 100.00. Current payout is 142.26.',
    }

    for _ in range(3):
        ws = MagicMock()
        bot._ws = ws
        rid = proposal(bot, ws, 'SELL')
        bot._handle_api_error({'req_id': rid, 'error': error})
        retry_req_id = bot._proposal_req_id
        bot._handle_api_error({'req_id': retry_req_id, 'error': error})

    callback.assert_called_once()
    assert callback.call_args.args[:3] == (bot.symbol, "proposal_payout_cap_rejected", 3)


def test_entry_cadence_disabled_evaluates_without_new_closed_candle(bot, monkeypatch):
    """Garante que, sem cadência de entrada, o tick avalia sinal e envia proposta sem nova vela fechada."""
    monkeypatch.setattr(config, "ENFORCE_ENTRY_CADENCE", False, raising=False)
    monkeypatch.setattr(config, "ENFORCE_RISK_GATE", False, raising=False)
    bot._api_history_loaded = True
    bot._cur_candle_epoch = 1700000000
    bot._cur_candle_open = 100.0
    bot._cur_candle_high = 100.0
    bot._cur_candle_low = 100.0
    bot._cur_candle_close = 100.0
    bot._last_tick_epoch = 1700000000
    bot._candle_closes.extend([100.0 + i * 0.01 for i in range(config.MIN_CANDLES)])
    bot._candles.extend([
        {"epoch": 1699990000 + i * 60, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0}
        for i in range(config.MIN_CANDLES)
    ])
    send = MagicMock(return_value=True)
    monkeypatch.setattr(bot, "_send_proposal", send)
    monkeypatch.setattr(executor, "get_signal", MagicMock(return_value=("BUY", {"model_id": "offline", "win_probability": .8})))

    bot._handle_tick(MagicMock(), {"symbol": bot.symbol, "epoch": 1700000001, "quote": 100.1})

    assert bot._new_candle_closed is False
    send.assert_called_once()


def test_risk_gate_disabled_allows_proposal_when_risk_manager_blocks(bot, monkeypatch):
    """Garante que, com o gate de risco desligado, a proposta é enviada mesmo que o RiskManager bloqueie."""
    ws = MagicMock()
    monkeypatch.setattr(config, "ENFORCE_RISK_GATE", False, raising=False)
    bot.risk_manager.can_trade = MagicMock(return_value=False)

    sent = bot._send_proposal(
        ws,
        "BUY",
        {"win_probability": .8, "ai_confidence": .8, "model_id": "offline"},
    )

    assert sent is True
    assert ws.send.call_count == 1


def test_risk_gate_enabled_blocks_proposal_when_risk_manager_blocks(bot, monkeypatch):
    """Garante que, com o gate de risco ligado, o bloqueio do RiskManager impede a proposta e é registrado."""
    ws = MagicMock()
    monkeypatch.setattr(config, "ENFORCE_RISK_GATE", True, raising=False)
    bot.risk_manager.can_trade = MagicMock(return_value=False)

    sent = bot._send_proposal(
        ws,
        "BUY",
        {"win_probability": .8, "ai_confidence": .8, "model_id": "offline"},
    )

    assert sent is False
    ws.send.assert_not_called()
    assert bot._decision_history[-1]["reason"] == "proposal_precheck_block"


@pytest.mark.parametrize('rid', [None, - 1])
def test_unsolicited_proposal_is_ignored(bot, rid):
    """Garante que proposta com req_id ausente ou diferente do pendente é ignorada."""
    ws = MagicMock()
    proposal(bot, ws)
    ws.reset_mock()
    bot._handle_proposal(ws, {'id': 'other', 'ask_price': 10, 'payout': 18.5}, req_id = rid)
    ws.send.assert_not_called()


def test_duplicate_proposal_cannot_buy_twice(bot):
    """Garante que a mesma proposta recebida duas vezes gera apenas uma compra."""
    ws = MagicMock()
    rid = proposal(bot, ws)
    ws.reset_mock()
    p = {'id': 'p', 'ask_price': 10, 'payout': 18.5}
    bot._handle_proposal(ws, p, req_id = rid)
    bot._handle_proposal(ws, p, req_id = rid)
    assert ws.send.call_count == 1
    assert bot._phase == 'buy'


@pytest.mark.parametrize('price,payout', [(10, 10.01), (10, None), (10, float('nan')), (11, 25), (0, 20)])
def test_invalid_or_unprofitable_proposal_not_bought(bot, price, payout):
    """Garante que proposta inválida ou sem retorno esperado positivo não é comprada."""
    ws = MagicMock()
    rid = proposal(bot, ws)
    ws.reset_mock()
    bot._handle_proposal(ws, {'id': 'p', 'ask_price': price, 'payout': payout}, req_id = rid)
    ws.send.assert_not_called()
    assert not bot._in_trade


def test_expired_proposal_is_rejected(bot):
    """Garante que proposta expirada não é comprada e o bot volta a idle."""
    ws = MagicMock()
    rid = proposal(bot, ws)
    ws.reset_mock()
    bot._pending_timestamp = time.time() - 30
    bot._handle_proposal(ws, {'id': 'p', 'ask_price': 10, 'payout': 18.5}, req_id = rid)
    ws.send.assert_not_called()
    assert bot._phase == 'idle'


@pytest.mark.parametrize('probability,payout,accepted', [
    (.56, 18.5, True),   # Ganho esperado positivo.
    (.60, 16.8, False),  # Payout líquido abaixo do piso shadow de 70%.
    (.625, 16.0, False), # Payout líquido abaixo do piso shadow de 70%.
    (.6249, 16.0, False),
    (.56, 17.0, False),  # Confiança >55% não basta para este payout.
])
def test_buy_requires_nonnegative_expected_return(bot, probability, payout, accepted):
    """Garante que a compra só ocorre quando o retorno esperado é não negativo para a probabilidade e o payout."""
    ws = MagicMock()
    bot._send_proposal(ws, 'BUY', {'win_probability': probability,
                                 'ai_confidence': probability, 'model_id': 'offline'})
    rid = bot._proposal_req_id
    ws.reset_mock()
    bot._handle_proposal(ws, {'id': 'p', 'ask_price': 10, 'payout': payout}, req_id=rid)
    assert (bot._phase == 'buy') is accepted
    assert ws.send.call_count == int(accepted)


@pytest.mark.parametrize('probability,proposal_sent,economic', [(.45, False, False), (.450001, True, False), (.500001, True, True)])
def test_confidence_must_be_strictly_above_approval_floor(bot, probability, proposal_sent, economic):
    """Garante que a proposta só é enviada com confiança estritamente acima do piso de aprovação."""
    from model_contract import profitable_proposal
    ws = MagicMock()
    bot._send_proposal(ws, 'BUY', {'win_probability': probability,
                                 'ai_confidence': probability, 'model_id': 'offline'})
    assert ws.send.call_count == int(proposal_sent)
    assert profitable_proposal(10, 20, probability) is economic


def test_delayed_buy_response_reconciles_only_matching_request(bot):
    """Garante que a resposta de compra atrasada só reconcilia quando o req_id coincide e não inventa preço de entrada."""
    ws = MagicMock()
    rid = proposal(bot, ws)
    bot._handle_proposal(ws, {'id': 'p', 'ask_price': 10, 'payout': 18.5}, req_id = rid)
    bot._phase = 'reconciling'
    bot._handle_buy(ws, {'contract_id': 555}, req_id = - 1)
    assert bot._open_contract_id is None
    bot._handle_buy(ws, {'contract_id': 12345}, req_id = bot._buy_req_id)
    assert bot._open_contract_id == '12345'
    assert bot._entry_price == 0  # não inventa preço de entrada a partir do tick local


def test_open_contract_update_persists_entry_point(bot):
    """Garante que a atualização do contrato aberto persiste preço e época de entrada no estado."""
    ws = MagicMock()
    buy(bot, ws)
    bot._handle_contract_update({
        'contract_id': 12345,
        'is_sold': 0,
        'entry_spot': '100.25',
        'entry_tick_time': 1700000000,
    })

    state = json.loads(executor._STATE_JSON.read_text())
    active = state['active_contract']
    assert active['entry_price'] == 100.25
    assert active['entry_epoch'] == 1700000000
    assert bot._in_trade


@pytest.mark.parametrize('profit,result', [(2.5, 'WIN'), (-10, 'LOSS'), (0, 'VOID')])
def test_settlement_updates_balance_once(bot, profit, result):
    """Garante que a liquidação atualiza o saldo e o CSV uma única vez, mesmo com evento duplicado."""
    ws = MagicMock()
    buy(bot, ws)
    event = {'contract_id': 12345, 'is_sold': 1, 'profit': str(profit), 'entry_spot': '100',
           'exit_spot': '101', 'date_start': 1000, 'date_expiry': 1900}
    bot._handle_contract_update(event)
    bot._handle_contract_update(event)
    assert bot.risk_manager.balance == 1000 + profit
    assert not bot._in_trade
    with (bot.risk_manager._path.parent / 'operations.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1 and rows[0]['result'] == result
    assert rows[0]['entry_price'] == '100' and rows[0]['contract_id'] == '12345'


@pytest.mark.parametrize('event', [
    {'contract_id': 99999, 'is_sold': 1, 'profit': 10},
    {'contract_id': 12345, 'is_sold': 0, 'profit': 10},
    {'contract_id': 12345, 'is_sold': 1},
    {'contract_id': 12345, 'is_sold': 1, 'profit': 'nan'},
])
def test_invalid_or_unrelated_settlement_not_counted(bot, event):
    """Garante que liquidação inválida ou de outro contrato não altera saldo nem encerra a operação."""
    ws = MagicMock()
    buy(bot, ws)
    bot._handle_contract_update(event)
    assert bot._in_trade and bot.risk_manager.balance == 1000


def test_restart_restores_open_contract_and_prevents_new_order(bot):
    """Garante que reiniciar restaura o contrato aberto e impede nova ordem."""
    ws = MagicMock()
    buy(bot, ws)
    restored = DerivBot(RiskManager(9000.))
    assert restored._open_contract_id == '12345'
    assert restored._pending_stake == 10 and restored._in_trade
    restored._send_proposal(ws, 'BUY', {'win_probability': .9})
    assert restored._phase == 'open'


def test_restart_preserves_uncertain_buy(bot):
    """Garante que reiniciar preserva uma compra incerta na fase buy sem contrato associado."""
    ws = MagicMock()
    rid = proposal(bot, ws)
    bot._handle_proposal(ws, {'id': 'p', 'ask_price': 10, 'payout': 18.5}, req_id = rid)
    restored = DerivBot(RiskManager(1000.))
    assert restored._in_trade and restored._phase == 'buy' and restored._open_contract_id is None


def test_wrong_account_mode_refused(bot):
    """Garante que conta de modo incorreto (não virtual) é recusada com ValueError e fecha o websocket."""
    ws = MagicMock()
    with pytest.raises(ValueError):
        bot._handle_authorize(ws, {'is_virtual': 0, 'loginid': 'test', 'balance': 1000})
    ws.close.assert_called_once()
    assert not bot._auth_verified


def test_history_is_merged_without_duplicates(bot):
    """Garante que o histórico de velas é mesclado sem duplicatas e em ordem."""
    candles = [dict(epoch = i * config.CANDLE_TIMEFRAME_SEC, open = 100 + i, high = 101 + i, low = 99 + i, close = 100 + i) for i in range(200)]
    bot._merge_candles(candles)
    bot._merge_candles(candles[-99:])
    assert len(bot._candles) == 200 and len(set(bot._candle_epochs)) == 200
    assert list(bot._candle_epochs) == sorted(bot._candle_epochs)


def test_preload_ignores_invalid_local_history(bot, tmp_path, monkeypatch, capsys):
    """Garante que CSV local inválido não derruba o bot antes do histórico autenticado."""
    import asset_history

    class EmptyStore:
        def __init__(self, *_args, **_kwargs):
            pass

        def read(self, _symbol):
            return []

    ticks_path = tmp_path / 'ticks.csv'
    ticks_path.write_text(
        'epoch,datetime,symbol,price,kind,timeframe,open,high,low\n'
        '1790878563,,R_100,158.231,candle,900,158.206,158.25,158.124\n'
    )
    monkeypatch.setattr(executor, 'TICKS_CSV', str(ticks_path))
    monkeypatch.setattr(asset_history, 'AssetHistoryStore', EmptyStore)

    bot._preload_candles_from_csv()

    assert list(bot._candles) == []
    assert 'Histórico local inválido' in capsys.readouterr().out


def test_preload_uses_shared_asset_history_when_ticks_csv_is_absent(bot, monkeypatch):
    """O buffer comum do executor recebe candles do histórico isolado do ativo."""
    import asset_history
    from pathlib import Path

    rows = [
        {'epoch': 1_800_000_000 + index * config.CANDLE_TIMEFRAME_SEC,
         'open': 100 + index, 'high': 101 + index, 'low': 99 + index,
         'price': 100.5 + index}
        for index in range(3)
    ]

    class Store:
        def __init__(self, *_args, **_kwargs):
            pass

        def read(self, symbol):
            assert symbol == bot.symbol
            return rows

    monkeypatch.setattr(asset_history, 'AssetHistoryStore', Store)
    monkeypatch.setattr(executor, 'TICKS_CSV', str(Path('/nonexistent/ticks_v2.csv')))
    monkeypatch.setattr(executor.time, 'time', lambda: 1_800_010_000)

    bot._preload_candles_from_csv()

    assert list(bot._candle_epochs) == [row['epoch'] for row in rows]
    assert list(bot._candle_closes) == [row['price'] for row in rows]


def test_active_model_module_keeps_executor_gates_even_without_majority_vote():
    """IA ativa mantém os gates do executor mesmo quando não apoiou a direção vencedora."""
    assert executor.DerivBot._uses_model_decision({
        'enabled_decision_modules': ['trained_model', 'weighted_score'],
        'decision_modules': ['weighted_score'],
    })
    assert not executor.DerivBot._uses_model_decision({
        'enabled_decision_modules': ['weighted_score', 'price_action'],
        'decision_modules': ['weighted_score', 'price_action'],
    })


def test_wrong_symbol_tick_does_not_contaminate_buffer(bot):
    """Garante que tick de outro símbolo não altera o buffer de velas nem o último preço."""
    before = list(bot._candle_closes)
    bot._handle_tick(MagicMock(), {'symbol': 'wrong', 'epoch': int(time.time()) + 1, 'quote': 50})
    assert list(bot._candle_closes) == before and bot._last_tick_price == 100


def test_reconnect_balance_does_not_double_count_settlement(bot):
    """Garante que o saldo verificado na reconexão não duplica a liquidação pendente."""
    ws = MagicMock()
    buy(bot, ws)
    # A API já inclui +8; estado local ainda aguarda a liquidação.
    bot._verify_account('test-account', 1008.)
    bot._handle_contract_update({'contract_id': 12345, 'is_sold': 1, 'profit': 8})
    assert bot.risk_manager.balance == 1008


def test_buy_error_does_not_authorize_retry(bot):
    """Garante que erro na compra mantém a operação em reconciliação e não libera nova tentativa."""
    ws = MagicMock()
    rid = proposal(bot, ws)
    bot._handle_proposal(ws, {'id': 'p', 'ask_price': 10, 'payout': 18.5}, req_id = rid)
    bot._on_message(ws, json.dumps({'error': {'code': 'InternalError'}, 'req_id': bot._buy_req_id}))
    assert bot._in_trade and bot._phase == 'reconciling'


def test_uncertain_buy_reconciles_open_contract_from_portfolio(bot):
    """Garante que compra incerta encontrada no portfólio é reconciliada e passa a acompanhar o contrato aberto."""
    ws = MagicMock()
    rid = proposal(bot, ws, 'BUY')
    bot._handle_proposal(ws, {'id': 'p', 'ask_price': 10, 'payout': 18.5}, req_id=rid)
    bot._phase = 'reconciling'
    bot._portfolio_req_id = 777
    ws.reset_mock()

    bot._on_message(ws, json.dumps({
        'msg_type': 'portfolio',
        'req_id': 777,
        'portfolio': {
            'contracts': [
                {'contract_id': 55555, 'symbol': bot.symbol, 'contract_type': 'CALL'},
            ],
        },
    }))

    assert bot._open_contract_id == '55555'
    assert bot._phase == 'open'
    sent = json.loads(ws.send.call_args.args[0])
    assert sent == {'proposal_open_contract': 1, 'contract_id': 55555, 'subscribe': 1}
    assert bot._decision_history[-1]['reason'] == 'buy_reconciled_from_portfolio'


def test_uncertain_buy_not_in_portfolio_releases_bot(bot):
    """Garante que compra incerta ausente do portfólio libera o bot para novos sinais."""
    ws = MagicMock()
    rid = proposal(bot, ws, 'SELL')
    bot._handle_proposal(ws, {'id': 'p', 'ask_price': 10, 'payout': 18.5}, req_id=rid)
    bot._phase = 'reconciling'
    bot._portfolio_req_id = 778

    bot._on_message(ws, json.dumps({
        'msg_type': 'portfolio',
        'req_id': 778,
        'portfolio': {'contracts': []},
    }))

    assert not bot._in_trade
    assert bot._phase == 'idle'
    assert bot._open_contract_id is None
    assert bot._decision_history[-1]['reason'] == 'uncertain_buy_not_in_portfolio'


def test_proposal_error_releases_bot_for_next_signal(bot):
    """Garante que erro da API na proposta libera o bot e registra o motivo da rejeição."""
    ws = MagicMock()
    rid = proposal(bot, ws, 'SELL')
    bot._on_message(ws, json.dumps({
        'error': {
            'code': 'ContractBuyValidationError',
            'message': 'Price moved too much',
        },
        'req_id': rid,
    }))
    assert not bot._in_trade
    assert bot._phase == 'idle'
    assert bot._open_contract_id is None
    assert bot._decision_history[-1]['reason'] == 'proposal_rejected_by_api'


def test_market_unavailable_proposal_error_requests_fallback(bot):
    """Garante que erro MarketIsClosed na proposta aciona o callback de mercado indisponível e libera o bot."""
    ws = MagicMock()
    callback = MagicMock()
    bot.on_market_unavailable = callback
    bot.symbol = 'frxEURGBP'
    rid = proposal(bot, ws, 'BUY')
    bot._on_message(ws, json.dumps({
        'error': {
            'code': 'MarketIsClosed',
            'message': 'Market is closed',
        },
        'req_id': rid,
    }))
    callback.assert_called_once_with('frxEURGBP')
    assert not bot._in_trade
    assert bot._phase == 'idle'


def test_unfavorable_symbol_streak_requests_fallback(bot, monkeypatch, tmp_path):
    """Garante que a sequência de decisões desfavoráveis aciona o callback de troca só ao atingir o limite."""
    callback = MagicMock()
    monkeypatch.setattr(executor, '_BOT_STATUS_JSON', tmp_path/'bot_status.json')
    monkeypatch.setattr(config, 'UNFAVORABLE_SYMBOL_SWITCH_DECISIONS', 3)
    monkeypatch.setattr(config, 'UNFAVORABLE_SYMBOL_SWITCH_REASONS', {'no_technical_signal'})
    bot.on_market_unfavorable = callback
    bot.symbol = 'frxGBPJPY'

    for epoch in (100, 160):
        bot._record_decision(
            reason='no_technical_signal',
            tick_epoch=epoch,
            indicators={'adx': 12.5},
        )

    callback.assert_not_called()

    bot._record_decision(
        reason='no_technical_signal',
        tick_epoch=220,
        indicators={'adx': 12.7},
    )

    callback.assert_called_once_with(
        'frxGBPJPY',
        'no_technical_signal',
        3,
        {'adx': 12.7},
    )


def test_reconnection_refetches_bearer_account_balance(bot, monkeypatch):
    """Garante que a reconexão com bearer rebusca o saldo da conta e preserva o lucro diário."""
    import hashlib
    bot.auth_mode = 'bearer'
    bot.account_id = 'virtual-test'
    bot.risk_manager.bind_account(hashlib.sha256(b'virtual-test').hexdigest(), 1000)
    bot.risk_manager.record_result('R_100', 'BUY', 10, 15, - 10, {})
    monkeypatch.setattr(executor.deriv_session, 'get_options_accounts', lambda ** k: [{'account_id': 'virtual-test', 'account_type': 'demo', 'balance': 990}])
    ws = MagicMock()
    bot._on_close(ws, 1000, '')
    bot._on_open(ws)
    assert bot.risk_manager.balance == 990 and bot.risk_manager._daily_profit == - 10


def test_expired_open_contract_remains_tracked_on_tick(bot, monkeypatch):
    """Garante que contrato aberto além do prazo continua rastreado e consultado no tick."""
    ws = MagicMock()
    buy(bot, ws)
    bot._buy_timestamp = time.time() - 10000
    bot._contract_poll_timer = None
    monkeypatch.setattr(bot, '_poll_contract_result', MagicMock())
    bot._handle_tick(ws, {'symbol': bot.symbol, 'epoch': bot._last_tick_epoch + 1, 'quote': 100})
    assert bot._in_trade and bot._open_contract_id == '12345'
    bot._poll_contract_result.assert_called_once_with('12345')


def test_market_switch_blocks_new_proposals(bot):
    """Garante que troca de mercado solicitada bloqueia novas propostas."""
    ws = MagicMock()
    bot.market_switch_requested = lambda: True
    proposal(bot, ws)
    ws.send.assert_not_called()
    assert bot._market_switch_ready()


def test_market_switch_cancels_unpurchased_proposal(bot):
    """Garante que troca de mercado solicitada cancela proposta ainda não comprada."""
    ws = MagicMock()
    rid = proposal(bot, ws)
    bot.market_switch_requested = lambda: True
    ws.reset_mock()
    bot._handle_proposal(ws, {'id': 'quote', 'ask_price': 10, 'payout': 18.5}, req_id=rid)
    ws.send.assert_not_called()
    assert bot._market_switch_ready()


def test_market_switch_preserves_open_contract_until_settlement(bot):
    """Garante que a troca de mercado espera a liquidação do contrato aberto antes de liberar."""
    ws = MagicMock()
    buy(bot, ws)
    bot.market_switch_requested = lambda: True
    assert not bot._market_switch_ready()
    bot._on_message(ws, json.dumps({'msg_type': 'proposal_open_contract',
        'proposal_open_contract': {'contract_id': 12345, 'is_sold': 1, 'profit': 8.5}}))
    assert bot._market_switch_ready()
    assert bot.risk_manager.balance == 1008.5


def test_market_switch_keeps_uncertain_purchase_for_reconciliation(bot):
    """Garante que a troca de mercado não libera enquanto uma compra incerta está em reconciliação."""
    ws = MagicMock()
    rid = proposal(bot, ws)
    bot._handle_proposal(ws, {'id': 'quote', 'ask_price': 10, 'payout': 18.5}, req_id=rid)
    bot.market_switch_requested = lambda: True
    bot._on_message(ws, json.dumps({'req_id': bot._buy_req_id, 'error': {'code': 'Disconnected'}}))
    assert bot._phase == 'reconciling'
    assert not bot._market_switch_ready()


@pytest.mark.parametrize('demo,payout,accepted', [
    (True, 15.99, False), (True, 16, False), (True, 17, True), (True, 18.5, True),
    (False, 15.99, False), (False, 16, False), (False, 17, False), (False, 18.5, False),
])
def test_model_shadow_gate_requires_demo_and_70_percent_net_payout(bot,demo,payout,accepted):
    """Garante demo-only e payout líquido mínimo de 70% para modelos validados em shadow."""
    bot.demo=demo
    ws=MagicMock()
    rid=proposal(bot,ws)
    ws.reset_mock()
    bot._handle_proposal(ws,{'id':'quote','ask_price':10,'payout':payout},req_id=rid)
    purchases=[json.loads(c.args[0]) for c in ws.send.call_args_list if 'buy' in json.loads(c.args[0])]
    assert bool(purchases) is accepted


@pytest.mark.parametrize('price,payout',[(.9,1.44),(10,16),(.5,.8)])
def test_exact_sixty_percent_payout_not_lost_to_float_rounding(price,payout):
    """Garante que payout de exatamente 60% é aceito sem perda por arredondamento de ponto flutuante."""
    from model_contract import profitable_proposal
    assert profitable_proposal(price,payout,.9)
    assert not profitable_proposal(price,payout-.00001,.9)
