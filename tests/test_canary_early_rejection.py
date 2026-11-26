"""Reprovação por impossibilidade, migração e troca sem estado financeiro real."""
import itertools
import json
import threading
from unittest.mock import Mock, MagicMock

import pytest

import config
import pipeline
from model_contract import fingerprint
from training_canary import saved_canary
from shadow_monitor import ShadowMonitor
from tests.test_training_canary import ENTRY, IDENTITY, evidence, tracker, observe, settle


def old_partial(path, *, failed=3, pending=True):
    """Constrói o formato válido produzido antes da reprovação antecipada."""
    monitor = tracker(path)
    original = json.loads(monitor.canary.path.read_text())
    results = [dict(entry_epoch=ENTRY + i * 900, expiry_epoch=ENTRY + (i + 1) * 900,
                    entry_price=100, direction=1, confidence=.8, net_payout=.73,
                    exit_epoch=ENTRY + (i + 1) * 900, exit_price=99, correct=False)
               for i in range(failed)]
    prediction = (dict(entry_epoch=ENTRY + failed * 900, expiry_epoch=ENTRY + (failed + 1) * 900,
                       entry_price=100, direction=1, confidence=.8, net_payout=.73)
                  if pending else None)
    original.update(results=results, pending=prediction)
    monitor.canary.path.write_text(json.dumps(original))
    legacy = json.loads(path.read_text()) if path.exists() else dict(
        schema_version=1, **IDENTITY, approved_at_epoch=None, rejected_at_epoch=None, archives=[])
    legacy.update(results=results, pending=prediction)
    path.write_text(json.dumps(legacy))
    return monitor.canary.path, original


def test_second_loss_rejects_immediately_and_never_approves_or_retries(tmp_path):
    path = tmp_path / 'shadow.json'
    monitor = tracker(path)
    assert settle(monitor, 0, correct=False) == 'settled'
    assert not monitor.rejected
    assert monitor.diagnostics()['maximum_possible_wins'] == 7
    assert settle(monitor, 1, correct=False) == 'rejected'
    assert monitor.rejected and not monitor.approved
    assert monitor.state['rejection_reason'] == 'minimum_wins_unreachable'
    assert monitor.state['rejected_at_epoch'] == ENTRY + 1800
    assert monitor.diagnostics()['maximum_possible_wins'] == 6
    before = monitor.canary.path.read_bytes()
    assert observe(monitor, ENTRY + 1800) == 'already_pending_or_approved'
    assert monitor.settle_tick(epoch=ENTRY + 9000, price=101) == 'no_pending'
    assert monitor.canary.path.read_bytes() == before
    resumed = tracker(path)
    assert resumed.rejected and resumed.result_count == 2
    assert resumed.state['frozen_at_epoch'] == ENTRY


def test_every_possible_eight_result_sequence_preserves_acceptance_and_stops_when_impossible(tmp_path):
    for index, sequence in enumerate(itertools.product((False, True), repeat=8)):
        path = tmp_path / str(index) / 'shadow.json'
        monitor = tracker(path)
        wins = 0
        for count, correct in enumerate(sequence, start=1):
            result = settle(monitor, count - 1, correct=correct)
            wins += int(correct)
            impossible = wins + 8 - count < 7
            assert monitor.rejected == impossible
            resumed = tracker(path)
            assert not resumed.canary.invalid
            assert resumed.rejected == monitor.rejected and resumed.approved == monitor.approved
            assert resumed.result_count == count
            if impossible:
                assert result == 'rejected' and not monitor.approved
                break
            assert monitor.approved == (count == 8 and wins >= 7)
        assert monitor.approved == (sum(sequence) >= 7)


def test_rejected_identity_stays_rejected_after_switching_away_and_returning(tmp_path):
    path = tmp_path / 'shadow.json'
    first = tracker(path)
    settle(first, 0, correct=False)
    settle(first, 1, correct=False)
    rejected = first.canary.path.read_bytes()
    tracker(path, evidence(model_id='different'), model_id='different')
    resumed = tracker(path)
    assert resumed.rejected and not resumed.approved
    assert resumed.result_count == resumed.legacy.result_count == 2
    assert resumed.canary.path.read_bytes() == rejected


def test_one_loss_and_seven_wins_still_requires_eighth_result(tmp_path):
    monitor = tracker(tmp_path / 'shadow.json')
    assert settle(monitor, 0, correct=False) == 'settled'
    for index in range(1, 7):
        assert settle(monitor, index) == 'settled'
        assert not monitor.approved and not monitor.rejected
    assert settle(monitor, 7) == 'approved'
    assert tracker(tmp_path / 'shadow.json').approved


def test_read_only_lookup_infers_failure_without_touching_old_files(tmp_path, monkeypatch):
    path = tmp_path / 'shadow.json'
    target, original = old_partial(path)
    before, legacy_before = target.read_bytes(), path.read_bytes()
    result = saved_canary(path, **IDENTITY)
    assert result['rejected_at_epoch'] == original['results'][1]['exit_epoch']
    assert result['rejection_reason'] == 'minimum_wins_unreachable'
    assert result['results'] == original['results'] and result['pending'] == original['pending']
    assert target.read_bytes() == before and path.read_bytes() == legacy_before
    monkeypatch.setattr(pipeline, '_SHADOW_STATE_JSON', path)
    monkeypatch.setattr(pipeline, 'model_timeframe_sec', lambda: 300)
    assert not pipeline._training_model_resumable(evidence())
    assert target.read_bytes() == before and path.read_bytes() == legacy_before


def test_runtime_migrates_old_impossible_state_preserving_pending_and_legacy_bytes(tmp_path):
    path = tmp_path / 'shadow.json'
    target, original = old_partial(path)
    legacy_before = path.read_bytes()
    resumed = tracker(path)
    assert resumed.rejected and not resumed.canary.invalid
    assert resumed.result_count == 3
    assert resumed.state['pending'] == original['pending']
    assert resumed.state['results'] == original['results']
    assert resumed.state['rejected_at_epoch'] == original['results'][1]['exit_epoch']
    assert path.read_bytes() == legacy_before
    terminal = target.read_bytes()
    assert resumed.settle_tick(epoch=ENTRY + 3600, price=101) == 'rejected'
    assert resumed.settle_tick(epoch=ENTRY + 3601, price=101) == 'no_pending'
    assert target.read_bytes() == terminal and path.read_bytes() == legacy_before
    assert saved_canary(path, **IDENTITY)['rejected_at_epoch'] is not None


def test_old_viable_state_is_not_rewritten_or_reset(tmp_path):
    path = tmp_path / 'shadow.json'
    monitor = tracker(path)
    settle(monitor, 0, correct=False)
    observe(monitor, ENTRY + 900)
    before, legacy_before = monitor.canary.path.read_bytes(), path.read_bytes()
    resumed = tracker(path)
    assert not resumed.rejected and not resumed.approved
    assert resumed.state['pending'] == monitor.state['pending']
    assert resumed.canary.path.read_bytes() == before and path.read_bytes() == legacy_before


def test_forged_early_rejection_without_impossibility_fails_closed(tmp_path):
    path = tmp_path / 'shadow.json'
    monitor = tracker(path)
    settle(monitor, 0)
    state = monitor.state.copy()
    state.update(rejected_at_epoch=ENTRY + 900, rejection_reason='minimum_wins_unreachable')
    monitor.canary.path.write_text(json.dumps(state))
    resumed = tracker(path)
    assert resumed.canary.invalid and not resumed.approved


def test_real_contract_pending_prevents_startup_rotation_and_mixed_modes_are_untouched(monkeypatch):
    reader = Mock(side_effect=AssertionError('não consultar estado antes da reconciliação'))
    monkeypatch.setattr(pipeline, 'model_timeframe_sec', lambda: 300)
    monkeypatch.setattr(pipeline.Path, 'read_text', reader)
    assert not pipeline._request_rejected_canary_switch(pending=dict(phase='reconciling'))
    reader.assert_not_called()
    monkeypatch.setattr(pipeline, 'model_timeframe_sec', lambda: 900)
    assert not pipeline._request_rejected_canary_switch()
    reader.assert_not_called()


def test_startup_recognizes_old_impossible_active_bundle_and_requests_market_switch(tmp_path, monkeypatch):
    path = tmp_path / 'shadow.json'
    model = tmp_path / 'model.pkl'
    model.write_bytes(b'offline artifact, no pickle loading')
    model_id = fingerprint(model)
    metadata = evidence(model_id=model_id, artifact_sha256=model_id)
    monitor = tracker(path, metadata, model_id=model_id)
    settle(monitor, 0, correct=False)
    settle(monitor, 1, correct=False)
    # Simula estado antigo ainda sem marcação final.
    state = json.loads(monitor.canary.path.read_text())
    state['rejected_at_epoch'] = None
    state.pop('rejection_reason')
    monitor.canary.path.write_text(json.dumps(state))
    model.with_name('model.pkl.metadata.json').write_text(json.dumps(metadata))
    event = threading.Event()
    monkeypatch.setattr(pipeline, '_market_switch', event)
    monkeypatch.setattr(pipeline, '_market_switch_reason', '')
    monkeypatch.setattr(pipeline, '_runtime_symbol', lambda: 'R_100')
    monkeypatch.setattr(pipeline, 'model_timeframe_sec', lambda: 300)
    monkeypatch.setattr(pipeline, 'AI_MODEL_PATH', str(model))
    monkeypatch.setattr(pipeline, '_SHADOW_STATE_JSON', path)
    before = monitor.canary.path.read_bytes()
    assert pipeline._request_rejected_canary_switch()
    assert event.is_set() and pipeline._market_switch_reason == 'shadow_validation_failed'
    assert monitor.canary.path.read_bytes() == before
    with pytest.raises(pipeline.MarketUnavailable): pipeline._check_market_switch()
    event.clear()
    model.write_bytes(b'different model hash')
    assert not pipeline._request_rejected_canary_switch()
    assert not event.is_set()


def test_main_session_checks_rejection_before_starting_collection_or_training(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'USE_AI_MODEL', True)
    for key in ('USE_WEIGHTED_SIGNAL', 'PRICE_ACTION_FILTER', 'CANDLE_PATTERN_FILTER'):
        monkeypatch.setattr(config, key, False)
    monkeypatch.setattr(pipeline.sys, 'argv', ['pipeline.py', '--demo', '--skip-collect'])
    monkeypatch.setattr(pipeline, '_validate_deriv_token', lambda **kwargs: True)
    monkeypatch.setattr(pipeline, 'TICKS_CSV', str(tmp_path / 'ticks.csv'))
    monkeypatch.setattr(pipeline, '_training_registry', lambda: Mock(activate=Mock(return_value=False)))
    monkeypatch.setattr(pipeline, '_first_live_symbol_with_compatible_model', lambda symbols: '')
    monkeypatch.setattr(pipeline, '_scan_candidates', lambda: [])
    monkeypatch.setattr(pipeline, '_prepare_training_start', Mock())
    monkeypatch.setattr(pipeline, '_set_runtime_symbol', Mock())
    monkeypatch.setattr(pipeline, '_runtime_symbol', lambda: 'R_100')
    event = threading.Event()
    monkeypatch.setattr(pipeline, '_market_switch', event)
    def reject(**kwargs):
        assert not kwargs['pending']
        event.set()
        return True
    check = Mock(side_effect=reject)
    monkeypatch.setattr(pipeline, '_request_rejected_canary_switch', check)
    collector = Mock(side_effect=AssertionError('não iniciar coletor do bundle reprovado'))
    trainer = Mock(side_effect=AssertionError('não retreinar antes de pedir a troca'))
    monkeypatch.setattr(pipeline, '_CollectorThread', collector)
    monkeypatch.setattr(pipeline, '_run_training', trainer)
    with pytest.raises(pipeline.MarketUnavailable): pipeline._main_session({'lock': Mock()})
    check.assert_called_once()
    collector.assert_not_called()
    trainer.assert_not_called()


def test_executor_requests_switch_once_and_never_sends_trade_on_second_loss(tmp_path, monkeypatch):
    import executor
    from risk_manager import RiskManager
    monkeypatch.setattr(config, 'USE_AI_MODEL', True)
    for key in ('USE_WEIGHTED_SIGNAL', 'PRICE_ACTION_FILTER', 'CANDLE_PATTERN_FILTER'):
        monkeypatch.setattr(config, key, False)
    path = tmp_path / 'shadow.json'
    monitor = tracker(path)
    settle(monitor, 0, correct=False)
    observe(monitor, ENTRY + 900)
    bot = executor.DerivBot(RiskManager(1000), demo=True)
    bot.symbol = 'R_100'
    bot._auth_verified = True
    bot._shadow_monitor, bot._shadow_model_id = monitor, 'model-a'
    bot._shadow_symbol, bot._shadow_timeframe = bot.symbol, 300
    callback = Mock()
    bot.on_shadow_rejected = callback
    monkeypatch.setattr(executor.ai_predictor, 'model_id', lambda: 'model-a')
    monkeypatch.setattr(bot, '_handle_training_tick', Mock())
    ws = MagicMock()
    for epoch in (ENTRY + 1800, ENTRY + 1801):
        bot._handle_tick(ws, dict(symbol=bot.symbol, epoch=epoch, quote=99))
    callback.assert_called_once_with('R_100')
    assert not bot._shadow_gate_ready('model-a')
    ws.send.assert_not_called()
    status = json.loads(executor._BOT_STATUS_JSON.read_text())['shadow']
    assert status['reason'] == 'rejected' and status['rejection_reason'] == 'minimum_wins_unreachable'
    assert status['maximum_possible_wins'] == 6


def test_rejection_does_not_close_socket_or_change_saved_financial_contract_until_reconciliation(tmp_path, monkeypatch):
    import executor
    from risk_manager import RiskManager
    monkeypatch.setattr(config, 'USE_AI_MODEL', True)
    for key in ('USE_WEIGHTED_SIGNAL', 'PRICE_ACTION_FILTER', 'CANDLE_PATTERN_FILTER'):
        monkeypatch.setattr(config, key, False)
    monitor = tracker(tmp_path / 'shadow.json')
    settle(monitor, 0, correct=False)
    observe(monitor, ENTRY + 900)
    bot = executor.DerivBot(RiskManager(1000), demo=True)
    bot.symbol = 'R_100'
    bot._auth_verified = True
    bot._in_trade, bot._phase = True, 'reconciling'
    bot._shadow_monitor, bot._shadow_model_id = monitor, 'model-a'
    bot._shadow_symbol, bot._shadow_timeframe = bot.symbol, 300
    executor._STATE_JSON.write_text(json.dumps(dict(symbol='R_100', active_contract=dict(phase='reconciling'))))
    original = executor._STATE_JSON.read_bytes()
    event = threading.Event()
    bot.on_shadow_rejected = lambda symbol: event.set()
    bot.market_switch_requested = event.is_set
    monkeypatch.setattr(executor.ai_predictor, 'model_id', lambda: 'model-a')
    monkeypatch.setattr(bot, '_handle_training_tick', Mock())
    ws = MagicMock()
    for epoch in (ENTRY + 1800, ENTRY + 1801):
        bot._on_message(ws, json.dumps(dict(msg_type='tick', tick=dict(symbol='R_100', epoch=epoch, quote=99))))
    assert event.is_set() and bot._in_trade
    assert not bot._market_switch_ready()
    ws.close.assert_not_called()
    ws.send.assert_not_called()
    assert executor._STATE_JSON.read_bytes() == original
    # Simula reconciliação concluída pelo fluxo financeiro já existente.
    bot._in_trade = False
    bot._on_message(ws, '{}')
    ws.close.assert_called_once()


def test_capture_after_rejection_cannot_overwrite_terminal_status_with_waiting(tmp_path, monkeypatch):
    import executor
    from risk_manager import RiskManager
    monkeypatch.setattr(config, 'USE_AI_MODEL', True)
    for key in ('USE_WEIGHTED_SIGNAL', 'PRICE_ACTION_FILTER', 'CANDLE_PATTERN_FILTER'):
        monkeypatch.setattr(config, key, False)
    monitor = tracker(tmp_path / 'shadow.json')
    settle(monitor, 0, correct=False)
    settle(monitor, 1, correct=False)
    bot = executor.DerivBot(RiskManager(1000), demo=True)
    bot.symbol = 'R_100'
    bot._shadow_monitor, bot._shadow_model_id = monitor, 'model-a'
    bot._shadow_symbol, bot._shadow_timeframe = bot.symbol, 300
    bot._last_tick_epoch = ENTRY + 1800
    monkeypatch.setattr(executor.ai_predictor, 'status', lambda: dict(ready=True))
    monkeypatch.setattr(executor.ai_predictor, 'predict', lambda *args, **kwargs: ('BUY', .8))
    monkeypatch.setattr(executor.ai_predictor, 'prediction_model_id', lambda: 'model-a')
    monkeypatch.setattr(bot, '_shadow_payout', lambda *args: .73)
    before = monitor.canary.path.read_bytes()
    bot._capture_shadow_prediction(101, ENTRY + 1800, candles=[])
    status = json.loads(executor._BOT_STATUS_JSON.read_text())['shadow']
    assert status['reason'] == 'rejected' and status['result_count'] == 2
    assert monitor.canary.path.read_bytes() == before


def test_canary_rejection_dominates_simultaneous_legacy_approval_without_misleading_counts(tmp_path):
    path = tmp_path / 'shadow.json'
    legacy = ShadowMonitor(path, **IDENTITY)
    for index in range(28):
        entry = ENTRY - (28 - index) * 900
        assert observe(legacy, entry) == 'pending'
        legacy.settle_tick(epoch=entry + 900, price=101)
    monitor = tracker(path)
    settle(monitor, 0, correct=False)
    assert settle(monitor, 1, correct=False) == 'rejected'
    assert monitor.legacy.approved
    assert monitor.rejected and not monitor.approved
    assert monitor.result_count == 2 and monitor.minimum_results == 8
    assert monitor.diagnostics()['approval_path'] == 'canary_8'
    assert monitor.diagnostics()['legacy_result_count'] == 30
    resumed = tracker(path)
    assert resumed.rejected and not resumed.approved
