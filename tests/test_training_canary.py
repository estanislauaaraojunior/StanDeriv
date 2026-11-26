"""Canary da IA com estado temporário, sem conta, modelo real ou rede."""
import json
from datetime import datetime, timezone

import pytest

import config
from shadow_monitor import ShadowMonitor
from training_canary import create_monitor, saved_canary, ParallelShadow

ENTRY = 1_800_000_000
IDENTITY = dict(symbol='R_100', model_id='model-a', timeframe_sec=300, duration_sec=900)


def evidence(**changes):
    metadata = dict(symbol='R_100', model_id='model-a', timeframe_sec=300,
                    duration=15, duration_unit='m', approved=True,
                    dataset_sha256='a' * 64,
                    trained_at=datetime.fromtimestamp(ENTRY - 100, timezone.utc).isoformat(),
                    metrics=dict(eligible=True, holdout_used_for_selection=False,
                                 trades=60, wins=35, fit_last_expiry=ENTRY - 10000,
                                 calibration_first_feature=ENTRY - 9000,
                                 calibration_last_expiry=ENTRY - 8000,
                                 test_first_feature=ENTRY - 7000,
                                 evaluation_start=ENTRY - 6000, evaluation_end=ENTRY - 200))
    metadata.update(changes)
    return metadata


def tracker(path, metadata=None, **identity):
    return create_monitor(path, evidence=evidence() if metadata is None else metadata,
                          now=lambda: ENTRY, **(IDENTITY | identity))


def observe(monitor, entry=ENTRY, **changes):
    return monitor.observe_prediction(**(dict(entry_epoch=entry, entry_price=100,
        direction=1, confidence=.8, net_payout=.73) | changes))


def settle(monitor, index, correct=True):
    entry = ENTRY + index * 900
    assert observe(monitor, entry) == 'pending'
    return monitor.settle_tick(epoch=entry + 900, price=101 if correct else 99)


@pytest.mark.parametrize('wins,expected', [(8, 'approved'), (7, 'approved'), (6, 'rejected'), (0, 'rejected')])
def test_fixed_eight_results_never_approve_early_or_roll_until_pass(tmp_path, wins, expected):
    path = tmp_path / 'shadow.json'
    monitor = tracker(path)
    assert isinstance(monitor, ParallelShadow)
    assert monitor.minimum_results == 8 and monitor.minimum_accuracy == 7 / 8
    for index in range(8):
        result = settle(monitor, index, correct=index < wins)
        if monitor.rejected:
            break
        if index < 7:
            assert result == 'settled' and not monitor.approved and not monitor.rejected
    assert result == expected
    assert monitor.approved == (expected == 'approved')
    assert monitor.rejected == (expected == 'rejected')
    expected_count = 2 if wins == 0 else 8
    assert monitor.result_count == expected_count and monitor.legacy.result_count == expected_count
    if monitor.approved:
        assert monitor.canary.unit_profit > 0
    resumed = tracker(path)
    assert resumed.approved == monitor.approved and resumed.rejected == monitor.rejected
    assert observe(resumed, ENTRY + 9000) == 'already_pending_or_approved'
    assert resumed.result_count == expected_count
    assert saved_canary(path, **IDENTITY)['approved_at_epoch'] == monitor.state['approved_at_epoch']


def test_legacy_progress_and_pending_continue_without_becoming_new_forward_results(tmp_path):
    path = tmp_path / 'shadow.json'
    legacy = ShadowMonitor(path, **IDENTITY)
    assert observe(legacy, ENTRY - 2700) == 'pending'
    legacy.settle_tick(epoch=ENTRY - 1800, price=101)
    assert observe(legacy, ENTRY - 1800) == 'pending'
    legacy.settle_tick(epoch=ENTRY - 900, price=99)
    assert observe(legacy, ENTRY - 900) == 'pending'
    before = path.read_bytes()
    monitor = tracker(path)
    assert path.read_bytes() == before
    assert monitor.result_count == 0 and monitor.legacy.result_count == 2
    assert monitor.settle_tick(epoch=ENTRY, price=101) == 'settled'
    assert monitor.legacy.result_count == 3 and monitor.result_count == 0
    assert observe(monitor, ENTRY - 1) == 'before_policy_freeze'
    assert settle(monitor, 0) == 'settled'
    assert monitor.result_count == 1 and monitor.legacy.result_count == 4
    resumed = tracker(path)
    assert resumed.result_count == 1 and resumed.legacy.result_count == 4
    assert resumed.state['frozen_at_epoch'] == ENTRY


@pytest.mark.parametrize('invalid', ['unapproved', 'too_few', 'selection', 'embargo', 'hash', 'identity'])
def test_unverified_holdout_falls_back_to_original_thirty(tmp_path, invalid):
    metadata = evidence()
    if invalid == 'unapproved': metadata['approved'] = False
    if invalid == 'too_few': metadata['metrics']['trades'] = 29
    if invalid == 'selection': metadata['metrics']['holdout_used_for_selection'] = True
    if invalid == 'embargo': metadata['metrics']['calibration_last_expiry'] = ENTRY
    if invalid == 'hash': metadata['dataset_sha256'] = 'missing'
    if invalid == 'identity': metadata['model_id'] = 'different'
    monitor = tracker(tmp_path / 'shadow.json', metadata)
    assert not isinstance(monitor, ParallelShadow)
    assert monitor.minimum_results == config.SHADOW_MIN_RESULTS == 30
    assert monitor.policy_reason == 'historical_holdout_not_verified'
    assert not (tmp_path / 'shadow.json.canary').exists()


def test_mixed_timeframe_keeps_original_policy(tmp_path):
    monitor = tracker(tmp_path / 'shadow.json', timeframe_sec=900)
    assert not isinstance(monitor, ParallelShadow)
    assert monitor.minimum_results == 30


def test_payout_gap_tie_and_nonoverlap_still_apply(tmp_path):
    monitor = tracker(tmp_path / 'shadow.json')
    assert observe(monitor, net_payout=.69) == 'payout_below_floor'
    assert observe(monitor, net_payout=.70) == 'pending'
    assert observe(monitor, ENTRY + 300) == 'already_pending_or_approved'
    assert monitor.settle_tick(epoch=ENTRY + 911, price=101) == 'expired_with_gap'
    assert monitor.result_count == 0
    assert observe(monitor, ENTRY + 1200) == 'pending'
    assert monitor.settle_tick(epoch=ENTRY + 2100, price=100) == 'tie'
    assert monitor.result_count == 0


def test_exact_identity_resumes_and_other_bundle_cannot_reuse_holdout(tmp_path):
    path = tmp_path / 'shadow.json'
    first = tracker(path)
    settle(first, 0)
    assert observe(first, ENTRY + 900) == 'pending'
    different = tracker(path, evidence(model_id='other'), model_id='other')
    assert not isinstance(different, ParallelShadow)
    assert different.policy_reason == 'historical_holdout_already_consumed'
    original = tracker(path)
    assert original.result_count == 1 and original.state['pending']['entry_epoch'] == ENTRY + 900
    assert original.legacy.state['pending']['entry_epoch'] == ENTRY + 900


def test_new_unconsumed_historical_period_can_admit_another_bundle(tmp_path):
    path = tmp_path / 'shadow.json'
    earlier = evidence()
    earlier['metrics'].update(fit_last_expiry=ENTRY - 20000,
        calibration_first_feature=ENTRY - 19000, calibration_last_expiry=ENTRY - 18000,
        test_first_feature=ENTRY - 17000, evaluation_start=ENTRY - 16000,
        evaluation_end=ENTRY - 9000)
    assert isinstance(tracker(path, earlier), ParallelShadow)
    assert isinstance(tracker(path, evidence(model_id='other'), model_id='other'), ParallelShadow)


def test_damaged_admission_from_another_bundle_does_not_crash_or_authorize_shortcut(tmp_path):
    path = tmp_path / 'shadow.json'
    first = tracker(path)
    saved = json.loads(first.canary.path.read_text())
    saved['evidence'] = ['invalid proof']
    first.canary.path.write_text(json.dumps(saved))
    other = tracker(path, evidence(model_id='other'), model_id='other')
    assert other.minimum_results == 30
    assert other.policy_reason == 'historical_admission_ledger_invalid'


@pytest.mark.parametrize('corruption', ['approval', 'results', 'json', 'evidence'])
def test_corrupt_canary_cannot_authorize_or_restart_same_attempt(tmp_path, corruption):
    path = tmp_path / 'shadow.json'
    monitor = tracker(path)
    settle(monitor, 0)
    state_path = monitor.canary.path
    state = json.loads(state_path.read_text())
    if corruption == 'approval': state['approved_at_epoch'] = ENTRY
    if corruption == 'results': state['results'][0]['correct'] = False
    if corruption == 'evidence': state['evidence']['historical_trades'] = 1
    raw = 'invalid json' if corruption == 'json' else json.dumps(state)
    state_path.write_text(raw)
    resumed = tracker(path)
    assert resumed.rejected and not resumed.approved
    assert state_path.read_text() == raw
    # A leitura do pipeline também falha fechada e não reescreve os arquivos.
    state_read = saved_canary(path, **IDENTITY)
    assert state_read.get('rejected_at_epoch') is not None


def test_legacy_final_rejection_is_not_erased_by_short_policy(tmp_path):
    path = tmp_path / 'shadow.json'
    legacy = ShadowMonitor(path, **IDENTITY)
    for index in range(30):
        settle(legacy, index, correct=False)
    monitor = tracker(path)
    assert monitor.rejected and monitor.minimum_results == 30
    assert not (tmp_path / 'shadow.json.canary').exists()


def test_original_progress_can_complete_first_without_waiting_for_new_eight(tmp_path):
    path = tmp_path / 'shadow.json'
    legacy = ShadowMonitor(path, **IDENTITY)
    for index in range(29):
        entry = ENTRY - (29 - index) * 900
        assert observe(legacy, entry) == 'pending'
        legacy.settle_tick(epoch=entry + 900, price=101)
    monitor = tracker(path)
    assert settle(monitor, 0) == 'approved'
    assert monitor.approved
    assert monitor.minimum_results == monitor.result_count == 30
    assert monitor.diagnostics()['canary_result_count'] == 1
    assert monitor.diagnostics()['approval_path'] == 'legacy_30'
    assert tracker(path).approved


def test_pipeline_recognizes_short_approval_and_preserves_rejection(tmp_path, monkeypatch):
    import pipeline
    monkeypatch.setattr(pipeline, 'model_timeframe_sec', lambda: 300)
    path = tmp_path / 'shadow.json'
    monkeypatch.setattr(pipeline, '_SHADOW_STATE_JSON', path)
    metadata = evidence()
    monitor = tracker(path)
    for index in range(2): settle(monitor, index, correct=False)
    assert not pipeline._training_model_resumable(metadata)
    assert not pipeline._market_switch.is_set()


def test_pipeline_checks_completed_canary_against_exact_active_bundle(tmp_path, monkeypatch):
    import pipeline
    from model_contract import fingerprint
    path = tmp_path / 'shadow.json'
    model_path = tmp_path / 'model.pkl'
    model_path.write_bytes(b'offline fixture, no pickle loading')
    model_id = fingerprint(model_path)
    metadata = evidence(model_id=model_id, artifact_sha256=model_id)
    monitor = tracker(path, metadata, model_id=model_id)
    for index in range(8): settle(monitor, index)
    model_path.with_name('model.pkl.metadata.json').write_text(json.dumps(metadata))
    monkeypatch.setattr(pipeline, '_SHADOW_STATE_JSON', path)
    monkeypatch.setattr(pipeline, 'AI_MODEL_PATH', str(model_path))
    monkeypatch.setattr(pipeline, '_runtime_symbol', lambda: 'R_100')
    monkeypatch.setattr(pipeline, 'model_timeframe_sec', lambda: 300)
    assert pipeline._active_model_shadow_approved()
    model_path.write_bytes(b'changed artifact')
    assert not pipeline._active_model_shadow_approved()


def test_executor_admits_short_policy_only_with_loaded_model_evidence(tmp_path, monkeypatch):
    import executor
    from risk_manager import RiskManager
    monkeypatch.setattr(config, 'USE_AI_MODEL', True)
    for key in ('USE_WEIGHTED_SIGNAL', 'PRICE_ACTION_FILTER', 'CANDLE_PATTERN_FILTER'):
        monkeypatch.setattr(config, key, False)
    monkeypatch.setattr(executor, '_SHADOW_STATE_JSON', tmp_path / 'shadow.json')
    monkeypatch.setattr(executor.ai_predictor, 'shadow_evidence', lambda: evidence(
        trained_at=datetime.fromtimestamp(100, timezone.utc).isoformat(),
        metrics=dict(evidence()['metrics'], evaluation_end=90, evaluation_start=80,
                     test_first_feature=70, calibration_last_expiry=60,
                     calibration_first_feature=50, fit_last_expiry=40)))
    bot = executor.DerivBot(RiskManager(1000))
    bot.symbol = IDENTITY['symbol']
    monitor = bot._get_shadow_monitor('model-a')
    assert monitor.minimum_results == 8
    assert not bot._shadow_gate_ready('model-a')
    bot._publish_shadow_status(model_id='model-a', reason='test')
    status = json.loads(executor._BOT_STATUS_JSON.read_text())['shadow']
    assert status['minimum_results'] == 8 and status['legacy_result_count'] == 0
    assert status['historical_trades'] == 60


def test_executor_buy_path_requires_full_canary_and_remains_demo_only(tmp_path, monkeypatch):
    from unittest.mock import MagicMock
    import executor
    from risk_manager import RiskManager
    monkeypatch.setattr(config, 'USE_AI_MODEL', True)
    for key in ('USE_WEIGHTED_SIGNAL', 'PRICE_ACTION_FILTER', 'CANDLE_PATTERN_FILTER'):
        monkeypatch.setattr(config, key, False)
    path = tmp_path / 'shadow.json'
    monitor = tracker(path)
    risk = RiskManager(1000)
    risk.bind_account('offline-account', 1000)
    bot = executor.DerivBot(risk)
    bot.symbol = 'R_100'
    bot.demo = True
    bot._auth_verified = True
    bot._shadow_monitor = monitor
    bot._shadow_model_id = 'model-a'
    bot._shadow_symbol = 'R_100'
    bot._shadow_timeframe = 300
    bot._last_tick_epoch = ENTRY + 9000
    monkeypatch.setattr(executor.time, 'time', lambda: ENTRY + 9000)
    monkeypatch.setattr(bot, '_market_switch_pending', lambda: False)
    monkeypatch.setattr(bot, '_risk_allows_trade', lambda: True)
    ws = MagicMock()
    indicators = dict(model_id='model-a', win_probability=.8,
                      enabled_decision_modules=['trained_model'])
    assert not bot._send_proposal(ws, 'BUY', indicators)
    for index in range(7): settle(monitor, index)
    assert not bot._send_proposal(ws, 'BUY', indicators)
    ws.send.assert_not_called()
    settle(monitor, 7)
    bot.demo = False
    assert not bot._send_proposal(ws, 'BUY', indicators)
    ws.send.assert_not_called()
    bot.demo = True
    assert bot._send_proposal(ws, 'BUY', indicators)
    proposal = json.loads(ws.send.call_args.args[0])
    assert proposal['proposal'] == 1 and 'buy' not in proposal
    assert proposal['duration'] == 15 and proposal['duration_unit'] == 'm'
    assert bot._phase == 'proposal'


def test_unverifiable_proof_does_not_reject_valid_saved_canary(tmp_path):
    """Relógio atrás de trained_at ou metadados incompletos não viram reprovação."""
    path = tmp_path / 'shadow.json'
    first = tracker(path)
    assert settle(first, 0) == 'settled'
    canary_file = next((tmp_path / 'shadow.json.canary').glob('*.json'))
    before = canary_file.read_bytes()

    # trained_at no futuro em relação ao relógio atual: prova não verificável.
    skewed = create_monitor(path, evidence=evidence(), now=lambda: ENTRY - 1_000, **IDENTITY)
    assert not isinstance(skewed, ParallelShadow)
    assert skewed.policy_reason == 'historical_holdout_not_verified'
    assert not skewed.rejected
    assert observe(skewed, ENTRY + 900) == 'pending'
    assert skewed.settle_tick(epoch=ENTRY + 1800, price=101) != 'rejected'
    assert canary_file.read_bytes() == before

    # Com a prova verificável de novo, a mesma tentativa é retomada.
    resumed = tracker(path)
    assert isinstance(resumed, ParallelShadow) and not resumed.rejected
    assert len(resumed.canary.state['results']) == 1


def test_unverifiable_proof_keeps_persisted_rejection_terminal(tmp_path):
    path = tmp_path / 'shadow.json'
    monitor = tracker(path)
    for index in range(2):
        settle(monitor, index, correct=False)
    assert monitor.rejected
    skewed = create_monitor(path, evidence=evidence(), now=lambda: ENTRY - 1_000, **IDENTITY)
    assert skewed.rejected
