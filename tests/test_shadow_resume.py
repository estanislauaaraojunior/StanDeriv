"""Retomada por identidade sem comprar contratos ou tocar o estado operacional."""
import json

import pytest

from shadow_monitor import ShadowMonitor


IDENTITY = dict(symbol='R_100', model_id='model-a', timeframe_sec=900, duration_sec=900)
ENTRY = 1_800_000_000


def monitor(path, **identity):
    return ShadowMonitor(path, **(IDENTITY | identity))


def observe(tracker, entry=ENTRY):
    assert tracker.observe_prediction(
        entry_epoch=entry, entry_price=100, direction=1, confidence=.8, net_payout=.75,
    ) == 'pending'


def test_legacy_active_file_is_preserved_completely_before_switch(tmp_path):
    path = tmp_path / 'shadow.json'
    first = monitor(path)
    observe(first)
    assert first.settle_tick(epoch=ENTRY + 900, price=101) == 'settled'
    observe(first, ENTRY + 900)
    legacy = json.loads(path.read_text())
    legacy['extra_diagnostic'] = {'source': 'legacy'}
    path.write_text(json.dumps(legacy))
    assert not path.with_name(path.name + '.states').exists()

    other = monitor(path, model_id='model-b')
    snapshots = list(path.with_name(path.name + '.states').glob('*.json'))
    assert len(snapshots) == 1
    assert json.loads(snapshots[0].read_text()) == legacy
    assert other.result_count == 0
    assert other.state['pending'] is None

    resumed = monitor(path)
    for key in ('schema_version', *IDENTITY, 'results', 'pending',
                'approved_at_epoch', 'rejected_at_epoch', 'extra_diagnostic'):
        assert resumed.state[key] == legacy[key]
    assert json.loads(path.read_text()) == resumed.state
    assert resumed.settle_tick(epoch=ENTRY + 1800, price=101) == 'settled'
    assert resumed.result_count == 2


@pytest.mark.parametrize('outcome,price', [('approved', 101), ('rejected', 99)])
def test_completed_shadow_resumes_without_revalidating_same_bundle(tmp_path, outcome, price):
    path = tmp_path / 'shadow.json'
    first = monitor(path)
    for index in range(30):
        entry = ENTRY + index * 900
        observe(first, entry)
        result = first.settle_tick(epoch=entry + 900, price=price)
    assert result == outcome
    before = json.loads(path.read_text())

    changed = monitor(path, model_id='model-b')
    assert not changed.approved and not changed.rejected
    resumed = monitor(path)
    assert resumed.approved == (outcome == 'approved')
    assert resumed.rejected == (outcome == 'rejected')
    assert resumed.result_count == 30
    assert resumed.state['approved_at_epoch'] == before['approved_at_epoch']
    assert resumed.state['rejected_at_epoch'] == before['rejected_at_epoch']
    assert resumed.observe_prediction(
        entry_epoch=ENTRY + 40 * 900, entry_price=100,
        direction=1, confidence=.8, net_payout=.75,
    ) == 'already_pending_or_approved'


@pytest.mark.parametrize('changed_identity', [
    {'symbol': 'R_50'},
    {'model_id': 'model-b'},
    {'timeframe_sec': 60},
    {'duration_sec': 60},
])
def test_other_identity_starts_fresh_and_original_resumes(tmp_path, changed_identity):
    path = tmp_path / 'shadow.json'
    first = monitor(path)
    observe(first)
    pending = dict(first.state['pending'])

    other = monitor(path, **changed_identity)
    assert other.result_count == 0
    assert other.state['pending'] is None
    assert not other.approved and not other.rejected
    observe(other, ENTRY + 9000)

    resumed = monitor(path)
    assert resumed.state['pending'] == pending
    assert resumed.settle_tick(epoch=ENTRY + 900, price=101) == 'settled'
    assert resumed.state['results'][0]['entry_epoch'] == ENTRY


def test_returned_pending_prediction_keeps_normal_gap_expiry(tmp_path):
    path = tmp_path / 'shadow.json'
    first = monitor(path)
    observe(first)
    monitor(path, model_id='model-b')

    resumed = monitor(path)
    assert resumed.settle_tick(epoch=ENTRY + 911, price=101) == 'expired_with_gap'
    assert resumed.result_count == 0
    assert resumed.state['pending'] is None
    assert monitor(path).state['pending'] is None


def test_snapshots_preserve_identity_beyond_display_archive_limit(tmp_path):
    path = tmp_path / 'shadow.json'
    first = monitor(path)
    observe(first)
    assert first.settle_tick(epoch=ENTRY + 900, price=101) == 'settled'

    for index in range(25):
        monitor(path, model_id=f'other-{index}')

    resumed = monitor(path)
    assert resumed.result_count == 1
    assert len(resumed.state['archives']) == 20


@pytest.mark.parametrize('mismatch', [
    {'schema_version': 2},
    {'symbol': 'WRONG'},
    {'model_id': 'wrong-model'},
    {'timeframe_sec': 60},
    {'duration_sec': 60},
])
def test_snapshot_payload_must_match_schema_and_complete_identity(tmp_path, mismatch):
    path = tmp_path / 'shadow.json'
    first = monitor(path)
    observe(first)
    monitor(path, model_id='model-b')
    snapshot = next(path.with_name(path.name + '.states').glob('*.json'))
    invalid = json.loads(snapshot.read_text()) | mismatch
    invalid['approved_at_epoch'] = ENTRY
    snapshot.write_text(json.dumps(invalid))

    resumed = monitor(path)
    assert not resumed.approved and not resumed.rejected
    assert resumed.state['pending'] is None
    assert resumed.result_count == 0


def test_unknown_schema_is_preserved_but_does_not_authorize_current_schema(tmp_path):
    path = tmp_path / 'shadow.json'
    first = monitor(path)
    observe(first)
    newer = json.loads(path.read_text())
    newer['schema_version'] = 2
    newer['approved_at_epoch'] = ENTRY
    path.write_text(json.dumps(newer))

    current = monitor(path)
    assert not current.approved
    assert current.state['pending'] is None
    snapshots = list(path.with_name(path.name + '.states').glob('*.json'))
    assert len(snapshots) == 1
    assert json.loads(snapshots[0].read_text()) == newer


def test_legacy_partial_archives_are_not_used_to_restore_approval(tmp_path):
    path = tmp_path / 'shadow.json'
    active = dict(schema_version=1, **(IDENTITY | {'model_id': 'model-b'}),
                  pending=None, results=[], approved_at_epoch=None,
                  rejected_at_epoch=None, archives=[dict(IDENTITY, approved_at_epoch=ENTRY,
                                                       results=[{'correct': True}])])
    path.write_text(json.dumps(active))

    resumed = monitor(path)
    assert not resumed.approved and not resumed.rejected
    assert resumed.result_count == 0
    assert resumed.state['pending'] is None


def test_saved_state_lookup_does_not_replace_active_identity_or_create_files(tmp_path):
    absent = tmp_path / 'absent' / 'shadow.json'
    assert ShadowMonitor.saved_state(absent, **IDENTITY) is None
    assert not absent.parent.exists()
    path = tmp_path / 'shadow.json'
    first = monitor(path)
    observe(first)
    monitor(path, model_id='model-b')
    before = path.read_bytes()
    saved = ShadowMonitor.saved_state(path, **IDENTITY)
    assert saved['pending'] is not None
    assert path.read_bytes() == before


def test_archived_shadow_rejection_prevents_resuming_bundle_in_pipeline(tmp_path, monkeypatch):
    import pipeline
    path = tmp_path / 'shadow.json'
    first = monitor(path)
    for index in range(30):
        entry = ENTRY + index * 900
        observe(first, entry)
        first.settle_tick(epoch=entry + 900, price=99)
    assert first.rejected
    monitor(path, model_id='model-b')
    before = path.read_bytes()
    monkeypatch.setattr(pipeline, '_SHADOW_STATE_JSON', path)
    assert not pipeline._training_model_resumable(dict(symbol='R_100', model_id='model-a'))
    assert pipeline._training_model_resumable(dict(symbol='R_100', model_id='model-b'))
    assert path.read_bytes() == before
