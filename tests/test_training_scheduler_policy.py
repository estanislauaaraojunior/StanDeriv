"""Progresso persistente da IA e prontidão temporal sem executar estimadores."""
import json

import numpy as np
import pandas as pd
import pytest

import training_policy
from training_policy import TrainingProgress, readiness_capacity, retry_plan


def labels(first, count):
    return [(i * 900, i * 900, (i + 1) * 900, i % 2) for i in range(first, first + count)]


def test_rolling_window_counts_new_labels_when_total_stays_constant(tmp_path):
    progress = TrainingProgress(tmp_path / 'progress.json', identity={'policy': 1})
    progress.mark_attempt('FIRST', labels(0, 800), {'eligible': False, 'rejection_reasons': ['taxa de acerto baixa']})
    current = progress.observe('FIRST', labels(120, 800))
    assert current['previous_samples'] == 800
    assert current['new_samples'] == 120
    plan = retry_plan(800, current['previous_samples'], current['metrics'], new_samples=current['new_samples'])
    assert plan['remaining_samples'] == 0
    assert retry_plan(800, 800, current['metrics'])['remaining_samples'] == 120


def test_symbols_duplicates_and_restart_preserve_their_own_attempts(tmp_path):
    path = tmp_path / 'progress.json'
    progress = TrainingProgress(path, identity='v1')
    assert progress.observe('FIRST', labels(0, 5))['attempted'] is False
    progress.mark_attempt('FIRST', labels(0, 5), {'eligible': False, 'rejection_reasons': ['amostra insuficiente']})
    progress.mark_attempt('SECOND', labels(0, 3), {'eligible': False, 'rejection_reasons': ['desempenho insuficiente']})
    restarted = TrainingProgress(path, identity='v1')
    assert restarted.observe('FIRST', labels(0, 7) * 2)['new_samples'] == 2
    second = restarted.observe('SECOND', labels(0, 7))
    assert second['new_samples'] == 4
    assert second['previous_samples'] == 3
    assert second['metrics']['rejection_reasons'] == ['desempenho insuficiente']
    assert restarted.observe('THIRD', labels(0, 7))['attempted'] is False


def test_attempt_consumes_only_snapshot_used_even_if_more_labels_were_observed(tmp_path):
    path = tmp_path / 'progress.json'
    progress = TrainingProgress(path)
    trained = labels(0, 600)
    progress.mark_attempt('FIRST', trained, {'eligible': False})
    assert progress.observe('FIRST', labels(0, 720))['new_samples'] == 120
    progress.mark_attempt('FIRST', trained, {'eligible': False, 'stage': 'final'})
    restarted = TrainingProgress(path)
    assert restarted.observe('FIRST', labels(0, 720))['new_samples'] == 120
    restarted.mark_attempt('FIRST', labels(120, 600), {'eligible': False})
    assert restarted.observe('FIRST', labels(0, 720))['new_samples'] == 0


def test_retention_watermark_prevents_recounting_expired_historical_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(training_policy, '_MAX_RETAINED_LABELS', 4)
    path = tmp_path / 'progress.json'
    progress = TrainingProgress(path)
    progress.mark_attempt('FIRST', labels(0, 9), {'eligible': False})
    state = json.loads(path.read_text())['symbols']['FIRST']
    assert len(state['consumed']) <= 4
    restarted = TrainingProgress(path)
    assert restarted.observe('FIRST', labels(0, 10))['new_samples'] == 1


def test_policy_identity_requires_a_new_attempt_and_does_not_rewrite_on_observation(tmp_path):
    path = tmp_path / 'progress.json'
    TrainingProgress(path, identity={'schema': 1}).mark_attempt('FIRST', labels(0, 5), {'eligible': False})
    before = path.read_bytes()
    changed = TrainingProgress(path, identity={'schema': 2})
    assert changed.observe('FIRST', labels(0, 5))['attempted'] is False
    assert path.read_bytes() == before


def test_corrupt_progress_fails_without_silently_losing_attempts(tmp_path):
    path = tmp_path / 'progress.json'
    path.write_text('{')
    with pytest.raises(ValueError, match='progresso de treino inválido'):
        TrainingProgress(path)
    assert path.read_text() == '{'


def test_failed_atomic_write_does_not_consume_new_labels(tmp_path, monkeypatch):
    progress = TrainingProgress(tmp_path / 'progress.json')
    progress.mark_attempt('FIRST', labels(0, 5), {'eligible': False})

    def fail(*args):
        raise OSError('falha simulada')

    monkeypatch.setattr(training_policy, 'write_json_atomic', fail)
    with pytest.raises(OSError):
        progress.mark_attempt('FIRST', labels(0, 7), {'eligible': False})
    assert progress.observe('FIRST', labels(0, 7))['new_samples'] == 2


def test_explicit_progress_preserves_smaller_sample_retry(tmp_path):
    plan = retry_plan(600, 600, {'rejection_reasons': ['amostra final insuficiente']}, new_samples=31)
    assert plan['retry_batch'] == 30
    assert plan['remaining_samples'] == 0
    assert plan['new_samples'] == 31


def temporal_frame(count, staggered=False):
    signal = 1_700_000_000 + np.arange(count) * 900
    offset = (np.arange(count) % 2) * 2 if staggered else np.zeros(count, dtype=int)
    expiry_offset = np.maximum(offset, np.roll(offset, -1))
    return pd.DataFrame(dict(signal_epoch=signal, feature_start_epoch=signal - 50 * 900,
                             entry_epoch=signal + offset, expiry_epoch=signal + 900 + expiry_offset,
                             target=np.arange(count) % 2))


def test_structural_floor_does_not_hide_insufficient_independent_capacity():
    result = readiness_capacity(temporal_frame(579, staggered=True))
    assert result['stage'] == 'evaluation_capacity'
    assert result['evaluation_capacity']['validation'] == 16
    assert readiness_capacity(temporal_frame(800, staggered=True))['stage'] == 'ready'


def test_preflight_preserves_sample_and_class_requirements():
    assert readiness_capacity(temporal_frame(578))['stage'] == 'insufficient_samples'
    assert readiness_capacity(temporal_frame(579))['stage'] == 'ready'
    frame = temporal_frame(800)
    frame['target'] = 1
    assert readiness_capacity(frame)['stage'] == 'insufficient_samples'
