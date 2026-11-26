"""Diagnóstico da IA: progresso corrompido e dados inválidos não derrubam nem repetem trabalho."""
from unittest.mock import Mock

import pytest

import pipeline


@pytest.fixture
def diagnose_env(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, 'TICKS_CSV', str(tmp_path / 'ticks.csv'))
    monkeypatch.setattr(pipeline, '_BOT_STATUS_JSON', tmp_path / 'bot_status.json')
    monkeypatch.setattr(pipeline, '_training_readiness_cache', {})
    clock = [1000.]
    monkeypatch.setattr(pipeline.time, 'monotonic', lambda: clock[0])
    return clock


def test_corrupt_progress_does_not_raise_from_cached_diagnostic(diagnose_env, monkeypatch):
    prepare = Mock(side_effect=ValueError('origem malformada'))
    monkeypatch.setattr(pipeline, '_prepare_model_snapshot', prepare)
    monkeypatch.setattr(pipeline, '_training_progress',
                        Mock(side_effect=ValueError('progresso de treino inválido')))

    first = pipeline._diagnose_training_symbol('R_100')
    # Antes da correção, a segunda chamada (cache válido) levantava ValueError
    # fora do try e derrubava _wait_for_ticks ou a thread de retreino.
    second = pipeline._diagnose_training_symbol('R_100')
    assert first['stage'] == second['stage'] == 'invalid_data'
    assert prepare.call_count == 1


def test_invalid_data_is_served_from_cache_within_60s(diagnose_env, monkeypatch):
    prepare = Mock(side_effect=ValueError('cache incompatível'))
    monkeypatch.setattr(pipeline, '_prepare_model_snapshot', prepare)
    progress = Mock()
    progress.observe.return_value = dict(revision=3, new_samples=0, attempted=True,
                                         previous_samples=0, metrics={})
    monkeypatch.setattr(pipeline, '_training_progress', lambda: progress)

    assert pipeline._diagnose_training_symbol('R_100')['stage'] == 'invalid_data'
    assert pipeline._diagnose_training_symbol('R_100')['stage'] == 'invalid_data'
    assert prepare.call_count == 1  # sem nova preparação nem recuperação pública
    diagnose_env[0] += 61
    pipeline._diagnose_training_symbol('R_100')
    assert prepare.call_count == 2


def test_new_progress_revision_invalidates_cached_invalid_data(diagnose_env, monkeypatch):
    prepare = Mock(side_effect=ValueError('cache incompatível'))
    monkeypatch.setattr(pipeline, '_prepare_model_snapshot', prepare)
    progress = Mock()
    progress.observe.return_value = dict(revision=1)
    monkeypatch.setattr(pipeline, '_training_progress', lambda: progress)
    pipeline._diagnose_training_symbol('R_100')
    progress.observe.return_value = dict(revision=2)  # treino concluído em outra thread
    pipeline._diagnose_training_symbol('R_100')
    assert prepare.call_count == 2
