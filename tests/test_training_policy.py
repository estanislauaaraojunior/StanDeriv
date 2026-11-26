import pandas as pd
from training_policy import evaluation_capacity, retry_plan


def test_capacity_rejects_overlapping_opportunities():
    frame = pd.DataFrame({'entry_epoch': [0, 1, 2, 11], 'expiry_epoch': [10, 11, 12, 21]})
    assert evaluation_capacity(frame) == 2


def test_sample_only_retry_is_smaller_but_quality_keeps_renewal():
    plan = retry_plan(603, 600, {'rejection_reasons': ['amostra final insuficiente']})
    assert plan['next_training_at'] == 630
    assert plan['remaining_samples'] == 27
    plan = retry_plan(603, 600, {'rejection_reasons': ['amostra insuficiente', 'taxa de acerto baixa']})
    assert plan['next_training_at'] == 720
    assert plan['remaining_samples'] == 117


def test_rejection_survives_readiness_refresh(monkeypatch):
    import pipeline
    monkeypatch.setattr(pipeline, '_training_outcomes', {'TEST': dict(samples=600,
        metrics={'eligible': False, 'rejection_reasons': ['taxa de acerto baixa']})})
    diagnostic = dict(stage='ready', symbol='TEST', labelable_contracts=603)
    updated = pipeline._training_status(diagnostic)
    assert updated['stage'] == 'model_rejected'
    assert updated['remaining_samples'] == 117
    assert diagnostic['stage'] == 'ready'
    assert pipeline._training_status(dict(diagnostic, symbol='OTHER'))['stage'] == 'ready'


def test_preflight_stops_before_estimators_or_model_write(tmp_path, monkeypatch):
    import json
    from unittest.mock import Mock
    import train_model
    frame = pd.DataFrame(dict(symbol=['TEST'] * 60, signal_epoch=range(60),
        entry_epoch=range(60), expiry_epoch=range(100, 160),
        feature_start_epoch=range(60), target=[0, 1] * 30,
        technical_signal=[0] * 60, tech_score=[0.] * 60))
    dataset = tmp_path / 'data.csv'
    frame.to_csv(dataset, index=False)
    (tmp_path / 'data.csv.metadata.json').write_text(json.dumps(dict(symbol='TEST', dataset_sha256='hash')))
    monkeypatch.setattr(train_model, 'FEATURES', [])
    monkeypatch.setattr(train_model, 'compatible', lambda *args: True)
    monkeypatch.setattr(train_model, 'fingerprint', lambda *args: 'hash')
    monkeypatch.setattr(train_model, 'temporal_partitions', lambda *args: (frame,) * 4)
    candidates = Mock(side_effect=AssertionError('não deve ajustar modelos'))
    monkeypatch.setattr(train_model, 'training_candidates', candidates)
    output = tmp_path / 'model.pkl'
    result = train_model.train(dataset, output)
    assert result['stage'] == 'evaluation_preflight'
    assert result['eligible'] is False
    candidates.assert_not_called()
    assert not output.exists()


def test_capacity_allows_entry_at_previous_expiry():
    frame = pd.DataFrame({'entry_epoch': [0, 10], 'expiry_epoch': [10, 20]})
    assert evaluation_capacity(frame) == 2
