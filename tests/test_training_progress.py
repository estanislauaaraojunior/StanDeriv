"""Regressões de cobertura temporal e progresso do treinamento."""
import csv
import json
from unittest.mock import Mock
import pytest

import pandas as pd

import config
import pipeline
from model_contract import minimum_history_candles, FEATURE_WINDOW, duration_seconds


def test_collection_budget_reserves_final_nonoverlapping_opportunities():
    horizon = duration_seconds(config.DURATION, config.DURATION_UNIT)
    rows = minimum_history_candles() - FEATURE_WINDOW - horizon // config.CANDLE_TIMEFRAME_SEC
    assert int(rows * .2) * config.CANDLE_TIMEFRAME_SEC >= 120 * horizon


def test_training_keeps_history_and_skips_unchanged_labels(tmp_path, monkeypatch):
    source = tmp_path / 'ticks.csv'
    with source.open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'symbol', 'price'])
        writer.writerows((i, 'R_100', 100) for i in range(200_005))
    monkeypatch.setattr(pipeline, 'TICKS_CSV', str(source))
    monkeypatch.setattr(pipeline, '_runtime_symbol', lambda: 'R_100')
    monkeypatch.delattr(pipeline._run_training, '_last_dataset_key', raising=False)
    observed = []
    label = ['initial']

    def build(snapshot, output, **kwargs):
        frame = pd.read_csv(snapshot)
        observed.append((len(frame), int(frame.epoch.min())))
        output.write_text(label[0])
        return 1000

    trainer = Mock(return_value={'best_model': 'test', 'eligible': False})
    monkeypatch.setattr(pipeline.dataset_builder, 'build_dataset', build)
    monkeypatch.setattr(pipeline.train_model, 'train', trainer)
    dataset = tmp_path / 'dataset.csv'
    model = str(tmp_path / 'model.candidate')
    assert pipeline._run_training(dataset, model)[0]
    assert not pipeline._run_training(dataset, model)[0]
    assert trainer.call_count == 1
    label[0] = 'new matured contracts'
    assert pipeline._run_training(dataset, model)[0]
    assert trainer.call_count == 2
    assert observed == [(200_005, 0)] * 3
    assert not (tmp_path / 'ticks.csv.train_snapshot').exists()


def test_rejected_candidate_reports_specific_blockers():
    eligible, reason = pipeline._should_promote_model(None, {
        'eligible': False, 'rejection_reasons': ['amostra insuficiente: 9/30', 'AUC abaixo do mínimo']})
    assert not eligible
    assert '9/30' in reason and 'AUC' in reason


@pytest.mark.parametrize('virtual,expected', [(1, True), (0, False), (None, False)])
def test_startup_verifies_actual_demo_account(monkeypatch, virtual, expected):
    monkeypatch.setattr(pipeline.deriv_session, 'is_bearer_token', lambda *args: False)
    class Socket:
        def __init__(self, url, **callbacks):
            self.callbacks = callbacks
        def run_forever(self):
            self.callbacks['on_message'](self, json.dumps({'msg_type': 'authorize',
                'authorize': {'is_virtual': virtual, 'loginid': 'offline', 'currency': 'USD'}}))
        def close(self):
            self.callbacks['on_close'](self)
    monkeypatch.setattr(pipeline.websocket, 'WebSocketApp', Socket)
    assert pipeline._validate_deriv_token(demo=True, timeout_sec=1) is expected


def test_fixed_symbol_synchronizes_training_and_execution(monkeypatch):
    import executor, strategy, dataset_builder, train_model
    for module in (config, executor, strategy, dataset_builder, train_model, pipeline):
        if hasattr(module, 'SYMBOL'):
            monkeypatch.setattr(module, 'SYMBOL', 'frxUSDCAD')
    monkeypatch.setattr(config, '_active_symbol', 'R_100')
    pipeline._set_runtime_symbol(config.SYMBOL)
    assert config.get_active_symbol() == pipeline._runtime_symbol() == executor.SYMBOL == 'frxUSDCAD'
