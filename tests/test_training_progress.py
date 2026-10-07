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
    """Garante que o histórico mínimo reserva ao menos 120 horizontes não sobrepostos no trecho final."""
    horizon = duration_seconds(config.DURATION, config.DURATION_UNIT)
    rows = minimum_history_candles() - FEATURE_WINDOW - horizon // config.CANDLE_TIMEFRAME_SEC
    assert int(rows * .2) * config.CANDLE_TIMEFRAME_SEC >= 120 * horizon


def test_training_keeps_history_and_skips_unchanged_labels(tmp_path, monkeypatch):
    """Garante que o treino usa todo o histórico e só retreina quando os rótulos mudam."""
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
        """Simula build_dataset registrando o snapshot lido e gravando o rótulo atual."""
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
    """Garante que candidato inelegível informa na razão os bloqueios específicos."""
    eligible, reason = pipeline._should_promote_model(None, {
        'eligible': False, 'rejection_reasons': ['amostra insuficiente: 9/30', 'AUC abaixo do mínimo']})
    assert not eligible
    assert '9/30' in reason and 'AUC' in reason


@pytest.mark.parametrize('virtual,expected', [(1, True), (0, False), (None, False)])
def test_startup_verifies_actual_demo_account(monkeypatch, virtual, expected):
    """Garante que a validação do token aceita apenas contas cujo is_virtual indica demo."""
    monkeypatch.setattr(pipeline.deriv_session, 'is_bearer_token', lambda *args: False)
    class Socket:
        """WebSocket falso que responde a autorização com a conta parametrizada."""
        def __init__(self, url, **callbacks):
            """Guarda os callbacks recebidos."""
            self.callbacks = callbacks
        def run_forever(self):
            """Entrega imediatamente a mensagem de autorização."""
            self.callbacks['on_message'](self, json.dumps({'msg_type': 'authorize',
                'authorize': {'is_virtual': virtual, 'loginid': 'offline', 'currency': 'USD'}}))
        def close(self):
            """Dispara o callback de fechamento."""
            self.callbacks['on_close'](self)
    monkeypatch.setattr(pipeline.websocket, 'WebSocketApp', Socket)
    assert pipeline._validate_deriv_token(demo=True, timeout_sec=1) is expected


def test_startup_retries_transient_dns_auth_error(monkeypatch):
    """Garante que a validação do token repete a tentativa após erro transitório de DNS."""
    monkeypatch.setattr(pipeline.deriv_session, 'is_bearer_token', lambda *args: False)
    monkeypatch.setattr(pipeline.time, 'sleep', lambda _seconds: None)
    attempts = {'count': 0}

    class Socket:
        """WebSocket falso que falha com erro de DNS na primeira tentativa."""
        def __init__(self, url, **callbacks):
            """Guarda os callbacks recebidos."""
            self.callbacks = callbacks
        def run_forever(self):
            """Emite erro de DNS na primeira chamada e autoriza conta demo nas seguintes."""
            attempts['count'] += 1
            if attempts['count'] == 1:
                self.callbacks['on_error'](self, OSError(-3, 'Temporary failure in name resolution'))
                return
            self.callbacks['on_message'](self, json.dumps({'msg_type': 'authorize',
                'authorize': {'is_virtual': 1, 'loginid': 'offline', 'currency': 'USD'}}))
        def close(self):
            """Dispara o callback de fechamento."""
            self.callbacks['on_close'](self)

    monkeypatch.setattr(pipeline.websocket, 'WebSocketApp', Socket)
    assert pipeline._validate_deriv_token(demo=True, timeout_sec=1) is True
    assert attempts['count'] == 2


def test_fixed_symbol_synchronizes_training_and_execution(monkeypatch):
    """Garante que o símbolo fixo é propagado ao treino, ao runtime e ao executor."""
    import executor, strategy, dataset_builder, train_model
    for module in (config, executor, strategy, dataset_builder, train_model, pipeline):
        if hasattr(module, 'SYMBOL'):
            monkeypatch.setattr(module, 'SYMBOL', 'frxUSDCAD')
    monkeypatch.setattr(config, '_active_symbol', 'R_100')
    pipeline._set_runtime_symbol(config.SYMBOL)
    assert config.get_active_symbol() == pipeline._runtime_symbol() == executor.SYMBOL == 'frxUSDCAD'
