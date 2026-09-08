"""Isolamento de artefatos: nenhum teste usa estado ou modelos operacionais."""
import os
from pathlib import Path
from unittest.mock import MagicMock
import pytest

os.environ['DERIV_TOKEN'] = 'offline_test_token'
os.environ['DERIV_APP_ID'] = '1089'


@pytest.fixture(autouse=True)
def isolated_operations(tmp_path, monkeypatch):
    import config, risk_manager, executor, ai_predictor
    monkeypatch.setattr(risk_manager,'OPERATIONS_LOG',str(tmp_path/'operations.csv'))
    monkeypatch.setattr(executor,'_STATE_JSON',tmp_path/'state.json')
    monkeypatch.setattr(executor.threading,'Timer',MagicMock())
    monkeypatch.setattr(config,'AI_MODEL_PATH',str(tmp_path/'model.pkl'))
    monkeypatch.setattr(config,'_active_symbol',config.SYMBOL)
    ai_predictor.reset()
    yield
    ai_predictor.reset()
