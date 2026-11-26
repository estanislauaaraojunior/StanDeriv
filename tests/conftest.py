"""Isolamento de artefatos: nenhum teste usa estado ou modelos operacionais."""
import os
from pathlib import Path
from unittest.mock import MagicMock
import pytest

os.environ['DERIV_TOKEN'] = 'offline_test_token'
os.environ['DERIV_APP_ID'] = '1089'


@pytest.fixture(autouse=True)
def isolated_operations(tmp_path, monkeypatch):
    """Redireciona log, estado, timer e modelo para diretório temporário e reseta a IA em cada teste."""
    import config, risk_manager, executor, ai_predictor
    monkeypatch.setattr(risk_manager,'OPERATIONS_LOG',str(tmp_path/'operations.csv'))
    monkeypatch.setattr(executor,'_STATE_JSON',tmp_path/'state.json')
    monkeypatch.setattr(executor,'_BOT_STATUS_JSON',tmp_path/'bot_status.json')
    monkeypatch.setattr(executor.threading,'Timer',MagicMock())
    monkeypatch.setattr(config,'AI_MODEL_PATH',str(tmp_path/'model.pkl'))
    monkeypatch.setattr(config,'_active_symbol',config.SYMBOL)
    import pipeline
    monkeypatch.setattr(pipeline, '_training_outcomes', {})
    monkeypatch.setattr(pipeline, '_training_readiness_cache', {})
    monkeypatch.setattr(pipeline, '_training_progress_instances', {})
    monkeypatch.setattr(pipeline, '_training_survey_at', 0.0)
    monkeypatch.setattr(pipeline, '_training_survey_offset', 0)
    monkeypatch.setattr(pipeline, '_training_survey_seen', set())
    monkeypatch.setattr(pipeline, '_training_next_symbol', '')
    monkeypatch.setattr(pipeline, '_training_rotation_allowed', True)
    from api_keepalive import ApiKeepalive
    monkeypatch.setattr(ApiKeepalive, 'start', MagicMock())
    from market_calendar import MarketCalendar
    monkeypatch.setattr(pipeline, '_BOT_STATUS_JSON', tmp_path/'bot_status.json')
    # Recuperação pública da IA também é substituída por dados sintéticos nos testes.
    monkeypatch.setattr(pipeline, '_training_history_provider', lambda: None)
    monkeypatch.setattr(pipeline, '_market_calendar', MarketCalendar(lambda _request: {'trading_times': {'markets': []}}))
    ai_predictor.reset()
    yield
    ai_predictor.reset()
