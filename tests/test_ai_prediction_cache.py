"""Mesmas features da IA não recalculam o estimador a cada tick."""
from unittest.mock import Mock

import pytest

import ai_predictor
import config

BASE = 1_800_000_000


@pytest.fixture
def predictor(monkeypatch):
    monkeypatch.setattr(config, 'USE_AI_MODEL', True)
    for key in ('USE_WEIGHTED_SIGNAL', 'PRICE_ACTION_FILTER', 'CANDLE_PATTERN_FILTER'):
        monkeypatch.setattr(config, key, False)
    model = Mock(classes_=[0, 1], predict_proba=Mock(return_value=[[.2, .8]]))
    monkeypatch.setattr(ai_predictor, '_model_loaded', True)
    monkeypatch.setattr(ai_predictor, '_loaded_symbol', config.get_active_symbol())
    monkeypatch.setattr(ai_predictor, '_model', model)
    monkeypatch.setattr(ai_predictor, '_metadata', {'model_id': 'cache-test'})
    extract = Mock(return_value=[0.])
    monkeypatch.setattr(ai_predictor, '_extract_features', extract)
    return model, extract


def candles(timeframe=300):
    return [dict(epoch=BASE + i * timeframe, open=100., high=101., low=99., close=100.)
            for i in range(50)]


def test_repeated_ticks_reuse_prediction_but_keep_causal_checks(predictor):
    model, extract = predictor
    rows = candles()
    closed_at = rows[-1]['epoch'] + 300
    for tick in range(120):
        assert ai_predictor.predict([100.] * 50, rows, evaluation_epoch=closed_at + tick) == ('BUY', .8)
        assert ai_predictor.prediction_model_id() == 'cache-test'
    assert ai_predictor.predict([100.] * 50, rows, evaluation_epoch=closed_at - 1) == (None, 0.)
    assert ai_predictor.prediction_model_id() == ''
    model.predict_proba.assert_called_once()
    extract.assert_called_once()


@pytest.mark.parametrize('change', ['close', 'epoch', 'model', 'confidence', 'prices'])
def test_changed_inputs_invalidate_cached_prediction(predictor, monkeypatch, change):
    model, extract = predictor
    rows = candles()
    prices = [100.] * 50
    ai_predictor.predict(prices, rows)
    if change == 'close': rows[-1]['close'] = 100.5
    if change == 'epoch': rows = [dict(row, epoch=row['epoch'] + 300) for row in rows]
    if change == 'model': monkeypatch.setattr(ai_predictor, '_metadata', {'model_id': 'other'})
    if change == 'confidence': monkeypatch.setattr(config, 'AI_CONFIDENCE_MIN', .85)
    if change == 'prices': prices[-1] = 100.5
    ai_predictor.predict(prices, rows)
    assert model.predict_proba.call_count == 2 and extract.call_count == 2


def test_mixed_mode_preserves_uncached_fifteen_minute_inference(predictor, monkeypatch):
    model, _ = predictor
    monkeypatch.setattr(config, 'USE_WEIGHTED_SIGNAL', True)
    for _ in range(2):
        assert ai_predictor.predict([100.] * 50, candles(900)) == ('BUY', .8)
    assert model.predict_proba.call_count == 2


def test_reset_discards_cached_prediction(predictor):
    ai_predictor.predict([100.] * 50, candles())
    assert ai_predictor._prediction_cache is not None
    ai_predictor.reset()
    assert ai_predictor._prediction_cache is None and ai_predictor.model_id() == ''
