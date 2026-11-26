"""Base de cinco minutos exclusiva da IA, com vencimento preservado em quinze."""
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import ai_predictor
import config
import dataset_builder
from feature_engine import FEATURES
from market_data import FIELDS
from model_contract import (FEATURE_WINDOW, compatible, minimum_history_candles,
                            minimum_training_rows, model_timeframe_sec, specification)


BASE = 1_800_000_000


@pytest.fixture
def model_only(monkeypatch):
    monkeypatch.setattr(config, 'USE_AI_MODEL', True)
    for key in ('USE_WEIGHTED_SIGNAL', 'PRICE_ACTION_FILTER', 'CANDLE_PATTERN_FILTER'):
        monkeypatch.setattr(config, key, False)
    monkeypatch.setattr(config, 'DURATION', 15)
    monkeypatch.setattr(config, 'DURATION_UNIT', 'm')


def write_source(path, *, candle_timeframe=300, gap=None):
    with path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        for i in range(FEATURE_WINDOW):
            price = 100 + i * .01
            writer.writerow([BASE + i * candle_timeframe, '', 'R_100', price,
                             'candle', candle_timeframe, price - .01, price + .02, price - .02])
        for offset in range(FEATURE_WINDOW * 300, FEATURE_WINDOW * 300 + 1802, 2):
            if gap and gap[0] < offset < gap[1]:
                continue
            writer.writerow([BASE + offset, '', 'R_100', 101 + offset * .00001,
                             'tick', 0, '', '', ''])


@pytest.mark.parametrize('extra_module', [None, 'USE_WEIGHTED_SIGNAL',
                                        'PRICE_ACTION_FILTER', 'CANDLE_PATTERN_FILTER'])
def test_timeframe_is_exclusive_to_model_only(model_only, monkeypatch, extra_module):
    assert config.CANDLE_TIMEFRAME_SEC == 900
    if extra_module:
        monkeypatch.setattr(config, extra_module, True)
    expected = 900 if extra_module else 300
    assert model_timeframe_sec() == expected
    assert specification('R_100')['timeframe_sec'] == expected
    assert specification('R_100')['training_history_candles'] == 28 * 86400 // expected
    assert minimum_history_candles() == 672
    assert config.CANDLE_TIMEFRAME_SEC == 900


def test_ai_disabled_preserves_operational_timeframe(model_only, monkeypatch):
    monkeypatch.setattr(config, 'USE_AI_MODEL', False)
    assert model_timeframe_sec() == 900


def test_model_only_metadata_rejects_old_timeframe(model_only, monkeypatch):
    assert minimum_training_rows() == 593
    metadata = specification('R_100')
    assert metadata['duration'] == 15 and metadata['duration_unit'] == 'm'
    assert compatible(metadata, 'R_100')
    metadata['timeframe_sec'] = 900
    assert not compatible(metadata, 'R_100')
    monkeypatch.setattr(config, 'USE_WEIGHTED_SIGNAL', True)
    assert minimum_training_rows() == 579


def test_five_minute_dataset_has_fifteen_minute_labels_and_matching_features(tmp_path, model_only):
    source, output = tmp_path / 'source.csv', tmp_path / 'dataset.csv'
    write_source(source)
    estimated = dataset_builder.count_labelable_contracts(source, 'R_100')
    assert dataset_builder.build_dataset(source, output, lookforward=3, symbol='R_100') == estimated
    frame = pd.read_csv(output)
    first = frame.iloc[0]
    assert first.signal_epoch == BASE + FEATURE_WINDOW * 300
    assert (frame.expiry_epoch - frame.entry_epoch == 900).all()
    assert (frame.signal_epoch.diff().dropna() == 300).all()
    series = dataset_builder._load_training_series(source, 'R_100')
    candles = series[3][:FEATURE_WINDOW]
    inference = ai_predictor._extract_features([row['close'] for row in candles], candles)
    np.testing.assert_allclose(inference, first[FEATURES].astype(float), rtol=1e-10, atol=1e-12)
    metadata = json.loads(Path(str(output) + '.metadata.json').read_text())
    assert metadata['timeframe_sec'] == 300 and metadata['duration'] == 15
    assert compatible(metadata, 'R_100')
    with pytest.raises(ValueError, match='vencimento'):
        dataset_builder.build_dataset(source, tmp_path / 'wrong.csv', lookforward=1, symbol='R_100')


def test_fifteen_minute_ohlc_is_not_split_into_five_minute_candles(tmp_path, model_only):
    source = tmp_path / 'source.csv'
    write_source(source, candle_timeframe=900)
    assert dataset_builder.count_labelable_contracts(source, 'R_100') == 0
    _, _, _, candles, _ = dataset_builder._load_training_series(source, 'R_100')
    assert len(candles) < FEATURE_WINDOW


def test_five_minute_labels_still_reject_tick_gaps(tmp_path, model_only):
    source = tmp_path / 'source.csv'
    start = FEATURE_WINDOW * 300
    write_source(source, gap=(start + 100, start + 130))
    rows = list(dataset_builder._labelable_rows(dataset_builder._load_training_series(source, 'R_100')))
    assert rows == []


@pytest.fixture
def loaded_predictor(monkeypatch):
    class Model:
        classes_ = np.array([0, 1])

        def predict_proba(self, _values):
            return np.array([[.2, .8]])

    monkeypatch.setattr(ai_predictor, '_model_loaded', True)
    monkeypatch.setattr(ai_predictor, '_loaded_symbol', config.get_active_symbol())
    monkeypatch.setattr(ai_predictor, '_model', Model())
    monkeypatch.setattr(ai_predictor, '_metadata', {'model_id': 'offline-five-minute-model'})
    monkeypatch.setattr(ai_predictor, '_extract_features', lambda *_args: [0.])


def candles_for(timeframe):
    return [dict(epoch=BASE + i * timeframe, open=100., high=101., low=99., close=100.)
            for i in range(FEATURE_WINDOW)]


def test_predictor_requires_five_minute_continuity_alignment_and_closed_candle(model_only, loaded_predictor):
    candles = candles_for(300)
    closed_at = candles[-1]['epoch'] + 300
    assert ai_predictor.predict([100.] * FEATURE_WINDOW, candles, evaluation_epoch=closed_at) == ('BUY', .8)
    assert ai_predictor.predict([100.] * FEATURE_WINDOW, candles, evaluation_epoch=closed_at - 1) == (None, .0)
    assert ai_predictor.predict([100.] * FEATURE_WINDOW, candles_for(900)) == (None, .0)
    shifted = [dict(row, epoch=row['epoch'] + 1) for row in candles]
    assert ai_predictor.predict([100.] * FEATURE_WINDOW, shifted) == (None, .0)
    gapped = [dict(row) for row in candles]
    gapped[-1]['epoch'] += 300
    assert ai_predictor.predict([100.] * FEATURE_WINDOW, gapped) == (None, .0)


def test_predictor_preserves_fifteen_minute_input_in_mixed_mode(model_only, loaded_predictor, monkeypatch):
    monkeypatch.setattr(config, 'USE_WEIGHTED_SIGNAL', True)
    assert ai_predictor.predict([100.] * FEATURE_WINDOW, candles_for(900)) == ('BUY', .8)
    assert ai_predictor.predict([100.] * FEATURE_WINDOW, candles_for(300)) == (None, .0)
