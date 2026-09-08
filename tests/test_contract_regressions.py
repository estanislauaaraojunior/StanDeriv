"""Regressões dos defeitos encontrados na auditoria de 08/09."""
import csv
import json
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import config
import ai_predictor
import strategy
import feature_engine
from market_data import FIELDS, canonicalize, read_records, recover
from dataset_builder import build_dataset
from model_contract import FEATURE_WINDOW, compatible, specification
from risk_manager import RiskManager


@pytest.mark.parametrize('direction,confidence', [('SELL', .9), (None, .53), (None, .7), ('BUY', .54), ('BUY', float('nan'))])
def test_ai_disagreement_abstention_and_invalid_probability_block(direction, confidence):
    with patch('strategy.ind.price_action_features', return_value = None), patch('ai_predictor.predict', return_value = (direction, confidence)):
        signal, values = strategy._apply_ai_filter('BUY', [100] * 100, {'tech_score': .8}, use_ai_model = True)
    assert signal is None and 'win_probability' not in values


def test_ai_score_preserves_actual_technical_strength():
    with patch('strategy.ind.price_action_features', return_value = None), patch('ai_predictor.predict', return_value = ('BUY', .8)):
        _, weak = strategy._apply_ai_filter('BUY', [100] * 100, {'tech_score': .05}, use_ai_model = True)
        _, strong = strategy._apply_ai_filter('BUY', [100] * 100, {'tech_score': .9}, use_ai_model = True)
    assert weak['ai_score'] < strong['ai_score']
    assert weak['ai_score'] == round(.55 * .05 + .45 * .8, 4)


def test_recovery_preserves_original_and_marks_unknown_provenance(tmp_path):
    p = tmp_path / 'legacy.csv'
    p.write_text('120,x,A,2\n60,x,A,1\n60,x,A,1\n90,x,B,5\nbad,x,A,nan\n')
    before = p.read_bytes()
    report = recover(p, tmp_path / 'recovered')
    clean, rejected = read_records(tmp_path / 'recovered/prices.csv')
    assert p.read_bytes() == before
    assert report['duplicates'] == 1 and report['rejected'] == 1 and report['training_eligible'] == 0
    assert [(r['symbol'], r['epoch']) for r in clean] == [('A', 60), ('A', 120), ('B', 90)]
    with pytest.raises(ValueError):
        recover(p, tmp_path / 'recovered')


def test_conflicting_prices_quarantined_instead_of_arbitrarily_selected(tmp_path):
    p = tmp_path / 'ticks.csv'
    p.write_text('60,x,A,1\n60,x,A,2\n120,x,A,3\n')
    records, _ = read_records(p)
    clean, conflicts, duplicates = canonicalize(records)
    assert len(clean) == 1 and len(conflicts) == 2 and duplicates == 0


def test_legacy_data_cannot_silently_train(tmp_path):
    p = tmp_path / 'ticks.csv'
    p.write_text('60,x,R_100,1\n120,x,R_100,2\n')
    with pytest.raises(ValueError, match = 'proveniência'):
        build_dataset(p, tmp_path / 'dataset.csv')


def write_ticks(path, minutes = 130, price_fn = None):
    base = 1_800_000_000
    with path.open('w', newline = '') as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        for offset in range(0, minutes * 60, 2):
            price = price_fn(offset) if price_fn else 100 + math.sin(offset / 90) + offset * .0001
            w.writerow([base + offset, '', 'R_100', price, 'tick', 0, '', '', ''])


def test_labels_use_entry_and_expiry_not_mean_next_two_prices(tmp_path):
    source = tmp_path / 'ticks.csv'
    out = tmp_path / 'dataset.csv'
    # Primeira entrada ocorre no segundo 6000, após 100 candles completos.
    def price(t):
        if t < 6000:
            return 100 + math.sin(t / 50)
        if t == 6000:
            return 100
        if t < 6900:
            return 101
        return 99
    write_ticks(source, price_fn = price)
    build_dataset(source, out)
    df = pd.read_csv(out)
    first = df.iloc[0]
    assert first.entry_epoch == 1_800_006_000 and first.expiry_epoch == 1_800_006_900
    assert first.target == 0 and first.entry_price == 100 and first.exit_price == 99
    metadata = json.loads(Path(str(out) + '.metadata.json').read_text())
    assert compatible(metadata, 'R_100')


def test_ties_kept_as_separate_outcome_without_future_noise_filter(tmp_path):
    source = tmp_path / 'ticks.csv'
    out = tmp_path / 'dataset.csv'
    write_ticks(source, price_fn = lambda t: 100.)
    build_dataset(source, out)
    df = pd.read_csv(out)
    assert set(df.target) == {2}
    assert len(df) > 1 and (df.signal_epoch.diff().dropna() == 60).all()


def test_train_and_inference_features_are_identical_and_pa_available(tmp_path):
    source = tmp_path / 'ticks.csv'
    out = tmp_path / 'dataset.csv'
    write_ticks(source)
    build_dataset(source, out)
    from indicators import ticks_to_candles_by_time
    records, _ = read_records(source)
    candles = ticks_to_candles_by_time(records, 60)[: FEATURE_WINDOW]
    prices = [c['close'] for c in candles]
    live = ai_predictor._extract_features(prices, candles)
    trained = pd.read_csv(out).iloc[0][feature_engine.FEATURES].astype(float).values
    np.testing.assert_allclose(live, trained, rtol = 1e-10, atol = 1e-12)
    assert any(live[i] != 0 for i, k in enumerate(feature_engine.FEATURES) if k.startswith('pa_'))


def test_feature_window_ignores_older_prices():
    candles = [dict(epoch = i * 60, open = 100 + i * .1, high = 101 + i * .1, low = 99 + i * .1, close = 100 + i * .1) for i in range(200)]
    prices = [c['close'] for c in candles]
    assert feature_engine.compute_feature_map(prices, candles) == feature_engine.compute_feature_map(prices[-100:], candles[-100:])


def test_gap_prevents_false_contiguous_training(tmp_path):
    source = tmp_path / 'ticks.csv'
    out = tmp_path / 'dataset.csv'
    write_ticks(source)
    rows = list(csv.reader(source.open()))
    rows = [r for i, r in enumerate(rows) if i == 0 or not 1_800_005_990 < int(r[0]) < 1_800_006_200]
    with source.open('w', newline = '') as f:
        csv.writer(f).writerows(rows)
    with pytest.raises(ValueError, match = 'contínua'):
        build_dataset(source, out)


def test_duration_cannot_disagree_with_model_contract(tmp_path):
    with pytest.raises(ValueError, match = 'vencimento'):
        build_dataset(tmp_path / 'missing', tmp_path / 'out', lookforward = 2)
    assert ai_predictor.predict_duration([]) == config.DURATION


def test_old_model_or_wrong_symbol_is_incompatible():
    assert not compatible({}, 'R_100')
    assert not compatible(specification('R_25'), 'R_100')


def test_sequence_targets_end_at_last_feature_and_gaps_fail():
    from train_model import _build_sequence_dataset
    X = np.arange(6).reshape(-1, 1)
    seq = _build_sequence_dataset(X, 3, epochs = np.arange(6) * 60)
    assert seq[: , - 1, 0].tolist() == [2, 3, 4, 5]
    with pytest.raises(ValueError):
        _build_sequence_dataset(X, 3, epochs = [0, 60, 120, 240, 300, 360])


def test_risk_restart_keeps_daily_limits_and_peak():
    r = RiskManager(1000)
    r.record_result('R_100', 'BUY', 10, 15, - 10, {}, contract_id = '1')
    r.record_result('R_100', 'BUY', 10, 15, - 10, {}, contract_id = '2')
    r.record_result('R_100', 'BUY', 10, 15, - 10, {}, contract_id = '3')
    restored = RiskManager(9000)
    assert restored.balance == 970 and restored.is_paused()
    assert restored._daily_profit == - 30 and restored.to_state_dict()['drawdown_pct'] == 3
    assert not restored.record_result('R_100', 'BUY', 10, 15, - 10, {}, contract_id = '3')
    assert restored.balance == 970


def test_zero_profit_does_not_reset_losses_or_become_win():
    r = RiskManager(1000)
    r.record_result('R_100', 'BUY', 10, 15, - 10, {})
    r.record_result('R_100', 'BUY', 10, 15, 0, {})
    assert r._consec_losses == 1 and r._recent_results == [0]


def test_outbox_recovers_csv_after_interrupted_append(monkeypatch):
    r = RiskManager(1000)
    original = r._flush_outbox
    count = 0
    def interrupted():
        nonlocal count
        count += 1
        if count == 2:
            raise OSError('simulated disk interruption')
        original()
    monkeypatch.setattr(r, '_flush_outbox', interrupted)
    with pytest.raises(OSError):
        r.record_result('R_100', 'BUY', 10, 15, - 10, {}, contract_id = 'one')
    restored = RiskManager(1000)
    assert restored.balance == 990 and not restored._outbox
    assert not restored.record_result('R_100', 'BUY', 10, 15, - 10, {}, contract_id = 'one')
    with (restored._path.parent / 'operations.csv').open() as f:
        assert len(list(csv.DictReader(f))) == 1


def test_training_partitions_purge_overlapping_labels_and_features():
    from train_model import temporal_partitions
    epochs = np.arange(4000) * 60
    frame = pd.DataFrame(dict(signal_epoch = epochs, feature_start_epoch = epochs - 6000,
                            entry_epoch = epochs, expiry_epoch = epochs + 900, target = np.arange(4000) % 2))
    parts = temporal_partitions(frame)
    for earlier, later in zip(parts, parts[1:]):
        assert earlier.expiry_epoch.max() < later.feature_start_epoch.min()
    assert parts[-1].index.min() == 3200


def test_model_gate_rejects_missing_quality_and_incomparable_periods():
    from pipeline import _should_promote_model
    assert not _should_promote_model(None, {'auc': .9, 'accuracy': .9})[0]
    candidate = dict(eligible = True, auc = .7, accuracy = .65, f1 = .65, evaluation_start = 100, evaluation_end = 200, n_test = 40)
    assert _should_promote_model(None, candidate)[0]
    assert not _should_promote_model({'auc': .6}, candidate)[0]
    candidate['champion_metrics'] = dict(candidate, auc = .6, evaluation_start = 99)
    assert not _should_promote_model({'auc': .6}, candidate)[0]
    candidate['champion_metrics'] = dict(candidate, auc = .6, evaluation_start = 100)
    assert _should_promote_model({'auc': .6}, candidate)[0]


def test_model_loader_refuses_legacy_and_mismatching_contract(tmp_path, monkeypatch):
    import joblib
    path = tmp_path / 'model.pkl'
    monkeypatch.setattr(config, 'AI_MODEL_PATH', str(path))
    for payload in ({'model': None, 'features': feature_engine.FEATURES},
                    {'model': None, 'features': feature_engine.FEATURES, 'metadata': specification('R_25')}):
        joblib.dump(payload, path)
        ai_predictor.reset()
        assert ai_predictor.status()['ready'] is False
        assert ai_predictor.predict([100] * 100)[0] is None


def test_one_minute_and_fifteen_minute_contracts_cannot_share_bundle(monkeypatch):
    metadata = specification(config.get_active_symbol())
    monkeypatch.setattr(config, 'DURATION', 1)
    assert not compatible(metadata, config.get_active_symbol())


def test_distinct_accounts_keep_distinct_risk_history():
    r = RiskManager(1000)
    r.bind_account('account-a', 1000)
    r.record_result('R_100', 'BUY', 10, 15, - 10, {}, contract_id = 'a1')
    r.bind_account('account-b', 2000)
    assert r.balance == 2000 and r._daily_profit == 0
    r.record_result('R_100', 'BUY', 10, 15, - 20, {}, contract_id = 'b1')
    r.bind_account('account-a', 990)
    assert r.balance == 990 and r._daily_profit == - 10
    assert r.to_state_dict()['drawdown_pct'] == 1


def test_writer_refuses_to_append_v2_rows_to_legacy_file(tmp_path):
    from market_data import append_record
    path = tmp_path / 'old.csv'
    path.write_text('60,x,R_100,100\n')
    original = path.read_bytes()
    with pytest.raises(ValueError):
        append_record(path, 61, 'R_100', 101)
    assert path.read_bytes() == original


def test_scanner_requests_time_candles(monkeypatch):
    import pipeline
    sent = []
    class FakeSocket:
        def __init__(self, url, on_open, on_message, on_error, on_close):
            self.on_open = on_open
            self.on_message = on_message
            self.on_close = on_close
        def run_forever(self):
            self.on_open(self)
        def send(self, raw):
            message = json.loads(raw)
            sent.append(message)
            if 'authorize' in message:
                self.on_message(self, json.dumps({'msg_type': 'authorize'}))
            else:
                self.on_message(self, json.dumps({'msg_type': 'candles', 'candles': []}))
        def close(self):
            self.on_close(self)
    monkeypatch.setattr(pipeline, '_get_ws_url', lambda: 'offline')
    monkeypatch.setattr(pipeline, '_uses_bearer_auth', lambda: False)
    monkeypatch.setattr(pipeline.websocket, 'WebSocketApp', FakeSocket)
    pipeline._fetch_prices_for_symbol('R_100', 100)
    request = next(m for m in sent if 'ticks_history' in m)
    assert request['style'] == 'candles' and request['granularity'] == config.CANDLE_TIMEFRAME_SEC


def test_minimum_quality_gate_uses_nonoverlapping_contracts():
    from train_model import _metrics
    class Model:
        classes_ = np.array([0, 1])
        def predict_proba(self, X):
            return np.tile([.1, .9], (len(X), 1))
    frame = pd.DataFrame(np.zeros((100, len(feature_engine.FEATURES))), columns = feature_engine.FEATURES)
    frame['target'] = 1
    frame['technical_signal'] = 1
    frame['tech_score'] = .9
    frame['entry_epoch'] = np.arange(100) * 60
    frame['expiry_epoch'] = frame.entry_epoch + 900
    result = _metrics(Model(), frame)
    assert result['trades'] == 7 and not result['eligible']


def test_dataset_does_not_use_other_symbols(tmp_path):
    source = tmp_path / 'ticks.csv'
    out = tmp_path / 'dataset.csv'
    write_ticks(source)
    with source.open('a', newline = '') as f:
        w = csv.writer(f)
        for i in range(100):
            w.writerow([1_800_000_000 + i, '', 'other', 100000, 'tick', 0, '', '', ''])
    build_dataset(source, out)
    frame = pd.read_csv(out)
    assert set(frame.symbol) == {'R_100'} and frame.ema9.max() < 200


def test_training_calibration_and_serialization_on_synthetic_data(tmp_path, monkeypatch):
    """Exercita o fluxo novo com quatro árvores e dados artificiais, sem modelos reais."""
    import joblib, train_model, xgboost
    from model_contract import fingerprint, write_json_atomic
    forest = train_model.RandomForestClassifier
    boosted = xgboost.XGBClassifier
    monkeypatch.setattr(train_model, 'RandomForestClassifier', lambda ** kw: forest(n_estimators = 4, max_depth = 2, min_samples_leaf = 2, random_state = 42, n_jobs = 1))
    monkeypatch.setattr(xgboost, 'XGBClassifier', lambda ** kw: boosted(n_estimators = 4, max_depth = 2, random_state = 42, n_jobs = 1))
    n = 4000
    labels = np.arange(n) % 2
    frame = pd.DataFrame(np.zeros((n, len(feature_engine.FEATURES))), columns = feature_engine.FEATURES)
    frame['ema9'] = labels
    frame['target'] = labels
    frame['technical_signal'] = labels
    frame['tech_score'] = .9
    frame['symbol'] = config.get_active_symbol()
    frame['signal_epoch'] = 1_800_000_000 + np.arange(n) * 60
    frame['entry_epoch'] = frame.signal_epoch
    frame['expiry_epoch'] = frame.entry_epoch + 900
    frame['feature_start_epoch'] = frame.signal_epoch - 6000
    source = tmp_path / 'synthetic.csv'
    frame.to_csv(source, index = False)
    write_json_atomic(str(source) + '.metadata.json', dict(specification(config.get_active_symbol()), dataset_sha256 = fingerprint(source)))
    dest = tmp_path / 'candidate.pkl'
    result = train_model.train(source, dest)
    payload = joblib.load(dest)
    assert payload['model'].__class__.__module__ == 'train_model'
    assert result['eligible'] and payload['metadata']['approved']
    assert result['n_test'] == 800 and result['trades'] < 100
    monkeypatch.setattr(config, 'AI_MODEL_PATH', str(dest))
    ai_predictor.reset()
    assert ai_predictor.status()['ready']
    assert ai_predictor.model_id() == fingerprint(dest)
    sidecar = json.loads(Path(str(dest) + '.metadata.json').read_text())
    assert sidecar['artifact_sha256'] == fingerprint(dest)


def test_model_quality_gate_requires_technical_agreement():
    from train_model import _metrics

    class Model:
        classes_ = np.array([0, 1])

        def predict_proba(self, X):
            return np.tile([.01, .99], (len(X), 1))

    frame = pd.DataFrame(np.zeros((100, len(feature_engine.FEATURES))), columns=feature_engine.FEATURES)
    frame['target'] = 1
    frame['tech_score'] = .9
    frame['entry_epoch'] = np.arange(100) * 60
    frame['expiry_epoch'] = frame.entry_epoch + 900
    for technical in (-1, 0):
        frame['technical_signal'] = technical
        result = _metrics(Model(), frame)
        assert result['trades'] == 0
        assert result['eligible'] is False
