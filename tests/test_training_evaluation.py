import csv
import json
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import config
from feature_engine import FEATURES, EXPERIMENTAL_FEATURES, compute_experimental_features
from payout_quotes import ProposalRecorder, QuoteHistory
from train_model import _metrics, walk_forward_partitions, selection_key, temporal_partitions


class FixedModel:
    classes_ = np.array([0, 1])
    def predict_proba(self, X):
        return np.tile([.1, .9], (len(X), 1))


def frame(n=100, spacing=60):
    df = pd.DataFrame(np.zeros((n, len(FEATURES))), columns=FEATURES)
    df['target'] = 1
    df['technical_signal'] = 1
    df['tech_score'] = .9
    df['entry_epoch'] = 1_800_000_000 + np.arange(n)*spacing
    df['expiry_epoch'] = df.entry_epoch + 900
    df['signal_epoch'] = df.entry_epoch
    df['feature_start_epoch'] = df.entry_epoch-6000
    df['symbol'], df['duration'], df['duration_unit'] = 'R_100', 15, 'm'
    return df


def test_components_explain_disagreement_without_changing_combined_gate():
    df = frame()
    df['technical_signal'] = 0
    m = _metrics(FixedModel(), df)
    assert m['trades'] == 0
    assert m['components']['ai_only']['trades'] == 7
    assert m['components']['technical_only']['wins'] == 0
    assert m['blockers']['direction_disagreement'] == 100
    assert sum(m['blockers'].values())+m['trades'] == len(df)
    json.dumps(m, allow_nan=False)


def test_calibration_penalizes_confident_wrong_predictions_and_missing_ties():
    good = _metrics(FixedModel(), frame())
    df = frame()
    df['target'] = 2
    bad = _metrics(FixedModel(), df)
    assert bad['wins'] == 0
    assert bad['calibration']['log_loss'] > good['calibration']['log_loss']
    assert bad['calibration']['multiclass_brier'] > good['calibration']['multiclass_brier']
    assert sum(b['count'] for b in good['calibration']['reliability_bins']) == 100


def test_segments_and_drawdown_use_selected_nonoverlapping_trades():
    df = frame()
    m = _metrics(FixedModel(), df)
    assert m['segments']['direction:BUY']['trades'] == m['trades'] == 7
    assert m['max_drawdown_pct'] == 0
    df['target'] = 0
    assert _metrics(FixedModel(), df)['max_drawdown_pct'] > 0


def test_walk_forward_never_crosses_final_holdout_or_feature_embargo():
    df = frame(6000)
    df['target'] = np.arange(len(df)) % 2
    test = temporal_partitions(df)[-1]
    development = df[df.expiry_epoch < test.feature_start_epoch.iloc[0]]
    windows = list(walk_forward_partitions(development))
    for fit, calibration, evaluation in windows:
        assert fit.expiry_epoch.max() < calibration.feature_start_epoch.min()
        assert calibration.expiry_epoch.max() < evaluation.feature_start_epoch.min()
        assert evaluation.expiry_epoch.max() < test.feature_start_epoch.min()
    for previous, current in zip(windows, windows[1:]):
        assert previous[2].expiry_epoch.max() < current[2].feature_start_epoch.min()
        assert len(current[0]) > len(previous[0])


def test_selection_prefers_supported_stability_over_auc_alone():
    stable = dict(trades=40, worst_expectancy=.1, mean_expectancy=.2,
                  max_drawdown_pct=2, mean_log_loss=.6, mean_auc=.58)
    unstable = dict(stable, worst_expectancy=-.2, mean_auc=.8)
    scarce = dict(stable, trades=2, worst_expectancy=.9)
    assert selection_key(stable) > selection_key(unstable)
    assert selection_key(stable) > selection_key(scarce)


def test_horizon_features_are_relative_causal_and_do_not_replace_baseline():
    prices = list(np.linspace(100, 110, 100))
    a = compute_experimental_features(prices, 1_800_000_000, 60)
    b = compute_experimental_features([p*10 for p in prices], 1_800_000_000, 60)
    assert set(a) == set(EXPERIMENTAL_FEATURES)
    assert len(FEATURES) == 27
    assert a == pytest.approx(b)
    assert a['return_15m'] == pytest.approx(prices[-1]/prices[-16]-1)
    assert compute_experimental_features(prices, 1_800_000_000, 1) is None


def test_proposal_sampling_is_rate_limited_correlated_and_never_buys(tmp_path):
    path = tmp_path/'quotes.csv'
    recorder = ProposalRecorder(path, 'R_100', 15, 'm')
    ws = Mock()
    recorder.sample(ws, bearer=True, now=100)
    recorder.sample(ws, bearer=True, now=101)
    sent = [json.loads(call.args[0]) for call in ws.send.call_args_list]
    assert len(sent) == 2 and all('proposal' in row and 'buy' not in row for row in sent)
    assert all(row['underlying_symbol'] == 'R_100' for row in sent)
    response = {'req_id': sent[0]['req_id'], 'proposal': {'id': 'test', 'ask_price': 1, 'payout': 1.7}}
    assert recorder.handle(response, now=102)
    assert not recorder.handle(response, now=103)
    history = QuoteHistory(path)
    assert history.lookup('R_100', 15, 'm', 1, 101) is None
    assert history.lookup('R_100', 15, 'm', 1, 102) == pytest.approx(.7)
    assert history.lookup('R_100', 15, 'm', 0, 102) is None
    assert history.lookup('R_100', 5, 'm', 1, 102) is None
    assert history.lookup('R_100', 15, 'm', 1, 2000) is None


def test_sampling_errors_do_not_escape_into_collector_reconnection(tmp_path):
    recorder = ProposalRecorder(tmp_path/'q.csv', 'R_100', 15, 'm')
    ws = Mock()
    recorder.sample(ws, bearer=False, now=100)
    rid = json.loads(ws.send.call_args_list[0].args[0])['req_id']
    assert recorder.handle({'req_id': rid, 'error': {'code': 'MarketIsClosed'}}, now=101)
    assert not (tmp_path/'q.csv').exists()


def test_observed_evaluation_does_not_fill_missing_historical_payout(tmp_path):
    from payout_quotes import FIELDS
    path = tmp_path/'quotes.csv'
    with path.open('w') as f:
        csv.writer(f).writerow(FIELDS)
    m = _metrics(FixedModel(), frame(), quotes=QuoteHistory(path))
    assert m['trades'] == 0
    assert m['blockers']['missing_observed_payout'] == 100


def test_retraining_requires_new_matured_labels_even_after_time_backstop(monkeypatch):
    import pipeline
    monkeypatch.setattr(pipeline, '_retrain_state', {'last_epoch': 0, 'last_label_count': 1000})
    monkeypatch.setattr(pipeline, '_count_labelable_contracts_for_training', lambda: 1001)
    assert not pipeline._should_retrain_now()[0]
    monkeypatch.setattr(pipeline, '_count_labelable_contracts_for_training', lambda: 1120)
    assert pipeline._should_retrain_now()[0]


def test_changing_final_targets_cannot_change_walk_forward_selection(tmp_path, monkeypatch):
    import train_model, xgboost
    from model_contract import specification, fingerprint, write_json_atomic
    rf, xgb = train_model.RandomForestClassifier, xgboost.XGBClassifier
    monkeypatch.setattr(train_model, 'RandomForestClassifier', lambda **kw: rf(n_estimators=4, max_depth=2, random_state=42, n_jobs=1))
    monkeypatch.setattr(xgboost, 'XGBClassifier', lambda **kw: xgb(n_estimators=4, max_depth=2, random_state=42, n_jobs=1))
    df = frame(4000)
    df['target'] = np.arange(len(df)) % 2
    df['ema9'] = df.target
    df['technical_signal'] = df.target
    path = tmp_path/'data.csv'
    results = []
    for iteration in range(2):
        if iteration:
            df.loc[df.index >= 3200, 'target'] = 1-df.loc[df.index >= 3200, 'target']
        df.to_csv(path, index=False)
        write_json_atomic(str(path)+'.metadata.json', dict(specification('R_100'), dataset_sha256=fingerprint(path)))
        results.append(train_model.train(path, tmp_path/f'model{iteration}.pkl'))
    assert results[0]['best_model'] == results[1]['best_model']
    assert results[0]['walk_forward'] == results[1]['walk_forward']
    assert results[0]['accuracy'] != results[1]['accuracy']


def test_xgboost_training_writes_json_report_with_missing_marker(tmp_path, monkeypatch):
    import train_model
    from xgboost import XGBClassifier
    from model_contract import specification, fingerprint, write_json_atomic
    monkeypatch.setattr(train_model, 'training_candidates', lambda search_trials=4: [('XGBoost',
        XGBClassifier(n_estimators=4, max_depth=2, n_jobs=1, random_state=42))])
    df = frame(4000)
    df['target'] = np.arange(len(df)) % 2
    df['ema9'], df['technical_signal'] = df.target, df.target
    path = tmp_path/'data.csv'
    df.to_csv(path, index=False)
    write_json_atomic(str(path)+'.metadata.json', dict(specification('R_100'), dataset_sha256=fingerprint(path)))
    result = train_model.train(path, tmp_path/'xgb.pkl')
    report = json.loads(next((tmp_path/'training_reports').glob('*.json')).read_text())
    assert report['hyperparameters']['missing'] == 'nan'
    assert result['best_model'] == 'XGBoost'


def test_payout_telemetry_failure_does_not_discard_live_tick(tmp_path, monkeypatch):
    import pipeline
    from market_data import ensure_header, read_records
    path = tmp_path/'ticks.csv'
    ensure_header(path)
    monkeypatch.setattr(pipeline, 'TICKS_CSV', str(path))
    collector = pipeline._CollectorThread('R_100')
    collector._payout_recorder.sample = Mock(side_effect=OSError('offline telemetry failure'))
    collector._on_message(Mock(), json.dumps({'tick': {'epoch': 1_800_000_000, 'quote': 100, 'symbol': 'R_100'}}))
    rows, rejected = read_records(path)
    assert not rejected and len(rows) == 1 and rows[0]['price'] == 100


@pytest.mark.parametrize('total,wins,expected', [(5,3,True),(30,14,True),(100,45,True),(200,90,True),(100,44,False),(0,0,False)])
def test_fixed_forty_five_percent_gate_for_any_sample(monkeypatch,total,wins,expected):
    import training_evaluation as evaluation
    df = frame(10)
    df['target'] = [0,1]*5
    class RankedModel:
        classes_ = np.array([0,1])
        def predict_proba(self,X):
            return np.array([[.9,.1],[.1,.9]]*5)
    result = dict(trades=total,wins=wins,win_rate=wins/total if total else 0,
                  win_rate_lower95=.1,break_even=.625)
    monkeypatch.setattr(evaluation,'simulate',lambda *a,**kw:dict(result))
    assert evaluation.evaluate(RankedModel(),df,.60)['eligible'] is expected


def test_low_auc_is_warning_not_approval_blocker(monkeypatch):
    import training_evaluation as evaluation
    df = frame(10)
    df['target'] = [0, 1] * 5

    class InvertedModel:
        classes_ = np.array([0, 1])

        def predict_proba(self, X):
            return np.array([[.1, .9], [.9, .1]] * 5)

    result = dict(trades=100, wins=45, win_rate=.45,
                  win_rate_lower95=.1, break_even=.625)
    monkeypatch.setattr(evaluation, 'simulate', lambda *a, **kw: dict(result))
    evaluated = evaluation.evaluate(InvertedModel(), df, .60)
    assert evaluated['eligible'] is True
    assert evaluated['rejection_reasons'] == []
    assert any('AUC' in warning for warning in evaluated['warnings'])


@pytest.mark.parametrize('ratio,blocked',[(.5999,True),(.60,False),(.85,False)])
@pytest.mark.parametrize('mode',['ai_only','technical_only','combined'])
def test_training_simulation_payout_floor(ratio,blocked,mode):
    from training_evaluation import simulate
    df=frame(100)
    result=simulate(df,np.ones(len(df),dtype=int),np.full(len(df),.9),mode=mode,payout=ratio)
    assert (result['trades']==0) is blocked


@pytest.mark.parametrize('probability,ratio,accepted', [
    (.60, .68, True), (.625, .60, True), (.6249, .60, False), (.56, .70, False),
])
def test_simulation_nonnegative_return_matches_executor(probability, ratio, accepted):
    from training_evaluation import simulate
    from model_contract import profitable_proposal
    df = frame(100)
    result = simulate(df, np.ones(len(df), dtype=int), np.full(len(df), probability),
                      mode='ai_only', payout=ratio)
    assert (result['trades'] > 0) is accepted
    assert profitable_proposal(100, 100 * (1 + ratio), probability) is accepted
