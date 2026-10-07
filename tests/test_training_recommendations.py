"""Calibração, baselines, isolamento e avaliação temporal das recomendações."""
import json
import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier

import config
import train_model as tm
from model_contract import specification, fingerprint, write_json_atomic
from training_evaluation import decision_curve
from training_experiments import compare_ablation, compare_training
from tests.test_training_evaluation import frame, FixedModel


def small_candidates(search_trials=0):
    """Substitui training_candidates por dois modelos pequenos e rápidos."""
    return [('RandomForest', RandomForestClassifier(n_estimators=4, max_depth=2, random_state=42)),
            ('ClassFrequency', DummyClassifier(strategy='prior'))]


def dataset(tmp_path, n=2400):
    """Grava um dataset sintético com metadados e retorna (DataFrame, caminho)."""
    df = frame(n)
    df['target'] = np.arange(n) % 2
    df['ema9'] = df.target
    df['technical_signal'] = df.target
    path = tmp_path/'source.csv'
    df.to_csv(path, index=False)
    write_json_atomic(str(path)+'.metadata.json', dict(specification('R_100'), dataset_sha256=fingerprint(path)))
    return df, path


def test_prior_uses_only_training_labels():
    """Garante que o prior do modelo calibrado vem apenas dos rótulos de treino."""
    X = np.zeros((10, 2))
    model = tm.CalibratedDirectionModel(DummyClassifier(strategy='prior'), 'raw').fit(X, [0]*7+[1]*3)
    model.calibrate(X, [1]*10)
    assert model.predict_proba(X[:1])[0] == pytest.approx([.7, .3])


def test_temperature_preserves_ranking_and_serializes(tmp_path):
    """Garante que a calibração por temperatura preserva o ranking das classes e gera probabilidades válidas."""
    import joblib
    X = np.arange(40).reshape(-1, 1)
    y = np.array([0]*20+[1]*20)
    model = tm.CalibratedDirectionModel(RandomForestClassifier(n_estimators=4, random_state=42), 'temperature').fit(X, y)
    # Calibração posterior intencionalmente ruidosa.
    calibration_y = np.array([0, 1]*20)
    model.calibrate(X, calibration_y)
    prob = model.predict_proba(X)
    raw = model._raw_aligned(X)
    assert np.array_equal(prob.argmax(axis=1), raw.argmax(axis=1))
    assert np.allclose(prob.sum(axis=1), 1)
    assert np.isfinite(prob).all() and model.temperature_ > 0
    raw_loss = -np.log(np.clip(raw[np.arange(40), calibration_y], 1e-12, 1)).mean()
    loss = -np.log(prob[np.arange(40), calibration_y]).mean()
    assert loss < raw_loss
    path = tmp_path/'model.pkl'
    joblib.dump(model, path)
    assert np.array_equal(joblib.load(path).predict_proba(X), prob)


def test_temperature_keeps_ties_seen_only_in_calibration():
    """Garante que classes vistas só na calibração permanecem com probabilidade positiva."""
    X = np.arange(20).reshape(-1, 1)
    model = tm.CalibratedDirectionModel(DummyClassifier(strategy='prior'), 'temperature').fit(X, [0, 1]*10)
    model.calibrate(X, [0, 1, 2, 1]*5)
    assert list(model.classes_) == [0, 1, 2]
    prob = model.predict_proba(X)
    assert (prob[:, 2] > 0).all()
    assert np.allclose(prob.sum(axis=1), 1)


def test_old_multinomial_bundles_remain_readable():
    """Garante que bundles antigos sem o atributo method continuam gerando as mesmas probabilidades."""
    X = np.arange(20).reshape(-1, 1)
    model = tm.CalibratedDirectionModel(DummyClassifier(strategy='prior')).fit(X, [0, 1]*10)
    model.calibrate(X, [0, 1]*10)
    expected = model.predict_proba(X)
    del model.method
    assert np.array_equal(model.predict_proba(X), expected)


def test_search_budget_is_bounded_and_deterministic():
    """Garante que training_candidates é determinístico, limitado a 12 modelos e rejeita search_trials inválido."""
    first, second = tm.training_candidates(4), tm.training_candidates(4)
    assert [n for n, _ in first] == [n for n, _ in second]
    assert len(first) == 12
    assert [tm.json_parameters(m) for _, m in first] == [tm.json_parameters(m) for _, m in second]
    for invalid in (-1, 13, 1.5, True):
        with pytest.raises(ValueError, match='search_trials'):
            tm.training_candidates(invalid)


def test_diagnostic_raw_recipe_cannot_win_selection():
    """Garante que receita diagnóstica bruta não é escolhida por choose_candidate."""
    base = dict(trades=40, worst_expectancy=.1, mean_expectancy=.2,
                max_drawdown_pct=2, mean_log_loss=.6, mean_auc=.58)
    raw = dict(base, diagnostic_only=True, name='raw', worst_expectancy=100)
    calibrated = dict(base, diagnostic_only=False, name='calibrated')
    assert tm.choose_candidate([raw, calibrated]) is calibrated


def test_decision_curve_keeps_floors_and_no_trade_precision_is_undefined():
    """Garante que a curva de decisão respeita os pisos de confiança e deixa a precisão indefinida sem operações."""
    before = config.AI_CONFIDENCE_MIN
    df = frame()
    curve = decision_curve(FixedModel(), df)
    assert config.AI_CONFIDENCE_MIN == before
    assert len(curve) == 12
    assert all(r['confidence_floor'] >= before for r in curve)
    assert curve[0]['economic_probability_floor'] == pytest.approx(.625)
    df['technical_signal'] = -1
    assert all(r['precision'] is None and r['coverage'] == 0 for r in decision_curve(FixedModel(), df))


def test_offline_training_preserves_source_and_operational_artifacts(tmp_path, monkeypatch):
    """Garante que o treino offline não altera a fonte nem o modelo operacional nem o carrega."""
    monkeypatch.setattr(tm, 'training_candidates', small_candidates)
    df, source = dataset(tmp_path)
    original = source.read_bytes()
    champion = tmp_path/'model.pkl'
    champion.write_bytes(b'operational sentinel')
    monkeypatch.setattr(config, 'AI_MODEL_PATH', str(champion))
    def forbidden(*args, **kwargs):
        """Falha o teste se o experimento tentar carregar o modelo campeão."""
        raise AssertionError('experimento tentou carregar champion')
    monkeypatch.setattr(tm, 'load_validated_bundle', forbidden)
    result = compare_training(source, tmp_path/'experiment', search_trials=0)
    assert source.read_bytes() == original
    assert champion.read_bytes() == b'operational sentinel'
    assert result['experiment_only'] and not result['promoted']
    assert not result['holdout_used_for_selection']
    assert result['fit_last_expiry'] < result['calibration_first_feature']
    assert result['calibration_last_expiry'] < result['test_first_feature']
    assert result['evaluation_risk_timezone'] == 'UTC'
    assert result['legacy_baseline']['metrics']['evaluation_start'] == result['evaluation_start']
    assert result['legacy_baseline']['metrics']['n_test'] == result['n_test']
    for method in ('raw', 'multinomial', 'temperature'):
        assert any(r['calibration_method'] == method for r in result['walk_forward'])
    json.dumps(result, allow_nan=False)
    saved = json.loads((tmp_path/'experiment/model.candidate.metadata.json').read_text())
    assert saved['evaluation_risk_timezone'] == result['evaluation_risk_timezone']
    with pytest.raises(FileExistsError):
        compare_training(source, tmp_path/'experiment')


def test_holdout_changes_do_not_change_recipe_or_calibration(tmp_path, monkeypatch):
    """Garante que alterar rótulos do holdout não muda receita nem walk-forward, apenas a acurácia."""
    monkeypatch.setattr(tm, 'training_candidates', small_candidates)
    df, source = dataset(tmp_path)
    a = compare_training(source, tmp_path/'first', search_trials=0)
    df.loc[df.index >= int(len(df)*.8), 'target'] = 1-df.loc[df.index >= int(len(df)*.8), 'target']
    df.to_csv(source, index=False)
    write_json_atomic(str(source)+'.metadata.json', dict(specification('R_100'), dataset_sha256=fingerprint(source)))
    b = compare_training(source, tmp_path/'second', search_trials=0)
    assert a['selected_recipe'] == b['selected_recipe']
    assert a['walk_forward'] == b['walk_forward']
    assert a['accuracy'] != b['accuracy']


def test_feature_ablation_profiles_are_offline_and_development_only(tmp_path, monkeypatch):
    """Garante que a ablação de features gera os perfis esperados sem avaliar o holdout."""
    monkeypatch.setattr(tm, 'training_candidates', small_candidates)
    df, source = dataset(tmp_path)
    for column in ('rsi', 'adx', 'bb_position', 'return_1', 'return_3', 'return_5',
                   'volatility_10', 'volatility_20', 'high_low_5'):
        df[column] = df['target']
    df.to_csv(source, index=False)
    write_json_atomic(str(source)+'.metadata.json', dict(specification('R_100'), dataset_sha256=fingerprint(source)))
    report = compare_ablation(source, tmp_path/'ablation.json', search_trials=0)
    assert report['holdout_evaluated'] is False
    assert set(['baseline_27', 'without_price_action', 'price_action_only',
                'lean_technical', 'without_trend', 'without_volatility']).issubset(report['profiles'])
    assert report['profiles']['baseline_27']['n_features'] == len(tm.FEATURES)
    assert report['profiles']['without_price_action']['n_features'] < len(tm.FEATURES)
    assert report['profiles']['lean_technical']['n_features'] == 9
    assert report['development_last_expiry'] < report['holdout_feature_start']
    assert not (tmp_path/'model.candidate').exists()
    json.dumps(report, allow_nan=False)


def test_invalid_timestamps_rejected_before_fitting(tmp_path, monkeypatch):
    """Garante que timestamps fora de ordem cronológica fazem train levantar ValueError antes do ajuste."""
    df, source = dataset(tmp_path)
    df.loc[0, 'expiry_epoch'] = df.loc[0, 'entry_epoch']-1
    df.to_csv(source, index=False)
    write_json_atomic(str(source)+'.metadata.json', dict(specification('R_100'), dataset_sha256=fingerprint(source)))
    with pytest.raises(ValueError, match='cronologia'):
        tm.train(source, tmp_path/'candidate')
