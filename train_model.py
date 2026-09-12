"""Treino temporal com calibração separada e teste final intocado.

Produz um bundle de direção/vencimento. Não promove TFT ou modelos de duração
legados: suas probabilidades não foram calibradas para este contrato.
"""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline

import config
from feature_engine import FEATURES
from model_contract import compatible, fingerprint, write_json_atomic, usable, load_validated_bundle


class CalibratedDirectionModel:
    """Calibração temporal multinomial/temperatura e controle bruto diagnóstico."""
    def __init__(self, estimator, method='multinomial'):
        if method not in ('multinomial', 'temperature', 'raw'):
            raise ValueError('método de calibração inválido')
        self.estimator = estimator
        self.encoder = LabelEncoder()
        self.method = method

    def fit(self, X, y):
        self.estimator.fit(X, self.encoder.fit_transform(y))
        self.classes_ = self.encoder.classes_
        return self

    def calibrate(self, X, y):
        if self.method == 'raw':
            return self
        if self.method == 'temperature':
            # Classes raras que só aparecem na calibração continuam representadas.
            self.classes_ = np.union1d(self.encoder.classes_, np.unique(y))
            logits = np.log(np.clip(self._raw_aligned(X), 1e-12, 1))
            labels = np.searchsorted(self.classes_, y)
            def loss(log_temperature):
                prob = softmax(logits / np.exp(log_temperature), axis=1)
                return float(-np.log(np.clip(prob[np.arange(len(y)), labels], 1e-15, 1)).mean())
            result = minimize_scalar(loss, bounds=(-3., 3.), method='bounded')
            if not result.success:
                raise ValueError('otimização da temperatura não convergiu')
            self.temperature_ = float(np.exp(result.x)) if result.fun < loss(0.) else 1.
            return self
        self.calibrator = LogisticRegression(max_iter = 1000)
        self.calibrator.fit(self.estimator.predict_proba(X), y)
        self.classes_ = self.calibrator.classes_
        return self

    def predict_proba(self, X):
        method = getattr(self, 'method', 'multinomial')  # bundles anteriores
        if method == 'raw':
            return self._raw_aligned(X)
        if method == 'temperature':
            return softmax(np.log(np.clip(self._raw_aligned(X), 1e-12, 1)) / self.temperature_, axis=1)
        return self.calibrator.predict_proba(self.estimator.predict_proba(X))

    def _raw_aligned(self, X):
        raw = self.estimator.predict_proba(X)
        result = np.zeros((len(X), len(self.classes_)))
        result[:, np.searchsorted(self.classes_, self.encoder.classes_)] = raw
        return result

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis = 1)]


def _build_sequence_dataset(X, seq_len, epochs = None, timeframe_sec = 60):
    """Sequência termina na linha do label, sem deslocamento de uma observação.

    Para treinamento sequencial, timestamps são obrigatórios e lacunas rejeitadas.
    """
    if epochs is None:
        raise ValueError('timestamps obrigatórios para verificar continuidade das sequências')
    if len(epochs) != len(X) or np.any(np.diff(epochs) != timeframe_sec):
        raise ValueError('sequência descontínua')
    if len(X) < seq_len:
        return np.empty((0, seq_len, X.shape[1]), dtype = np.float32)
    return np.asarray([X[i - seq_len + 1: i + 1] for i in range(seq_len - 1, len(X))], dtype = np.float32)


def temporal_partitions(df, test_ratio = .2):
    if not .1 <= test_ratio <= .3:
        raise ValueError('test_ratio deve estar entre 0.1 e 0.3')
    n = len(df)
    boundaries = [0, int(n * (.7 - test_ratio)), int(n * (.85 - test_ratio)), int(n * (1 - test_ratio)), n]
    parts = []
    for start, end in zip(boundaries, boundaries[1:]):
        part = df.iloc[start: end]
        if end < n:
            # Nem rótulos nem janelas de features atravessam a próxima partição.
            part = part[part.expiry_epoch < df.iloc[end].feature_start_epoch]
        if len(part) < 30 or part.target.nunique() < 2:
            raise ValueError('amostra temporal insuficiente após embargo; colete mais ticks identificados')
        parts.append(part)
    return parts


def _metrics(model, df, payout=.85, *, features=None, quotes=None):
    from training_evaluation import evaluate
    return evaluate(model, df, payout, features=features, quotes=quotes)


def walk_forward_partitions(development, folds=3):
    """Expanding fit → calibração → avaliação; nenhum acesso ao holdout final."""
    n = len(development)
    edges = np.linspace(int(n*.55), n, folds+1, dtype=int)
    for start, end in zip(edges, edges[1:]):
        calibration_start = int(start*.8)
        fit = development.iloc[:calibration_start]
        calibration = development.iloc[calibration_start:start]
        evaluation = development.iloc[start:end]
        fit = fit[fit.expiry_epoch < development.iloc[calibration_start].feature_start_epoch]
        calibration = calibration[calibration.expiry_epoch < development.iloc[start].feature_start_epoch]
        if end < n:
            evaluation = evaluation[evaluation.expiry_epoch < development.iloc[end].feature_start_epoch]
        if any(len(part) < 30 or part.target.nunique() < 2 for part in (fit, calibration, evaluation)):
            raise ValueError('walk-forward insuficiente após embargo; colete mais contratos rotulados')
        yield fit, calibration, evaluation


def candidate_estimators():
    from xgboost import XGBClassifier
    return [('RandomForest', RandomForestClassifier(n_estimators=300, max_depth=10,
                min_samples_leaf=10, random_state=42, n_jobs=1)),
            ('XGBoost', XGBClassifier(n_estimators=300, max_depth=5, learning_rate=.05,
                random_state=42, n_jobs=1))]


def training_candidates(search_trials=4):
    """Orçamento fechado: dois baselines e até 12 configurações de árvores."""
    if isinstance(search_trials, bool) or not isinstance(search_trials, int) or not 0 <= search_trials <= 10:
        raise ValueError('search_trials deve ser inteiro entre 0 e 10')
    candidates = candidate_estimators()
    candidates += [('LogisticRegression', make_pipeline(StandardScaler(),
                    LogisticRegression(C=1., max_iter=2000, random_state=42))),
                   ('ClassFrequency', DummyClassifier(strategy='prior'))]
    rng = np.random.default_rng(42)
    trees = [(name, model) for name, model in candidates if name in ('RandomForest', 'XGBoost')]
    for i in range(search_trials):
        if not trees:
            break
        name, template = trees[i % len(trees)]
        estimator = clone(template)
        if name == 'RandomForest':
            params = dict(max_depth=int(rng.choice([3, 5, 8])),
                          min_samples_leaf=int(rng.choice([20, 40, 80])), max_features=.7)
        else:
            params = dict(max_depth=int(rng.integers(2, 6)),
                          min_child_weight=int(rng.choice([5, 10, 20, 30])),
                          subsample=float(rng.uniform(.7, 1)), colsample_bytree=float(rng.uniform(.7, 1)),
                          reg_lambda=float(rng.uniform(1, 20)))
        candidates.append((f'{name}_regularized_{i+1}', estimator.set_params(**params)))
    return candidates


def fit_calibration_partition(development):
    """Mesma receita 80/20 usada nas janelas internas; teste fica fora."""
    boundary = int(len(development)*.8)
    calibration = development.iloc[boundary:]
    fit = development.iloc[:boundary]
    fit = fit[fit.expiry_epoch < calibration.feature_start_epoch.iloc[0]]
    if any(len(part) < 30 or part.target.nunique() < 2 for part in (fit, calibration)):
        raise ValueError('ajuste/calibração insuficientes após embargo')
    return fit, calibration


def walk_forward_evaluate(development, features=None, *, candidates=None):
    from copy import copy
    features = FEATURES if features is None else features
    partitions = list(walk_forward_partitions(development))
    candidates = training_candidates() if candidates is None else candidates
    results = []
    for name, estimator in candidates:
        methods = ('raw',) if name == 'ClassFrequency' else ('raw', 'multinomial', 'temperature')
        windows = {method: [] for method in methods}
        for fit, calibration, evaluation in partitions:
            base = CalibratedDirectionModel(clone(estimator), 'raw').fit(fit[features].values, fit.target.values)
            for method in methods:
                model = copy(base)  # calibrações compartilham exatamente o mesmo estimador ajustado
                model.method = method
                model.calibrate(calibration[features].values, calibration.target.values)
                metrics = _metrics(model, evaluation, payout=.70, features=features)
                metrics['partition'] = {
                    'fit_last_expiry': int(fit.expiry_epoch.max()),
                    'calibration_first_feature': int(calibration.feature_start_epoch.min()),
                    'calibration_last_expiry': int(calibration.expiry_epoch.max()),
                    'evaluation_first_feature': int(evaluation.feature_start_epoch.min()),
                }
                windows[method].append(metrics)
        for method, folds in windows.items():
            trades = sum(w['trades'] for w in folds)
            results.append({'name': name, 'calibration_method': method,
                   'recipe_id': name + '/' + method, 'diagnostic_only': method == 'raw',
                   'folds': folds, 'trades': trades,
                   'empty_folds': sum(w['trades'] == 0 for w in folds),
                   'worst_expectancy': min(w['expectancy'] for w in folds),
                   'mean_expectancy': float(np.mean([w['expectancy'] for w in folds])),
                   'max_drawdown_pct': max(w['max_drawdown_pct'] for w in folds),
                   'mean_log_loss': float(np.mean([w['calibration']['log_loss'] for w in folds])),
                   'mean_brier': float(np.mean([w['calibration']['multiclass_brier'] for w in folds])),
                   'mean_auc': float(np.mean([w['auc'] for w in folds])),
                   'reference_net_payout': .70})
    baseline = next((r for r in results if r['name'] == 'ClassFrequency'), None)
    for result in results:
        result['log_loss_gain_vs_prior'] = baseline['mean_log_loss']-result['mean_log_loss'] if baseline else 0.
        result['brier_gain_vs_prior'] = baseline['mean_brier']-result['mean_brier'] if baseline else 0.
    return results


def choose_candidate(results):
    return max((r for r in results if not r.get('diagnostic_only', False)), key=selection_key)


def selection_key(result):
    return (result.get('log_loss_gain_vs_prior', 1.) > 0,
            result['trades'] >= 30 and result.get('empty_folds', 0) == 0,
            result['worst_expectancy'], result['mean_expectancy'],
            -result['max_drawdown_pct'], -result['mean_log_loss'], result['mean_auc'])


def train(dataset_path, output_path, test_ratio = .2, gap_ratio = .02,
          tft_output_path = None, duration_output_path = None, payout_quotes_path = None, search_trials=4,
          compare_champion=True):
    metadata = json.loads(Path(str(dataset_path) + '.metadata.json').read_text())
    if not compatible(metadata, config.get_active_symbol()) or metadata.get('dataset_sha256') != fingerprint(dataset_path):
        raise ValueError('dataset incompatível, alterado ou sem proveniência')
    df = pd.read_csv(dataset_path)
    if metadata['dataset_sha256'] != fingerprint(dataset_path):
        raise ValueError('dataset mudou durante a leitura; use snapshot estável')
    required = FEATURES + ['symbol', 'signal_epoch', 'entry_epoch', 'expiry_epoch', 'feature_start_epoch', 'target', 'technical_signal', 'tech_score']
    if any(c not in df for c in required) or not np.isfinite(df[FEATURES].values).all():
        raise ValueError('dataset inválido')
    if (set(df.symbol) != {metadata['symbol']} or df.signal_epoch.duplicated().any()
            or not df.signal_epoch.is_monotonic_increasing
            or not set(df.target).issubset({0, 1, 2})
            or not np.isfinite(df[['signal_epoch', 'entry_epoch', 'expiry_epoch', 'feature_start_epoch', 'tech_score']].values).all()
            or not ((df.feature_start_epoch <= df.signal_epoch) & (df.signal_epoch <= df.entry_epoch)
                    & (df.entry_epoch < df.expiry_epoch)).all()):
        raise ValueError('símbolo, target ou cronologia inválidos')
    fit, calibration, validation, test = temporal_partitions(df, test_ratio)
    legacy_fit, legacy_calibration = fit, calibration
    # O último período fica fora de toda seleção, incluindo comparação de features.
    development = df[df.expiry_epoch < int(test.feature_start_epoch.iloc[0])]
    candidates = training_candidates(search_trials)
    walk_forward = walk_forward_evaluate(development, candidates=candidates)
    winner = choose_candidate(walk_forward)
    legacy_recipes = [r for r in walk_forward if r['name'] in ('RandomForest', 'XGBoost')
                      and r['calibration_method'] == 'multinomial']
    legacy_winner = max(legacy_recipes, key=lambda r: (
        r['trades'] >= 30, r['worst_expectancy'], r['mean_expectancy'],
        -r['max_drawdown_pct'], -r['mean_log_loss'], r['mean_auc'])) if legacy_recipes else None
    name = winner['name']
    estimator = dict(candidates)[name]
    fit, calibration = fit_calibration_partition(development)
    method = winner['calibration_method']
    model = CalibratedDirectionModel(estimator, method).fit(fit[FEATURES].values, fit.target.values)
    model.calibrate(calibration[FEATURES].values, calibration.target.values)
    final = _metrics(model, test)
    final['walk_forward'] = walk_forward
    final['selection_method'] = 'walk_forward_baselines_calibration_v2'
    final['selected_recipe'] = winner['recipe_id']
    final['calibration_method'] = method
    final['search_trials'] = search_trials
    final['holdout_used_for_selection'] = False
    final['fit_last_expiry'] = int(fit.expiry_epoch.max())
    final['calibration_first_feature'] = int(calibration.feature_start_epoch.min())
    final['calibration_last_expiry'] = int(calibration.expiry_epoch.max())
    final['test_first_feature'] = int(test.feature_start_epoch.min())
    raw_model = CalibratedDirectionModel(estimator, 'raw')
    raw_model.encoder = model.encoder
    raw_model.classes_ = model.encoder.classes_
    final['uncalibrated_metrics'] = _metrics(raw_model, test)
    from training_evaluation import decision_curve
    final['decision_curve'] = decision_curve(model, test)
    if legacy_winner:
        baseline = CalibratedDirectionModel(clone(dict(candidates)[legacy_winner['name']]))
        baseline.fit(legacy_fit[FEATURES].values, legacy_fit.target.values)
        baseline.calibrate(legacy_calibration[FEATURES].values, legacy_calibration.target.values)
        final['legacy_baseline'] = {'name': legacy_winner['name'],
                                    'metrics': _metrics(baseline, test),
                                    'note': 'Receita anterior, selecionada somente no development; mesmo teste final.'}
    conservative = _metrics(model, test, payout=.70)
    final['payout_scenarios'] = {'net_70pct': conservative, 'net_60pct': _metrics(model, test, payout=.60)}
    final['selection_warnings'] = []
    if not conservative['eligible']:
        final['selection_warnings'].append('cenário conservador de payout líquido 70% não aprovado')
    if winner['log_loss_gain_vs_prior'] <= 0:
        final['selection_warnings'].append('sem ganho de log loss sobre frequência de classes no development')
    if winner['trades'] < 30 or winner['empty_folds'] or winner['worst_expectancy'] <= 0:
        final['selection_warnings'].append('walk-forward sem amostra ou consistência positiva em todas as janelas')
    if payout_quotes_path is not None and Path(payout_quotes_path).exists():
        from payout_quotes import QuoteHistory
        quotes = QuoteHistory(payout_quotes_path, currency=config.CURRENCY)
        final['observed_payout_evaluation'] = _metrics(model, test, quotes=quotes)
        final['observed_payout_evaluation']['quotes_sha256'] = quotes.source_sha256
        final['observed_payout_evaluation']['note'] = (
            'Somente cotações anteriores, até 900s; sem preencher lacunas. '
            'Razão observada na stake cotada, sem modelar efeito da stake simulada.')
    final.update(best_model = name, n_train = len(fit), stage = 'final', timestamp = datetime.now(timezone.utc).isoformat(),
                 symbol = metadata['symbol'], dataset_sha256 = metadata['dataset_sha256'])
    final['training_manifest'] = {
        'seed': 42, 'feature_schema': list(FEATURES),
        'code_sha256': {name: fingerprint(Path(__file__).parent / name) for name in
                       ('train_model.py', 'training_evaluation.py', 'backtest.py', 'model_contract.py')},
        'candidate_parameters': {name: json_parameters(estimator) for name, estimator in candidates},
        'calibration_methods': ['raw_diagnostic_only', 'multinomial', 'temperature'],
        'evaluation_risk_timezone': final['evaluation_risk_timezone'],
    }
    # O baseline é o artefato ativo avaliado nas MESMAS linhas, nunca a última métrica global.
    final['champion_metrics'] = None
    if compare_champion and Path(config.AI_MODEL_PATH).exists() and Path(config.AI_MODEL_PATH).resolve() != Path(output_path).resolve():
        try:
            champion, _ = load_validated_bundle(config.AI_MODEL_PATH, metadata['symbol'])
            cm = champion.get('metadata', {})
            if usable(cm, metadata['symbol']) and cm.get('last_seen_epoch', float('inf')) < int(test.feature_start_epoch.iloc[0]):
                final['champion_metrics'] = _metrics(champion['model'], test)
        except Exception:
            pass
    metadata.update(approved = final['eligible'], calibration = method + '_disjoint_temporal',
                    evaluation_risk_timezone = final['evaluation_risk_timezone'],
                    trained_at = final['timestamp'], last_seen_epoch = int(df.expiry_epoch.iloc[-1]), metrics = final,
                    model_name = name, dependencies = {})
    import sklearn, xgboost
    metadata['dependencies'] = {'sklearn': sklearn.__version__, 'xgboost': xgboost.__version__}
    report_dir = Path(output_path).resolve().parent / 'training_reports'
    report_dir.mkdir(exist_ok=True)
    report_path = report_dir / (metadata['dataset_sha256'] + '-evaluation.json')
    write_json_atomic(report_path, {'metadata': metadata, 'metrics': final,
                                   'hyperparameters': json_parameters(estimator)})
    # Persistência de um único bundle evita combinações parciais de três versões.
    joblib.dump({'model': model, 'features': FEATURES, 'name': name, 'metadata': metadata}, output_path)
    write_json_atomic(str(output_path) + '.metadata.json', dict(metadata, artifact_sha256 = fingerprint(output_path)))
    metrics_path = os.path.join(os.path.dirname(os.path.abspath(output_path)), 'model_metrics.json')
    try:
        history = json.loads(Path(metrics_path).read_text())
        if not isinstance(history, list):
            history = []
    except (OSError, ValueError):
        history = []
    write_json_atomic(metrics_path, history + [dict(final, model_path = str(output_path))])
    return final


def json_parameters(estimator):
    """Pipelines e marcadores NaN também precisam de representação auditável."""
    result = {}
    for key, value in estimator.get_params().items():
        try:
            json.dumps(value, allow_nan=False)
            result[key] = value
        except (ValueError, TypeError):
            result[key] = repr(value)
    return result


def main():
    p = argparse.ArgumentParser(description = __doc__)
    p.add_argument('--input', default = config.DATASET_CSV)
    p.add_argument('--output', default = config.AI_MODEL_PATH + '.candidate')
    p.add_argument('--test-ratio', type = float, default = .2)
    p.add_argument('--symbol', default = None)
    p.add_argument('--payout-quotes', default=None)
    p.add_argument('--search-trials', type=int, default=4, help='configurações adicionais, de 0 a 10')
    a = p.parse_args()
    if a.symbol:
        config.set_active_symbol(a.symbol)
    from train_model import train as train_bundle
    print(json.dumps(train_bundle(a.input, a.output, a.test_ratio, payout_quotes_path=a.payout_quotes, search_trials=a.search_trials), indent = 2))


if __name__ == '__main__':
    main()
