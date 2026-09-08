"""Treino temporal com calibração separada e teste final intocado.

Produz um bundle de direção/vencimento. Não promove TFT ou modelos de duração
legados: suas probabilidades não foram calibradas para este contrato.
"""
import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

import config
from feature_engine import FEATURES
from model_contract import compatible, fingerprint, write_json_atomic, MIN_EXPECTED_RETURN, usable


class CalibratedDirectionModel:
    """Calibração multinomial em período posterior, sem refit no teste."""
    def __init__(self, estimator):
        self.estimator = estimator
        self.encoder = LabelEncoder()

    def fit(self, X, y):
        self.estimator.fit(X, self.encoder.fit_transform(y))
        return self

    def calibrate(self, X, y):
        self.calibrator = LogisticRegression(max_iter = 1000)
        self.calibrator.fit(self.estimator.predict_proba(X), y)
        self.classes_ = self.calibrator.classes_
        return self

    def predict_proba(self, X):
        return self.calibrator.predict_proba(self.estimator.predict_proba(X))

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


def _metrics(model, df, payout = .85):
    if not math.isfinite(payout) or payout <= 0:
        raise ValueError("payout de avaliação inválido")
    prob = model.predict_proba(df[FEATURES].values)
    by_class = {int(c): prob[: , i] for i, c in enumerate(model.classes_)}
    up = by_class.get(1, np.zeros(len(df)))
    down = by_class.get(0, np.zeros(len(df)))
    direction = np.where(up > down, 1, 0)
    confidence = np.maximum(up, down)
    predictions = model.classes_[prob.argmax(axis = 1)]
    binary_up = (df.target.values == 1).astype(int)
    auc = roc_auc_score(binary_up, up) if len(np.unique(binary_up)) > 1 else .5
    from backtest import BacktestSettings, HistoricalRiskManager
    settings=BacktestSettings(payout_ratio=payout)
    risk=HistoricalRiskManager(settings)
    wins, total, pnl = 0,0,0.
    pending=None
    candles_since_entry=config.ENTRY_CANDLE_INTERVAL
    for i,row in enumerate(df.itertuples(index=False)):
        if pending is not None and row.entry_epoch>=pending[0]:
            risk.settle(*pending)
            pending=None
        if pending is not None or not risk.can_trade(row.entry_epoch)[0]:
            continue
        candles_since_entry+=1
        interval=config.ENTRY_CANDLE_INTERVAL+(1 if risk.consecutive_losses>=2 else 0)
        if candles_since_entry<interval:
            continue
        technical=int(row.technical_signal)
        score=config.AI_TECH_WEIGHT*min(1.,abs(row.tech_score))+config.AI_MODEL_WEIGHT*confidence[i]
        if (technical not in (0,1) or technical!=direction[i]
                or confidence[i]<config.AI_CONFIDENCE_MIN or score<config.AI_SCORE_MIN
                or confidence[i]*(1+payout)-1<MIN_EXPECTED_RETURN):
            continue
        total+=1
        win=int(row.target)==direction[i]
        wins+=int(win)
        unit_profit=payout if win else -1
        pnl+=unit_profit
        pending=(row.expiry_epoch,risk.get_stake()*unit_profit)
        candles_since_entry=0
    if pending is not None:
        risk.settle(*pending)
    rate = wins / total if total else 0.
    z = 1.96
    lower = ((rate + z * z / (2 * total) - z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)))
             / (1 + z * z / total)) if total else 0.
    return {'auc': float(auc), 'accuracy': float(accuracy_score(df.target, predictions)),
            'f1': float(f1_score(df.target, predictions, average = 'macro', zero_division = 0)),
            'n_test': len(df), 'trades': total, 'wins': wins, 'win_rate': rate, 'win_rate_lower95': lower,
            'unit_profit': pnl, 'reference_net_payout': payout,
            'evaluation_start': int(df.entry_epoch.iloc[0]), 'evaluation_end': int(df.expiry_epoch.iloc[-1]),
            'evaluation_kind': 'deployed_strategy_reference_payout', 'final_balance':risk.balance,
            'eligible': bool(total >= 30 and lower > 1 / (1 + payout) and auc > .5)}


def train(dataset_path, output_path, test_ratio = .2, gap_ratio = .02,
          tft_output_path = None, duration_output_path = None):
    metadata = json.loads(Path(str(dataset_path) + '.metadata.json').read_text())
    if not compatible(metadata, config.get_active_symbol()) or metadata.get('dataset_sha256') != fingerprint(dataset_path):
        raise ValueError('dataset incompatível, alterado ou sem proveniência')
    df = pd.read_csv(dataset_path)
    required = FEATURES + ['symbol', 'signal_epoch', 'entry_epoch', 'expiry_epoch', 'feature_start_epoch', 'target', 'technical_signal', 'tech_score']
    if any(c not in df for c in required) or not np.isfinite(df[FEATURES].values).all():
        raise ValueError('dataset inválido')
    if (set(df.symbol) != {metadata['symbol']} or df.signal_epoch.duplicated().any()
            or not df.signal_epoch.is_monotonic_increasing
            or not set(df.target).issubset({0, 1, 2})):
        raise ValueError('símbolo, target ou cronologia inválidos')
    fit, calibration, validation, test = temporal_partitions(df, test_ratio)
    estimators = [('RandomForest', RandomForestClassifier(n_estimators = 300, max_depth = 10, min_samples_leaf = 10, random_state = 42, n_jobs = 1))]
    from xgboost import XGBClassifier
    estimators.append(('XGBoost', XGBClassifier(n_estimators = 300, max_depth = 5, learning_rate = .05, random_state = 42, n_jobs = 1)))
    candidates = []
    for name, estimator in estimators:
        model = CalibratedDirectionModel(estimator).fit(fit[FEATURES].values, fit.target.values)
        model.calibrate(calibration[FEATURES].values, calibration.target.values)
        metrics = _metrics(model, validation)
        candidates.append((metrics['auc'], name, model))
    _, name, model = max(candidates, key = lambda x: x[0])
    final = _metrics(model, test)
    final.update(best_model = name, n_train = len(fit), stage = 'final', timestamp = datetime.now(timezone.utc).isoformat(),
                 symbol = metadata['symbol'], dataset_sha256 = metadata['dataset_sha256'])
    # O baseline é o artefato ativo avaliado nas MESMAS linhas, nunca a última métrica global.
    final['champion_metrics'] = None
    if Path(config.AI_MODEL_PATH).exists() and Path(config.AI_MODEL_PATH).resolve() != Path(output_path).resolve():
        try:
            champion = joblib.load(config.AI_MODEL_PATH)
            cm = champion.get('metadata', {})
            if usable(cm, metadata['symbol']) and cm.get('last_seen_epoch', float('inf')) < int(test.feature_start_epoch.iloc[0]):
                final['champion_metrics'] = _metrics(champion['model'], test)
        except Exception:
            pass
    metadata.update(approved = final['eligible'], calibration = 'multinomial_disjoint_temporal',
                    trained_at = final['timestamp'], last_seen_epoch = int(df.expiry_epoch.iloc[-1]), metrics = final,
                    model_name = name, dependencies = {})
    import sklearn, xgboost
    metadata['dependencies'] = {'sklearn': sklearn.__version__, 'xgboost': xgboost.__version__}
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


def main():
    p = argparse.ArgumentParser(description = __doc__)
    p.add_argument('--input', default = config.DATASET_CSV)
    p.add_argument('--output', default = config.AI_MODEL_PATH + '.candidate')
    p.add_argument('--test-ratio', type = float, default = .2)
    p.add_argument('--symbol', default = None)
    a = p.parse_args()
    if a.symbol:
        config.set_active_symbol(a.symbol)
    from train_model import train as train_bundle
    print(json.dumps(train_bundle(a.input, a.output, a.test_ratio), indent = 2))


if __name__ == '__main__':
    main()
