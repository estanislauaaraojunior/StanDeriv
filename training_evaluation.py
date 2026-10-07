"""Diagnóstico causal de previsões e execução; não escreve estado operacional."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import math

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import config
from feature_engine import FEATURES
from model_contract import MIN_EXPECTED_RETURN, EVALUATION_RISK_TIMEZONE, meets_payout_floor


def wilson_lower(wins, total):
    """Limite inferior (95%) do intervalo de Wilson para a taxa de acerto; 0.0 sem amostras."""
    if not total:
        return 0.0
    rate, z = wins / total, 1.96
    return (rate + z*z/(2*total) - z*math.sqrt(rate*(1-rate)/total + z*z/(4*total*total))) / (1+z*z/total)


def probability_diagnostics(prob, classes, targets):
    """Classes ausentes continuam penalizadas; empate nunca vira PUT vencedora."""
    full = np.zeros((len(targets), 3))
    for index, label in enumerate(classes):
        full[:, int(label)] = prob[:, index]
    actual = np.eye(3)[targets]
    direction = np.where(full[:, 1] > full[:, 0], 1, 0)
    confidence = np.maximum(full[:, 0], full[:, 1])
    bins = []
    for low in np.arange(0, 1, .1):
        mask = (confidence >= low) & (confidence < low+.1 if low < .9 else confidence <= 1)
        if mask.any():
            bins.append({'from': round(float(low), 1), 'to': round(float(low+.1), 1),
                         'count': int(mask.sum()), 'mean_probability': float(confidence[mask].mean()),
                         'observed_win_rate': float((direction[mask] == targets[mask]).mean())})
    return full, direction, confidence, {
        'multiclass_brier': float(np.square(full-actual).sum(axis=1).mean()),
        'log_loss': float(-np.log(np.clip(full[np.arange(len(targets)), targets], 1e-15, 1)).mean()),
        'reliability_bins': bins, 'target_counts': dict(Counter(map(str, targets))),
        'prediction_counts': dict(Counter(map(str, classes[prob.argmax(axis=1)]))),
        'confidence_quantiles': {str(q): float(np.quantile(confidence, q)) for q in (0., .1, .5, .9, 1.)},
        'auc_by_direction': {
            label: float(roc_auc_score((targets == code).astype(int), full[:, code]))
            if len(np.unique(targets == code)) > 1 else None
            for code, label in ((0, 'SELL'), (1, 'BUY'))
        },
        'note': 'Bins usam todas as linhas sobrepostas; não são contratos independentes.'}


def simulate(df, direction, confidence, *, mode, payout, quotes=None, confidence_floor=None):
    """Simula entradas sequenciais (risco, cadência, payout e confiança) e retorna métricas e bloqueios por modo."""
    from backtest import BacktestSettings, HistoricalRiskManager
    floor = config.AI_CONFIDENCE_MIN if confidence_floor is None else confidence_floor
    if not math.isfinite(floor) or not config.AI_CONFIDENCE_MIN <= floor <= 1:
        raise ValueError('limiar experimental não pode reduzir o piso operacional')
    risk = HistoricalRiskManager(BacktestSettings(payout_ratio=payout, risk_timezone=EVALUATION_RISK_TIMEZONE))
    blockers, groups = Counter(), defaultdict(lambda: {'trades': 0, 'wins': 0, 'unit_profit': 0.0})
    pending = None
    candles_since_entry = config.ENTRY_CANDLE_INTERVAL
    total = wins = 0
    pnl = 0.0
    peak = risk.balance
    max_drawdown = 0.0
    break_even = []

    def settle():
        """Liquida o contrato pendente no gerenciador de risco e atualiza pico e drawdown."""
        nonlocal pending, peak, max_drawdown
        risk.settle(*pending)
        peak = max(peak, risk.balance)
        max_drawdown = max(max_drawdown, (peak-risk.balance)/peak if peak else 0)
        pending = None

    for i, row in enumerate(df.itertuples(index=False)):
        if pending is not None and row.entry_epoch >= pending[0]:
            settle()
        if pending is not None:
            blockers['active_contract'] += 1
            continue
        allowed, reason = risk.can_trade(row.entry_epoch)
        if not allowed:
            blockers['risk_' + reason] += 1
            continue
        candles_since_entry += 1
        interval = config.ENTRY_CANDLE_INTERVAL + (1 if risk.consecutive_losses >= 2 else 0)
        if candles_since_entry < interval:
            blockers['cadence'] += 1
            continue
        technical = int(row.technical_signal)
        selected = technical if mode == 'technical_only' else int(direction[i])
        if mode != 'ai_only' and technical not in (0, 1):
            blockers['no_technical_signal'] += 1
            continue
        if mode == 'combined' and technical != direction[i]:
            blockers['direction_disagreement'] += 1
            continue
        ratio = payout
        if quotes is not None:
            ratio = quotes.lookup(row.symbol, row.duration, row.duration_unit, selected, row.entry_epoch)
            if ratio is None:
                blockers['missing_observed_payout'] += 1
                continue
        if not meets_payout_floor(ratio):
            blockers['low_payout'] += 1
            continue
        if mode != 'technical_only':
            if confidence[i] <= floor:
                blockers['low_confidence'] += 1
                continue
            score = config.AI_TECH_WEIGHT*min(1., abs(row.tech_score)) + config.AI_MODEL_WEIGHT*confidence[i]
            if mode == 'combined' and score < config.AI_SCORE_MIN:
                blockers['low_combined_score'] += 1
                continue
            if confidence[i]*(1+ratio)-1 < MIN_EXPECTED_RETURN:
                blockers['insufficient_expected_return'] += 1
                continue
        total += 1
        win = int(row.target) == selected
        wins += int(win)
        profit = ratio if win else -1.0
        pnl += profit
        break_even.append(1/(1+ratio))
        hour = datetime.fromtimestamp(row.entry_epoch, timezone.utc).hour
        labels = ['direction:' + ('BUY' if selected == 1 else 'SELL'), f'hour_utc:{hour:02d}']
        if mode != 'technical_only':
            labels.append(f'confidence:{min(int(confidence[i]*10), 9)/10:.1f}')
        for label in labels:
            group = groups[label]
            group['trades'] += 1
            group['wins'] += int(win)
            group['unit_profit'] += profit
        pending = (row.expiry_epoch, risk.get_stake()*profit)
        candles_since_entry = 0
    if pending is not None:
        settle()
    return {'trades': total, 'wins': wins, 'win_rate': wins/total if total else 0.,
            'win_rate_lower95': wilson_lower(wins, total), 'unit_profit': pnl,
            'expectancy': pnl/total if total else 0., 'max_drawdown_pct': max_drawdown*100,
            'final_balance': risk.balance, 'blockers': dict(blockers), 'segments': dict(groups),
            'break_even': max(break_even) if break_even else 1/(1+payout),
            'payout_source': 'observed_asof_no_fallback' if quotes is not None else 'fixed_assumption',
            'evaluation_risk_timezone': EVALUATION_RISK_TIMEZONE,
            'coverage': total / len(df) if len(df) else 0.,
            'trades_per_day': total / max((df.expiry_epoch.iloc[-1]-df.entry_epoch.iloc[0])/86400, 1/86400),
            'note': 'Comparação diagnóstica; risco reinicia em cada janela e componente.'}


def evaluate(model, df, payout=.85, *, features=None, quotes=None):
    """Avalia o modelo no dataset (AUC, acurácia, F1, calibração e simulação por modo) e indica elegibilidade."""
    if not math.isfinite(payout) or payout <= 0 or df.empty:
        raise ValueError('payout ou dataset de avaliação inválido')
    features = FEATURES if features is None else features
    prob = np.asarray(model.predict_proba(df[features].values), dtype=float)
    classes = np.asarray(model.classes_, dtype=int)
    if (prob.shape != (len(df), len(classes)) or len(set(classes)) != len(classes)
            or not set(classes).issubset({0, 1, 2}) or not np.isfinite(prob).all()
            or (prob < 0).any() or (prob > 1).any() or not np.allclose(prob.sum(axis=1), 1)):
        raise ValueError('probabilidades inválidas para avaliação')
    targets = df.target.to_numpy(dtype=int)
    full, direction, confidence, calibration = probability_diagnostics(prob, classes, targets)
    binary = (targets == 1).astype(int)
    auc = float(roc_auc_score(binary, full[:, 1])) if len(set(binary)) > 1 else .5
    components = {mode: simulate(df, direction, confidence, mode=mode, payout=payout, quotes=quotes)
                  for mode in ('ai_only', 'technical_only', 'combined')}
    result = dict(components['combined'])
    lower, total = result['win_rate_lower95'], result['trades']
    reasons = []
    if total == 0:
        reasons.append('nenhum contrato no teste')
    elif result['win_rate'] < config.MIN_APPROVAL_WIN_RATE:
        reasons.append(f"taxa de acerto {result['win_rate']:.1%} abaixo do mínimo {config.MIN_APPROVAL_WIN_RATE:.1%}")
    warnings = []
    if auc <= .5:
        warnings.append(f'AUC {auc:.4f} não supera 0.5; probabilidade fraca, mas não bloqueia aprovação')
    result.update(auc=auc, accuracy=float(accuracy_score(targets, classes[prob.argmax(axis=1)])),
                  f1=float(f1_score(targets, classes[prob.argmax(axis=1)], average='macro', zero_division=0)),
                  n_test=len(df), reference_net_payout=payout, evaluation_start=int(df.entry_epoch.iloc[0]),
                  evaluation_end=int(df.expiry_epoch.iloc[-1]), evaluation_kind='deployed_strategy_reference_payout',
                  eligible=not reasons, rejection_reasons=reasons, warnings=warnings,
                  components=components, calibration=calibration)
    result['evaluation_risk_timezone'] = EVALUATION_RISK_TIMEZONE
    return result


def decision_curve(model, df, *, features=None):
    """Diagnóstico de limiares fixos; nunca escolhe limiar nem altera config."""
    selected_features = FEATURES if features is None else features
    prob = model.predict_proba(df[selected_features].values)
    _, direction, confidence, _ = probability_diagnostics(prob, np.asarray(model.classes_), df.target.values)
    curve = []
    for payout in (.60, .70, .85):
        for threshold in sorted({config.AI_CONFIDENCE_MIN, max(config.AI_CONFIDENCE_MIN, .60),
                                 max(config.AI_CONFIDENCE_MIN, .65), max(config.AI_CONFIDENCE_MIN, .70)}):
            m = simulate(df, direction, confidence, mode='combined', payout=payout, confidence_floor=threshold)
            curve.append({'payout': payout, 'payout_source': 'fixed_assumption',
                          'confidence_floor': threshold,
                          'economic_probability_floor': (1+MIN_EXPECTED_RETURN)/(1+payout),
                          **{key: m[key] for key in ('trades', 'wins', 'coverage', 'expectancy', 'max_drawdown_pct')},
                          'precision': m['win_rate'] if m['trades'] else None})
    return curve
