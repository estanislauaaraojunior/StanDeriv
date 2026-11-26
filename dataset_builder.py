"""Dataset de CALL/PUT: OHLC causal, entrada no tick seguinte e vencimento fixo."""
import argparse
import bisect
import csv
from collections import deque
import strategy
from pathlib import Path

import config
import indicators as ind
from feature_engine import (compute_feature_map, compute_experimental_features,
                            EXPERIMENTAL_FEATURES, EXPERIMENTAL_FEATURES_VERSION)
from model_contract import (FEATURE_WINDOW, MAX_TICK_GAP_SEC, duration_seconds,
                            specification, fingerprint, write_json_atomic, model_timeframe_sec)
from training_data import TRAINING_HISTORY_DAYS, load_records


def _extract_features(window, candles = None):
    """Atalho para `compute_feature_map` sobre a janela de preços."""
    return compute_feature_map(window, candles)


def _record_to_candle(record):
    """Converte um registro de candle do CSV no dicionário OHLC usado pelos indicadores."""
    return dict(epoch=record['epoch'], open=record['open'], high=record['high'],
                low=record['low'], close=record['price'])


def _limit_training_history(records):
    """Mantém somente os candles alinhados da janela de treino mais recente."""
    timeframe = model_timeframe_sec()
    candle_count = TRAINING_HISTORY_DAYS * 24 * 60 * 60 // timeframe
    if not records or candle_count <= 0:
        return []
    latest_epoch = max(int(record['epoch']) for record in records)
    latest_candle = latest_epoch // timeframe * timeframe
    cutoff = latest_candle - (candle_count - 1) * timeframe
    return [record for record in records if int(record['epoch']) >= cutoff]


def _merge_feature_candles(selected, ticks):
    """OHLC confirmado independe dos gaps de ticks; agregações exigem cobertura."""
    timeframe = model_timeframe_sec()
    latest_tick = ticks[-1]['epoch']
    by_epoch = {r['epoch']: _record_to_candle(r) for r in selected
                if r['kind'] == 'candle' and r['timeframe'] == timeframe
                and r['epoch'] + timeframe <= latest_tick}
    epochs = [tick['epoch'] for tick in ticks]
    gaps = [0]
    for a, b in zip(epochs, epochs[1:]):
        gaps.append(gaps[-1] + (b - a > MAX_TICK_GAP_SEC))
    for candle in ind.ticks_to_candles_by_time(ticks, timeframe):
        start = bisect.bisect_left(epochs, candle['epoch'])
        end = bisect.bisect_left(epochs, candle['epoch'] + timeframe) - 1
        if (end < start or epochs[start] - candle['epoch'] > MAX_TICK_GAP_SEC
                or candle['epoch'] + timeframe - epochs[end] > MAX_TICK_GAP_SEC
                or gaps[end] != gaps[start]):
            continue
        by_epoch.setdefault(candle['epoch'], candle)
    return [by_epoch[e] for e in sorted(by_epoch)]


def _load_training_series(ticks_path, symbol):
    """Lê e valida o CSV; retorna registros do símbolo, ticks, épocas dos ticks e candles de features."""
    records, preparation = load_records(ticks_path, symbol)
    return _training_series(records, preparation, symbol)


def _training_series(records, preparation, symbol):
    """Monta a série a partir dos registros já validados pela preparação da IA."""
    selected = [r for r in records if r['symbol'] == symbol]
    latest_tick = max((r['epoch'] for r in selected if r['kind'] == 'tick'), default=0)
    selected = [r for r in selected if r['kind'] != 'candle'
                or r['epoch'] + r['timeframe'] <= latest_tick]
    selected = _limit_training_history(selected)
    if any(r['kind'] == 'legacy_unknown' for r in selected):
        raise ValueError('dados legados sem proveniência: não é possível distinguir tick de close histórico')
    ticks = [r for r in selected if r['kind'] == 'tick']
    if not ticks:
        raise ValueError(f'nenhum tick identificado para {symbol}; candles históricos não servem como ticks de entrada')
    return selected, ticks, [r['epoch'] for r in ticks], _merge_feature_candles(selected, ticks), preparation


def _excluded_window(preparation, start, end):
    """Veta qualquer amostra que alcance um registro excluído da origem."""
    epochs = preparation['excluded_epochs']
    index = bisect.bisect_left(epochs, start)
    return index < len(epochs) and epochs[index] <= end


def count_labelable_contracts(ticks_path, symbol = None, window_size = FEATURE_WINDOW):
    """Conta janelas com features causais, entrada real e vencimento real observáveis."""
    if window_size != FEATURE_WINDOW:
        raise ValueError(f'janela deve ser {FEATURE_WINDOW}, igual à inferência')
    symbol = symbol or config.get_active_symbol()
    return _count_labelable_series(_load_training_series(ticks_path, symbol))


def _count_labelable_series(series):
    """Conta os rótulos sobre a série já validada, sem reler a fonte."""
    return sum(1 for _ in _labelable_rows(series))


def _labelable_rows(series):
    """Descreve rótulos causais sem ajustar modelos nem calcular features."""
    horizon = duration_seconds(config.DURATION, config.DURATION_UNIT)
    timeframe = model_timeframe_sec()
    selected, ticks, epochs, candles, preparation = series
    bad_gaps = [0]
    for a, b in zip(epochs, epochs[1:]):
        bad_gaps.append(bad_gaps[-1] + (b - a > MAX_TICK_GAP_SEC))
    segment_start = 0
    for i in range(len(candles)):
        if i and candles[i]['epoch'] - candles[i - 1]['epoch'] != timeframe:
            segment_start = i
        window = candles[max(segment_start, i - FEATURE_WINDOW + 1): i + 1]
        if len(window) < FEATURE_WINDOW:
            continue
        signal_epoch = window[-1]['epoch'] + timeframe
        entry = bisect.bisect_left(epochs, signal_epoch)
        if entry == len(ticks) or epochs[entry] - signal_epoch > MAX_TICK_GAP_SEC:
            continue
        expiry_epoch = epochs[entry] + horizon
        expiry = bisect.bisect_left(epochs, expiry_epoch, lo=entry)
        if (expiry == len(ticks) or epochs[expiry] - expiry_epoch > MAX_TICK_GAP_SEC
                or bad_gaps[expiry] != bad_gaps[entry]
                or _excluded_window(preparation, window[0]['epoch'], epochs[expiry])
                or any(b['epoch'] - a['epoch'] != timeframe for a, b in zip(window, window[1:]))):
            continue
        change = ticks[expiry]['price'] - ticks[entry]['price']
        yield dict(signal_epoch=signal_epoch, entry_epoch=epochs[entry],
                   expiry_epoch=epochs[expiry], feature_start_epoch=window[0]['epoch'],
                   target=1 if change > 0 else (0 if change < 0 else 2))


def training_diagnostics(ticks_path, symbol):
    """Retorna progresso real do treino e continuidade, sem executar modelos."""
    series = _load_training_series(ticks_path, symbol)
    return _training_diagnostics(series, symbol)


def training_diagnostics_from_records(records, preparation, symbol):
    """Diagnostica páginas já validadas, sem regravar snapshots entre lotes."""
    return _training_diagnostics(_training_series(records, preparation, symbol), symbol)


def _training_diagnostics(series, symbol):
    from model_contract import minimum_training_rows
    selected, ticks, epochs, candles, preparation = series
    rows = list(_labelable_rows(series))
    labels = len(rows)
    longest = 0
    start = previous = epochs[0]
    gaps = 0
    for epoch in epochs[1:]:
        if epoch - previous > MAX_TICK_GAP_SEC:
            longest = max(longest, previous - start)
            start = epoch
            gaps += 1
        previous = epoch
    longest = max(longest, previous - start)
    required = minimum_training_rows()
    import pandas as pd
    from training_policy import readiness_capacity
    readiness = readiness_capacity(pd.DataFrame(rows), required=required)
    return dict(readiness,
                symbol=symbol, labelable_contracts=labels, required_contracts=required,
                ticks=len(ticks), feature_candles=len(candles), history_days=TRAINING_HISTORY_DAYS,
                tick_gaps=gaps, longest_tick_segment_sec=longest,
                quarantined_epochs=len(preparation['excluded_epochs']),
                _label_ids=[(r['signal_epoch'], r['entry_epoch'], r['expiry_epoch'], r['target'])
                            for r in rows])


def build_dataset(ticks_path, output_path, window_size = FEATURE_WINDOW, lookforward = None, symbol = None):
    """Gera o CSV de treino (features, alvo CALL/PUT/empate) e seu metadado .metadata.json; retorna o número de linhas."""
    if window_size != FEATURE_WINDOW:
        raise ValueError(f'janela deve ser {FEATURE_WINDOW}, igual à inferência')
    horizon = duration_seconds(config.DURATION, config.DURATION_UNIT)
    timeframe = model_timeframe_sec()
    if lookforward is not None and lookforward * timeframe != horizon:
        raise ValueError('lookforward incompatível com o vencimento configurado')
    symbol = symbol or config.get_active_symbol()
    source_hash = fingerprint(ticks_path)
    selected, ticks, epochs, candles, preparation = _load_training_series(ticks_path, symbol)
    bad_gaps = [0]
    for a, b in zip(epochs, epochs[1:]):
        bad_gaps.append(bad_gaps[-1] + (b - a > MAX_TICK_GAP_SEC))
    rows = []
    adx_history, alerts = deque(maxlen=500), deque(maxlen=20)
    segment_start = 0
    for i in range(len(candles)):
        if i and candles[i]['epoch']-candles[i-1]['epoch'] != timeframe:
            adx_history.clear()
            alerts.clear()
            segment_start = i
        window = candles[max(segment_start,i-FEATURE_WINDOW+1):i+1]
        if not window:
            continue
        prices = [c['close'] for c in window]
        prior_adx = list(adx_history)
        current_adx = ind.adx(prices, config.ADX_PERIOD)
        if current_adx is not None:
            adx_history.append(current_adx)
        decision = bisect.bisect_left(epochs, window[-1]['epoch']+timeframe)
        if decision < len(epochs):
            for pattern in ind.detect_candle_patterns(window):
                if len(window)>=4 and pattern['strength']>=.5:
                    alerts.append(dict(pattern,timestamp=epochs[decision]))
        if len(window)<FEATURE_WINDOW:
            continue
        signal_epoch = window[-1]['epoch'] + timeframe
        entry = bisect.bisect_left(epochs, signal_epoch)
        if entry == len(ticks) or epochs[entry] - signal_epoch > MAX_TICK_GAP_SEC:
            continue
        expiry_epoch = epochs[entry] + horizon
        expiry = bisect.bisect_left(epochs, expiry_epoch, lo = entry)
        start = bisect.bisect_left(epochs, window[0]['epoch'])
        if (expiry == len(ticks) or epochs[expiry] - expiry_epoch > MAX_TICK_GAP_SEC
                or bad_gaps[expiry] != bad_gaps[entry]
                or _excluded_window(preparation, window[0]['epoch'], epochs[expiry])
                or any(b['epoch'] - a['epoch'] != timeframe for a, b in zip(window, window[1:]))):
            continue
        features = compute_feature_map([c['close'] for c in window], window)
        if features is None:
            continue
        technical_signal, technical = strategy.get_signal(
            prices, adx_min=strategy.get_adaptive_adx_min(prior_adx), adx_history=prior_adx,
            candle_alerts=[], use_ai_model=False, evaluation_time=epochs[entry], candles=window)
        change = ticks[expiry]['price'] - ticks[entry]['price']
        features.update(technical_signal={'BUY':1,'SELL':0}.get(technical_signal,-1),
                        tech_score=technical.get('tech_score',1.0 if technical_signal else 0.0),
                        symbol = symbol, signal_epoch = signal_epoch,
                        entry_epoch = epochs[entry], expiry_epoch = epochs[expiry],
                        feature_start_epoch = window[0]['epoch'],
                        entry_price = ticks[entry]['price'], exit_price = ticks[expiry]['price'],
                        target = 1 if change > 0 else (0 if change < 0 else 2),
                        duration = config.DURATION, duration_unit = config.DURATION_UNIT)
        experimental = compute_experimental_features(prices, signal_epoch, timeframe)
        features.update(experimental or {key: float('nan') for key in EXPERIMENTAL_FEATURES})
        rows.append(features)
    if not rows:
        raise ValueError('nenhuma janela contínua com entrada e vencimento observáveis')
    if fingerprint(ticks_path) != source_hash:
        raise ValueError('arquivo de origem mudou; use um snapshot estável')
    with Path(output_path).open('w', newline = '') as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    metadata = specification(symbol)
    metadata['experimental_features_version'] = EXPERIMENTAL_FEATURES_VERSION
    metadata['training_preparation'] = preparation
    metadata.update(source_sha256 = source_hash, dataset_sha256 = fingerprint(output_path),
                    first_signal = rows[0]['signal_epoch'], last_expiry = rows[-1]['expiry_epoch'],
                    n_rows = len(rows), labeling = 'offline proxy; official contract prices must be reconciled live')
    write_json_atomic(str(output_path) + '.metadata.json', metadata)
    print(f'[DATASET] {len(rows)} contratos rotulados para {symbol}; duração {config.DURATION}{config.DURATION_UNIT}')
    return len(rows)


def main():
    """CLI: constrói o dataset a partir dos ticks conforme os argumentos informados."""
    p = argparse.ArgumentParser(description = __doc__)
    p.add_argument('--input', default = config.TICKS_CSV)
    p.add_argument('--output', default = config.DATASET_CSV)
    p.add_argument('--symbol', default = config.get_active_symbol())
    p.add_argument('--window', type = int, default = FEATURE_WINDOW)
    p.add_argument('--lookforward', type = int, default = None)
    a = p.parse_args()
    build_dataset(a.input, a.output, a.window, a.lookforward, a.symbol)


if __name__ == '__main__':
    main()
