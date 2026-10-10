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
from market_data import read_records, canonicalize
from model_contract import (FEATURE_WINDOW, MAX_TICK_GAP_SEC, duration_seconds,
                            specification, fingerprint, write_json_atomic)


def _extract_features(window, candles = None):
    """Atalho para `compute_feature_map` sobre a janela de preços."""
    return compute_feature_map(window, candles)


def _record_to_candle(record):
    """Converte um registro de candle do CSV no dicionário OHLC usado pelos indicadores."""
    return dict(epoch=record['epoch'], open=record['open'], high=record['high'],
                low=record['low'], close=record['price'])


def _limit_training_history(records):
    """Mantém somente os candles alinhados da janela de treino mais recente."""
    timeframe = config.CANDLE_TIMEFRAME_SEC
    candle_count = config.TRAINING_HISTORY_DAYS * 24 * 60 * 60 // timeframe
    if not records or candle_count <= 0:
        return []
    latest_epoch = max(int(record['epoch']) for record in records)
    latest_candle = latest_epoch // timeframe * timeframe
    cutoff = latest_candle - (candle_count - 1) * timeframe
    return [record for record in records if int(record['epoch']) >= cutoff]


def _merge_feature_candles(selected, ticks):
    """Candles históricos aquecem features; ticks reais continuam rotulando contratos."""
    by_epoch = {r['epoch']: _record_to_candle(r) for r in selected if r['kind'] == 'candle'}
    for candle in ind.ticks_to_candles_by_time(ticks, config.CANDLE_TIMEFRAME_SEC):
        by_epoch.setdefault(candle['epoch'], candle)
    return [by_epoch[e] for e in sorted(by_epoch)]


def _load_training_series(ticks_path, symbol):
    """Lê e valida o CSV; retorna registros do símbolo, ticks, épocas dos ticks e candles de features."""
    records, rejected = read_records(ticks_path)
    if rejected:
        raise ValueError('CSV contém registros inválidos; recupere antes de treinar')
    records, conflicts, _ = canonicalize(records)
    if conflicts:
        raise ValueError('preços conflitantes no mesmo instante; requer recuperação')
    selected = [r for r in records if r['symbol'] == symbol]
    selected = _limit_training_history(selected)
    if any(r['kind'] == 'legacy_unknown' for r in selected):
        raise ValueError('dados legados sem proveniência: não é possível distinguir tick de close histórico')
    ticks = [r for r in selected if r['kind'] == 'tick']
    if not ticks:
        raise ValueError(f'nenhum tick identificado para {symbol}; candles históricos não servem como ticks de entrada')
    return selected, ticks, [r['epoch'] for r in ticks], _merge_feature_candles(selected, ticks)


def count_labelable_contracts(ticks_path, symbol = None, window_size = FEATURE_WINDOW):
    """Conta janelas com features causais, entrada real e vencimento real observáveis."""
    if window_size != FEATURE_WINDOW:
        raise ValueError(f'janela deve ser {FEATURE_WINDOW}, igual à inferência')
    symbol = symbol or config.get_active_symbol()
    horizon = duration_seconds(config.DURATION, config.DURATION_UNIT)
    selected, ticks, epochs, candles = _load_training_series(ticks_path, symbol)
    bad_gaps = [0]
    for a, b in zip(epochs, epochs[1:]):
        bad_gaps.append(bad_gaps[-1] + (b - a > MAX_TICK_GAP_SEC))
    rows = 0
    segment_start, previous_tick = 0, 0
    for i in range(len(candles)):
        last_tick = bisect.bisect_left(epochs, candles[i]['epoch'] + config.CANDLE_TIMEFRAME_SEC) - 1
        if (i and candles[i]['epoch'] - candles[i - 1]['epoch'] != config.CANDLE_TIMEFRAME_SEC
                or (last_tick >= 0 and previous_tick >= 0 and bad_gaps[last_tick] != bad_gaps[previous_tick])):
            segment_start = i + 1
        previous_tick = last_tick
        window = candles[max(segment_start, i - FEATURE_WINDOW + 1): i + 1]
        if len(window) < FEATURE_WINDOW:
            continue
        signal_epoch = window[-1]['epoch'] + config.CANDLE_TIMEFRAME_SEC
        entry = bisect.bisect_left(epochs, signal_epoch)
        if entry == len(ticks) or epochs[entry] - signal_epoch > MAX_TICK_GAP_SEC:
            continue
        expiry_epoch = epochs[entry] + horizon
        expiry = bisect.bisect_left(epochs, expiry_epoch, lo=entry)
        if (expiry == len(ticks) or epochs[expiry] - expiry_epoch > MAX_TICK_GAP_SEC
                or bad_gaps[expiry] != bad_gaps[entry]
                or any(b['epoch'] - a['epoch'] != config.CANDLE_TIMEFRAME_SEC for a, b in zip(window, window[1:]))):
            continue
        rows += 1
    return rows


def build_dataset(ticks_path, output_path, window_size = FEATURE_WINDOW, lookforward = None, symbol = None):
    """Gera o CSV de treino (features, alvo CALL/PUT/empate) e seu metadado .metadata.json; retorna o número de linhas."""
    if window_size != FEATURE_WINDOW:
        raise ValueError(f'janela deve ser {FEATURE_WINDOW}, igual à inferência')
    horizon = duration_seconds(config.DURATION, config.DURATION_UNIT)
    if lookforward is not None and lookforward * config.CANDLE_TIMEFRAME_SEC != horizon:
        raise ValueError('lookforward incompatível com o vencimento configurado')
    symbol = symbol or config.get_active_symbol()
    source_hash = fingerprint(ticks_path)
    selected, ticks, epochs, candles = _load_training_series(ticks_path, symbol)
    bad_gaps = [0]
    for a, b in zip(epochs, epochs[1:]):
        bad_gaps.append(bad_gaps[-1] + (b - a > MAX_TICK_GAP_SEC))
    rows = []
    adx_history, alerts = deque(maxlen=500), deque(maxlen=20)
    segment_start, previous_tick = 0, 0
    for i in range(len(candles)):
        last_tick = bisect.bisect_left(epochs, candles[i]['epoch']+config.CANDLE_TIMEFRAME_SEC)-1
        if (i and candles[i]['epoch']-candles[i-1]['epoch'] != config.CANDLE_TIMEFRAME_SEC
                or (last_tick >= 0 and previous_tick >= 0
                    and bad_gaps[last_tick] != bad_gaps[previous_tick])):
            adx_history.clear()
            alerts.clear()
            segment_start = i + 1
        previous_tick = last_tick
        window = candles[max(segment_start,i-FEATURE_WINDOW+1):i+1]
        if not window:
            continue
        prices = [c['close'] for c in window]
        prior_adx = list(adx_history)
        current_adx = ind.adx(prices, config.ADX_PERIOD)
        if current_adx is not None:
            adx_history.append(current_adx)
        decision = bisect.bisect_left(epochs, window[-1]['epoch']+config.CANDLE_TIMEFRAME_SEC)
        if decision < len(epochs):
            for pattern in ind.detect_candle_patterns(window):
                if len(window)>=4 and pattern['strength']>=.5:
                    alerts.append(dict(pattern,timestamp=epochs[decision]))
        if len(window)<FEATURE_WINDOW:
            continue
        signal_epoch = window[-1]['epoch'] + config.CANDLE_TIMEFRAME_SEC
        entry = bisect.bisect_left(epochs, signal_epoch)
        if entry == len(ticks) or epochs[entry] - signal_epoch > MAX_TICK_GAP_SEC:
            continue
        expiry_epoch = epochs[entry] + horizon
        expiry = bisect.bisect_left(epochs, expiry_epoch, lo = entry)
        start = bisect.bisect_left(epochs, window[0]['epoch'])
        if (expiry == len(ticks) or epochs[expiry] - expiry_epoch > MAX_TICK_GAP_SEC
                or bad_gaps[expiry] != bad_gaps[entry]
                or any(b['epoch'] - a['epoch'] != config.CANDLE_TIMEFRAME_SEC for a, b in zip(window, window[1:]))):
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
        experimental = compute_experimental_features(prices, signal_epoch, config.CANDLE_TIMEFRAME_SEC)
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
