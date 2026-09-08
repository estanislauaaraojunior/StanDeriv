"""Contrato versionado entre dados, modelo, inferência e expiração."""
import hashlib
import json
import math
from pathlib import Path

SCHEMA_VERSION = 2
FEATURE_WINDOW = 100
MAX_TICK_GAP_SEC = 10
MIN_EXPECTED_RETURN = 0.02


def duration_seconds(duration, unit):
    if unit not in ('s', 'm') or int(duration) != duration or duration <= 0:
        raise ValueError('modelo de contratos exige duração positiva em segundos ou minutos')
    return int(duration) * (60 if unit == 'm' else 1)


def specification(symbol):
    import config
    import time
    from feature_engine import FEATURES
    return {'schema_version': SCHEMA_VERSION, 'symbol': symbol,
            'timeframe_sec': config.CANDLE_TIMEFRAME_SEC, 'feature_window': FEATURE_WINDOW,
            'duration': config.DURATION, 'duration_unit': config.DURATION_UNIT,
            'target': 'first_tick_after_signal_to_first_tick_at_expiry_strict_v2',
            'features': FEATURES, 'ohlc': 'time_candles', 'ties': 'separate_class_2',
            'max_tick_gap_sec': MAX_TICK_GAP_SEC,
            'strategy_parameters': {key:getattr(config,key) for key in (
                'USE_WEIGHTED_SIGNAL','SIGNAL_SCORE_MIN','AI_CONFIDENCE_MIN','AI_SCORE_MIN',
                'AI_TECH_WEIGHT','AI_MODEL_WEIGHT','ADX_MIN','ADX_ADAPTIVE','ADX_ADAPTIVE_PERCENTILE',
                'RSI_OVERSOLD','RSI_OVERBOUGHT','CANDLE_PATTERN_FILTER','CANDLE_PATTERN_MIN_STRENGTH',
                'CANDLE_PATTERN_MAX_AGE_SEC','STAKE_PCT','STOP_LOSS_PCT','TAKE_PROFIT_PCT',
                'MAX_CONSEC_LOSSES','PAUSE_BASE_SEC','PAUSE_SCALE_FACTOR','RESUME_ON_WIN','ENTRY_CANDLE_INTERVAL')},
            'risk_timezone':list(time.tzname),
            'indicator_parameters': {key: getattr(config,key) for key in (
                'EMA_FAST','EMA_SLOW','RSI_PERIOD','MACD_FAST','MACD_SLOW','MACD_SIGNAL',
                'ADX_PERIOD','BB_PERIOD','BB_STD','PA_SR_TOLERANCE')}}


def compatible(metadata, symbol):
    return isinstance(metadata, dict) and all(metadata.get(k) == v for k, v in specification(symbol).items())


def usable(metadata, symbol):
    if not compatible(metadata,symbol) or not metadata.get('approved'):
        return False
    import sklearn, xgboost
    return metadata.get('dependencies') == {'sklearn':sklearn.__version__,'xgboost':xgboost.__version__}


def fingerprint(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json_atomic(path, value):
    import os, tempfile
    p = Path(path)
    fd, tmp = tempfile.mkstemp(prefix = p.name + '.', dir = p.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(value, f, allow_nan = False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def profitable_proposal(price, payout, probability, margin = MIN_EXPECTED_RETURN):
    try:
        price, payout, probability = map(float, (price, payout, probability))
    except (TypeError, ValueError):
        return False
    return (all(math.isfinite(v) for v in (price, payout, probability))
            and 0 < price < payout and 0 <= probability <= 1
            and probability * payout / price - 1 >= margin)


def acquire_execution_lock(path):
    """Mantido até o processo sair; protege CLI e pipeline antes de alterar estado."""
    import fcntl
    handle = open(path, 'a')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise RuntimeError('já existe um executor ou pipeline neste workspace') from None
    return handle
