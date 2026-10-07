"""Contrato versionado entre dados, modelo, inferência e expiração."""
import hashlib
import json
import math
from pathlib import Path

SCHEMA_VERSION = 2
FEATURE_WINDOW = 100
MAX_TICK_GAP_SEC = 10
# Entrada autorizada quando o retorno esperado não é negativo.
MIN_EXPECTED_RETURN = 0.0
EVALUATION_RISK_TIMEZONE = 'UTC'


def minimum_training_rows():
    import config
    horizon = math.ceil(duration_seconds(config.DURATION, config.DURATION_UNIT)
                        / config.CANDLE_TIMEFRAME_SEC)
    # 15% calibration/validation partitions lose a feature window plus expiry.
    embargo = FEATURE_WINDOW + horizon
    # Menor calibração do walk-forward: 20% dos primeiros 55% do development.
    theoretical = max(math.ceil((embargo + 31) / .15),
                      math.ceil(((embargo + 32) / (.55 * .2) + embargo) / .8))
    # Na prática, gaps de mercado e a separação temporal final podem remover
    # mais linhas que o embargo idealizado. Com 15m/1m, 1.853 contratos
    # rotuláveis já produziram só 2 amostras de calibração no primeiro fold.
    return max(theoretical, 2280)


def minimum_history_candles():
    import config
    horizon = math.ceil(duration_seconds(config.DURATION, config.DURATION_UNIT)
                        / config.CANDLE_TIMEFRAME_SEC)
    # Collection budget includes room for final evaluation; it is not a gate
    # for attempting training when fewer usable rows survive real market gaps.
    # Only 20% reaches final evaluation. Reserve 120 non-overlapping
    # opportunities there, before technical filters and historical risk pauses.
    # The old budget reserved 30 at best, making approval practically unreachable.
    reserve = max(minimum_training_rows(), math.ceil(120 * (horizon + 1) / .2))
    return FEATURE_WINDOW + horizon + reserve + 5


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
            'approval_policy': {'version': 2, 'min_net_payout': config.MIN_NET_PAYOUT,
                                'min_win_rate': config.MIN_APPROVAL_WIN_RATE,
                                'min_trades': 1, 'wilson_gate': False,
                                'auc_gate': False},
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


def load_validated_bundle(path, symbol):
    """Valida sidecar e contrato antes de abrir o pickle do modelo."""
    import sklearn, xgboost
    import joblib

    model_path = Path(path)
    metadata_path = Path(str(model_path) + '.metadata.json')
    if not model_path.exists():
        raise FileNotFoundError(f"modelo não encontrado: {model_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"metadado do modelo não encontrado: {metadata_path}; "
            "gere um bundle v2 com train_model.py ou pipeline.py --force-retrain"
        )
    metadata = json.loads(metadata_path.read_text())
    artifact_sha256 = fingerprint(model_path)
    if metadata.get('artifact_sha256') != artifact_sha256:
        raise ValueError('fingerprint do modelo difere do metadado')
    metadata = {k: v for k, v in metadata.items() if k != 'artifact_sha256'}
    if not compatible(metadata, symbol) or not metadata.get('approved'):
        raise ValueError('modelo sem validação compatível com ativo, OHLC e expiração')
    if metadata.get('dependencies') != {'sklearn': sklearn.__version__, 'xgboost': xgboost.__version__}:
        raise ValueError('versões de dependências incompatíveis')
    payload = joblib.load(model_path)
    payload_metadata = payload.get('metadata', {})
    if payload_metadata != metadata:
        raise ValueError('metadado embutido difere do sidecar validado')
    return payload, dict(metadata, model_id=artifact_sha256)


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


def net_payout_ratio(price, payout):
    # Decimal preserves the inclusive boundary for quoted currency amounts.
    from decimal import Decimal
    price, payout = Decimal(str(price)), Decimal(str(payout))
    return float((payout - price) / price)


def meets_payout_floor(net_payout):
    import config
    try:
        ratio = float(net_payout)
    except (TypeError, ValueError):
        return False
    return math.isfinite(ratio) and ratio >= config.MIN_NET_PAYOUT


def valid_payout_proposal(price, payout):
    """Valida preço e payout mínimo sem inferir probabilidade não calibrada."""
    try:
        price, payout = map(float, (price, payout))
    except (TypeError, ValueError):
        return False
    return (
        math.isfinite(price)
        and math.isfinite(payout)
        and 0 < price < payout
        and meets_payout_floor(net_payout_ratio(price, payout))
    )


def profitable_proposal(price, payout, probability, margin = MIN_EXPECTED_RETURN):
    import config
    try:
        price, payout, probability = map(float, (price, payout, probability))
    except (TypeError, ValueError):
        return False
    return (all(math.isfinite(v) for v in (price, payout, probability))
            and valid_payout_proposal(price, payout)
            and config.AI_CONFIDENCE_MIN < probability <= 1
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
