"""Inferência de bundle validado para o ativo, OHLC e vencimento configurados."""
import threading
import io
import hashlib
from pathlib import Path
import math
import numpy as np
import config
from feature_engine import FEATURES as _FEATURES, compute_feature_map
from model_contract import FEATURE_WINDOW, compatible, fingerprint

_model = None
_model_features = []
_model_loaded = False
_metadata = {}
_loaded_symbol = None
_load_error = ''
_lock = threading.RLock()
_prediction_context = threading.local()


def reset():
    global _model, _model_loaded, _metadata, _loaded_symbol, _model_features
    with _lock:
        _model, _model_loaded, _metadata, _loaded_symbol, _model_features = None, False, {}, None, []


def _load_model():
    global _model, _model_loaded, _model_features, _metadata, _loaded_symbol, _load_error
    _model_loaded = True
    _loaded_symbol = config.get_active_symbol()
    _model = None
    try:
        import joblib
        artifact = Path(config.AI_MODEL_PATH).read_bytes()
        payload = joblib.load(io.BytesIO(artifact))
        metadata = payload.get('metadata', {})
        if not compatible(metadata, _loaded_symbol) or not metadata.get('approved'):
            raise ValueError('modelo sem validação compatível com ativo, OHLC e expiração')
        import sklearn, xgboost
        if metadata.get('dependencies') != {'sklearn': sklearn.__version__, 'xgboost': xgboost.__version__}:
            raise ValueError('versões de dependências incompatíveis')
        _model = payload['model']
        _model_features = payload['features']
        if _model_features != _FEATURES:
            raise ValueError('ordem de features incompatível')
        _metadata = dict(metadata, model_id = hashlib.sha256(artifact).hexdigest())
        _load_error = ''
    except Exception as exc:
        _model = None
        _metadata = {}
        _load_error = type(exc).__name__ + ': modelo ausente ou incompatível; entradas com IA bloqueadas'
        print('[IA] ' + _load_error)


def model_id():
    return _metadata.get('model_id', '')


def prediction_model_id():
    return getattr(_prediction_context, 'model_id', '')


def status():
    with _lock:
        if not _model_loaded or _loaded_symbol != config.get_active_symbol():
            _load_model()
        return {'ready': _model is not None, 'model_id': model_id(), 'reason': _load_error}


def _compute_feature_map(prices, candles = None):
    return compute_feature_map(prices, candles)


def _extract_features(prices, candles = None):
    fm = compute_feature_map(prices, candles)
    return None if fm is None else [fm[f] for f in _FEATURES]


def _extract_features_sequence(prices, seq_len, candles = None):
    """Janelas terminam na última vela; cada ponto usa exatamente FEATURE_WINDOW."""
    if candles is None or len(prices) < FEATURE_WINDOW + seq_len - 1:
        return None
    seq = []
    for end in range(len(prices) - seq_len + 1, len(prices) + 1):
        values = _extract_features(prices[end - FEATURE_WINDOW: end], candles[end - FEATURE_WINDOW: end])
        if values is None:
            return None
        seq.append(values)
    return np.asarray([seq], dtype = np.float32)


def predict(prices, candles = None):
    _prediction_context.model_id = ""
    with _lock:
        if not _model_loaded or _loaded_symbol != config.get_active_symbol():
            _load_model()
        if _model is None or candles is None or len(prices) < FEATURE_WINDOW:
            return None, 0.0
        if len(candles) < FEATURE_WINDOW:
            return None, 0.0
        recent = candles[-FEATURE_WINDOW:]
        if any(b['epoch'] - a['epoch'] != config.CANDLE_TIMEFRAME_SEC for a, b in zip(recent, recent[1:])):
            return None, 0.0
        features = _extract_features(prices, candles)
        if features is None:
            return None, 0.0
        try:
            probabilities = _model.predict_proba([features])[0]
            if (not all(math.isfinite(float(p)) and 0 <= p <= 1 for p in probabilities)
                    or not math.isclose(sum(probabilities), 1.0, abs_tol=1e-6)
                    or not set(_model.classes_).issubset({0,1,2})):
                return None, 0.0
            by_class = dict(zip(_model.classes_, probabilities))
            down, up = float(by_class.get(0, 0)), float(by_class.get(1, 0))
            if not all(math.isfinite(p) and 0 <= p <= 1 for p in (down, up)):
                return None, 0.0
            direction, confidence = ('BUY', up) if up > down else ('SELL', down)
            _prediction_context.model_id = _metadata['model_id']
            return (direction if confidence >= config.AI_CONFIDENCE_MIN else None), confidence
        except Exception:
            return None, 0.0


def predict_duration(prices = None):
    """A duração é parte do contrato do modelo, não uma heurística concorrente."""
    return config.DURATION
