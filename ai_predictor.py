"""Inferência de bundle validado para o ativo, OHLC e vencimento configurados."""
import threading
import math
import config
from feature_engine import FEATURES as _FEATURES, compute_feature_map
from model_contract import FEATURE_WINDOW, load_validated_bundle, model_timeframe_sec

_model = None
_model_features = []
_model_loaded = False
_metadata = {}
_loaded_symbol = None
_load_error = ''
_lock = threading.RLock()
_prediction_context = threading.local()
_prediction_cache = None


def reset():
    """Descarta o modelo carregado e seu estado, forçando novo carregamento na próxima consulta."""
    global _model, _model_loaded, _metadata, _loaded_symbol, _model_features, _prediction_cache
    with _lock:
        _model, _model_loaded, _metadata, _loaded_symbol, _model_features = None, False, {}, None, []
        _prediction_cache = None


def _load_model():
    """Carrega e valida o bundle do ativo ativo; em falha, deixa o modelo vazio e registra o motivo."""
    global _model, _model_loaded, _model_features, _metadata, _loaded_symbol, _load_error
    _model_loaded = True
    _loaded_symbol = config.get_active_symbol()
    _model = None
    try:
        payload, metadata = load_validated_bundle(config.AI_MODEL_PATH, _loaded_symbol)
        _model = payload['model']
        _model_features = payload['features']
        if _model_features != _FEATURES:
            raise ValueError('ordem de features incompatível')
        _metadata = metadata
        _load_error = ''
    except Exception as exc:
        _model = None
        _metadata = {}
        _load_error = type(exc).__name__ + ': modelo ausente ou incompatível; entradas com IA bloqueadas'
        print('[IA] ' + _load_error)


def model_id():
    """Identificador (hash) do modelo carregado, ou string vazia."""
    return _metadata.get('model_id', '')


def prediction_model_id():
    """Identificador do modelo usado na última previsão desta thread, ou string vazia."""
    return getattr(_prediction_context, 'model_id', '')


def status():
    """Garante o carregamento do modelo e informa se está pronto, seu id e o motivo de falha."""
    with _lock:
        if not _model_loaded or _loaded_symbol != config.get_active_symbol():
            _load_model()
        return {'ready': _model is not None, 'model_id': model_id(), 'reason': _load_error}


def shadow_evidence():
    """Metadados já validados em memória para admitir o canary, sem reler pickle."""
    with _lock:
        return {key: _metadata[key] for key in (
            'model_id', 'symbol', 'timeframe_sec', 'duration', 'duration_unit',
            'approved', 'dataset_sha256', 'trained_at', 'metrics') if key in _metadata}


def _compute_feature_map(prices, candles = None):
    """Atalho para `compute_feature_map`."""
    return compute_feature_map(prices, candles)


def _extract_features(prices, candles = None):
    """Vetor de features na ordem do modelo, ou None se não for possível calcular."""
    fm = compute_feature_map(prices, candles)
    return None if fm is None else [fm[f] for f in _FEATURES]


def predict(prices, candles = None, *, evaluation_epoch=None):
    """Retorna (direção 'BUY'/'SELL' ou None, confiança); None com 0.0 se modelo ou dados forem inválidos."""
    global _prediction_cache
    _prediction_context.model_id = ""
    with _lock:
        if not _model_loaded or _loaded_symbol != config.get_active_symbol():
            _load_model()
        if _model is None or candles is None or len(prices) < FEATURE_WINDOW:
            return None, 0.0
        if len(candles) < FEATURE_WINDOW:
            return None, 0.0
        recent = candles[-FEATURE_WINDOW:]
        timeframe = model_timeframe_sec()
        try:
            epochs = [int(candle['epoch']) for candle in recent]
            if any(epoch != candle['epoch'] or epoch % timeframe
                   for epoch, candle in zip(epochs, recent)):
                return None, 0.0
            if evaluation_epoch is not None and (
                    not math.isfinite(float(evaluation_epoch))
                    or epochs[-1] + timeframe > float(evaluation_epoch)):
                return None, 0.0
        except (KeyError, TypeError, ValueError, OverflowError):
            return None, 0.0
        if any(b - a != timeframe for a, b in zip(epochs, epochs[1:])):
            return None, 0.0
        cache_key = None
        if timeframe == 300:
            # Features fechadas não mudam a cada tick. Os gates de tempo,
            # payout, risco e execução continuam sendo verificados normalmente.
            try:
                cache_key = (id(_model), model_id(), config.AI_CONFIDENCE_MIN,
                             tuple(prices[-FEATURE_WINDOW:]),
                             tuple(tuple(row[k] for k in ('epoch', 'open', 'high', 'low', 'close'))
                                   for row in recent))
            except KeyError:
                return None, 0.0
            if _prediction_cache is not None and _prediction_cache[0] == cache_key:
                _prediction_context.model_id = _metadata['model_id']
                return _prediction_cache[1]
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
            result = (direction if confidence > config.AI_CONFIDENCE_MIN else None), confidence
            if cache_key is not None:
                _prediction_cache = (cache_key, result)
            return result
        except Exception:
            return None, 0.0


def predict_duration(prices = None):
    """A duração é parte do contrato do modelo, não uma heurística concorrente."""
    return config.DURATION
