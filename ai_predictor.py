"""
ai_predictor.py — módulo de inferência em tempo real.

Responsabilidades:
  - Carregar model.pkl uma única vez (lazy singleton)
  - Extrair features de uma lista de preços ao vivo usando as mesmas
    funções de indicators.py que o dataset_builder.py usa no treino
  - Retornar direção prevista e probabilidade (confiança)

Interface pública:
    from ai_predictor import predict

    direction, confidence = predict(prices)
    # direction  → "BUY" | "SELL" | None
    # confidence → float entre 0.0 e 1.0
    #   None é retornado quando: model.pkl não encontrado, dados insuficientes
    #   ou confiança abaixo de AI_CONFIDENCE_MIN (configurável em config.py)
"""

import os
from typing import Optional

import numpy as np

import indicators as ind
from config import (
    EMA_FAST, EMA_SLOW,
    RSI_PERIOD,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    ADX_PERIOD,
    BB_PERIOD, BB_STD,
    AI_MODEL_PATH, AI_CONFIDENCE_MIN,
    DURATION, DURATION_MODEL_PATH,
    USE_TRANSFORMER, TRANSFORMER_MODEL_PATH,
    TRANSFORMER_BLEND_WEIGHT, TRANSFORMER_SEQ_LEN,
    CANDIDATE_DURATIONS,
    CANDLE_SIZE, PA_SR_TOLERANCE,
)
from feature_engine import compute_feature_map as _fe_compute_feature_map, FEATURES as _FE_FEATURES

# Lista de features — importada de feature_engine (fonte única)
_FEATURES = _FE_FEATURES

# Singleton: modelo carregado uma única vez
_model = None
_model_features: list = []
_model_loaded: bool = False  # True mesmo se não encontrado (evita re-tentativas)

# Singleton do modelo de duração dinâmica
_dur_model: object = None
_dur_model_features: list = []
_dur_model_loaded: bool = False

# Singleton do Temporal Fusion Transformer
_tft_model: object = None
_tft_features: list  = []
_tft_seq_len: int    = TRANSFORMER_SEQ_LEN
_tft_dur_classes: list = []
_tft_model_loaded: bool = False


def _load_model() -> None:
    """Carrega model.pkl no singleton global. Chamado automaticamente na primeira predição."""
    global _model, _model_features, _model_loaded
    _model_loaded = True

    if not os.path.exists(AI_MODEL_PATH):
        print(
            f"[IA] model.pkl não encontrado em '{AI_MODEL_PATH}'. "
            "Execute train_model.py para treinar o modelo. "
            "Bot operará apenas com indicadores técnicos."
        )
        return

    try:
        import joblib
        payload = joblib.load(AI_MODEL_PATH)
        _model          = payload["model"]
        _model_features = payload.get("features", _FEATURES)
        model_name      = payload.get("name", "desconhecido")

        # P3: Validar compatibilidade de features entre modelo e código atual
        if _model_features and _model_features != _FEATURES:
            print(f"[IA] AVISO: lista de features do modelo difere da atual!")
            print(f"[IA]   Modelo : {_model_features}")
            print(f"[IA]   Atual  : {_FEATURES}")
        assert len(_model_features) == len(_FEATURES), (
            f"[IA] ERRO FATAL: modelo tem {len(_model_features)} features, "
            f"código espera {len(_FEATURES)}. Execute train_model.py novamente."
        )

        print(f"[IA] Modelo carregado: {model_name} ({AI_MODEL_PATH})")
    except AssertionError as exc:
        print(exc)
        _model = None
    except Exception as exc:
        print(f"[IA] Erro ao carregar modelo: {exc}. Bot operará sem IA.")
        _model = None


# ─────────────────────────────────────────────────────────────────
#  Extração de features (idêntica ao dataset_builder.py)
# ─────────────────────────────────────────────────────────────────

def _compute_feature_map(prices: list) -> Optional[dict]:
    """Delega para feature_engine.compute_feature_map (fonte única de features)."""
    return _fe_compute_feature_map(prices)


def _extract_features(prices: list) -> Optional[list]:
    """
    Extrai features de uma lista de preços (ticks recentes).

    Retorna lista de floats na ordem de _FEATURES, ou None se insuficiente.
    """
    fm = _compute_feature_map(prices)
    if fm is None:
        return None
    feat_order = _model_features if _model_features else _FEATURES
    return [fm.get(f, 0.0) for f in feat_order]


def _extract_features_sequence(
    prices: list,
    seq_len: int,
) -> Optional[np.ndarray]:
    """
    Constrói a janela temporal de features para o TFT.

    Para cada um dos últimos seq_len ticks, calcula o vetor de features
    usando uma janela de até 100 preços anteriores (idêntico ao dataset_builder).

    Retorna ndarray (1, seq_len, n_features) ou None se dados insuficientes.
    Requer len(prices) >= seq_len + 100.
    """
    _WINDOW = 100   # janela de histórico por ponto (igual ao dataset_builder)
    n = len(prices)
    if n < seq_len + _WINDOW:
        return None

    seq = []
    for j in range(seq_len):
        # Posição do tick mais recente desta sub-janela
        end_idx = n - seq_len + j
        window  = prices[max(0, end_idx - _WINDOW) : end_idx + 1]
        fm = _compute_feature_map(window)
        if fm is None:
            return None
        seq.append([fm.get(f, 0.0) for f in _FEATURES])

    return np.array([seq], dtype=np.float32)  # (1, seq_len, n_features)


# ─────────────────────────────────────────────────────────────────
#  TFT — carregamento e predição
# ─────────────────────────────────────────────────────────────────

def _load_tft_model() -> None:
    """Carrega o TFT de TRANSFORMER_MODEL_PATH no singleton global (lazy)."""
    global _tft_model, _tft_features, _tft_seq_len, _tft_dur_classes, _tft_model_loaded
    _tft_model_loaded = True

    if not USE_TRANSFORMER:
        return

    if not os.path.exists(TRANSFORMER_MODEL_PATH):
        return  # fallback silencioso para o modelo clássico

    try:
        import joblib
        payload          = joblib.load(TRANSFORMER_MODEL_PATH)
        _tft_model       = payload["model"]
        _tft_features    = payload.get("features", _FEATURES)
        _tft_seq_len     = payload.get("seq_len", TRANSFORMER_SEQ_LEN)
        _tft_dur_classes = payload.get("dur_classes", CANDIDATE_DURATIONS)
        print(f"[IA] Modelo TFT carregado ({TRANSFORMER_MODEL_PATH})")
    except Exception as exc:
        print(f"[IA] Erro ao carregar TFT: {exc} — usando apenas modelo clássico.")
        _tft_model = None


def _predict_classical(prices: list) -> tuple:
    """
    Prediz direção usando o modelo clássico (sklearn StackingClassifier).

    Retorna (direction, confidence) ou (None, 0.0) se modelo indisponível/insuficiente.
    """
    if _model is None:
        return None, 0.0

    feat_vec = _extract_features(prices)
    if feat_vec is None:
        return None, 0.0

    try:
        proba  = _model.predict_proba([feat_vec])[0]   # [P_DOWN, P_UP]
        p_up   = float(proba[1])
        p_down = float(proba[0])

        if p_up >= p_down:
            direction, conf = "BUY",  p_up
        else:
            direction, conf = "SELL", p_down

        return direction, conf
    except Exception:
        return None, 0.0


def _predict_tft(prices: list) -> tuple:
    """
    Prediz direção usando o Temporal Fusion Transformer.

    Retorna (direction, confidence) ou (None, 0.0) se TFT indisponível/insuficiente.
    """
    if _tft_model is None:
        return None, 0.0

    X_seq = _extract_features_sequence(prices, _tft_seq_len)
    if X_seq is None:
        return None, 0.0

    try:
        proba  = _tft_model.predict_proba(X_seq)[0]   # [P_DOWN, P_UP]
        p_up   = float(proba[1])
        p_down = float(proba[0])

        if p_up >= p_down:
            direction, conf = "BUY",  p_up
        else:
            direction, conf = "SELL", p_down

        return direction, conf
    except Exception:
        return None, 0.0


def _predict_duration_tft(prices: list) -> Optional[int]:
    """
    Prediz duração usando o head de duração do TFT.

    Retorna duração em ticks (int) ou None se TFT indisponível/insuficiente.
    """
    if _tft_model is None:
        return None

    X_seq = _extract_features_sequence(prices, _tft_seq_len)
    if X_seq is None:
        return None

    try:
        dur_classes = _tft_dur_classes if _tft_dur_classes else CANDIDATE_DURATIONS
        dur_idx = int(_tft_model.predict_duration(X_seq)[0])
        if 0 <= dur_idx < len(dur_classes):
            return int(dur_classes[dur_idx])
        return None
    except Exception:
        return None


def predict(prices: list) -> tuple:
    """
    Prevê a direção do próximo tick via ensemble TFT + clássico.

    Lógica de blend:
      - Ambos concordam  → conf = TFT*BLEND + clássico*(1-BLEND)
      - Discordância      → penalidade de 15% no mais confiante
      - Apenas um disponível → usa o que estiver disponível diretamente
      - Nenhum disponível → (None, 0.0)

    Args:
        prices: lista de floats com histórico de preços
    Returns:
        (direction, confidence)
        direction  → "BUY" | "SELL" | None
        confidence → probabilidade ponderada (0.0–1.0)
    """
    global _model_loaded, _tft_model_loaded

    if not _model_loaded:
        _load_model()
    if not _tft_model_loaded:
        _load_tft_model()

    classical_dir, classical_conf = _predict_classical(prices)
    tft_dir,       tft_conf       = _predict_tft(prices)

    # Nenhum modelo disponível
    if classical_dir is None and tft_dir is None:
        return None, max(classical_conf, tft_conf)

    # Apenas modelo clássico disponível
    if tft_dir is None or _tft_model is None:
        if classical_conf >= AI_CONFIDENCE_MIN:
            return classical_dir, classical_conf
        return None, classical_conf

    # Apenas TFT disponível
    if classical_dir is None or _model is None:
        if tft_conf >= AI_CONFIDENCE_MIN:
            return tft_dir, tft_conf
        return None, tft_conf

    # Ambos disponíveis: ensemble blend
    blend = TRANSFORMER_BLEND_WEIGHT   # 0.55

    if tft_dir == classical_dir:
        # Concordância: média ponderada das confianças
        blended = tft_conf * blend + classical_conf * (1.0 - blend)
        if blended >= AI_CONFIDENCE_MIN:
            return tft_dir, blended
        return None, blended
    else:
        # Discordância: não operar — modelos divergem
        blended = tft_conf * blend + classical_conf * (1.0 - blend)
        return None, blended


# ─────────────────────────────────────────────────────────────────
#  Duração dinâmica
# ─────────────────────────────────────────────────────────────────

def _load_duration_model() -> None:
    """Carrega duration_model.pkl no singleton global (lazy). Falha silenciosa."""
    global _dur_model, _dur_model_features, _dur_model_loaded
    _dur_model_loaded = True

    if not os.path.exists(DURATION_MODEL_PATH):
        return   # fallback para DURATION fixo de config.py

    try:
        import joblib
        payload             = joblib.load(DURATION_MODEL_PATH)
        _dur_model          = payload["model"]
        _dur_model_features = payload.get("features", _FEATURES)
        print(f"[IA] Modelo de duração carregado ({DURATION_MODEL_PATH})")
    except Exception as exc:
        print(f"[IA] Erro ao carregar modelo de duração: {exc} — usando DURATION fixo.")


def predict_duration(prices: list) -> int:
    """
    Retorna a duração prevista (em ticks) para o próximo contrato.

    Prioridade:
      1. Head de duração do TFT (se disponível e com dados suficientes)
      2. duration_model.pkl (RF clássico)
      3. DURATION fixo de config.py (fallback final)
    """
    global _dur_model_loaded, _tft_model_loaded
    if not _dur_model_loaded:
        _load_duration_model()
    if not _tft_model_loaded:
        _load_tft_model()

    # 1. TFT duration head (maior precisão temporal)
    dur_tft = _predict_duration_tft(prices)
    if dur_tft is not None:
        return dur_tft

    # 2. Modelo clássico de duração (RF)
    if _dur_model is None:
        return DURATION

    feat_vec = _extract_features(prices)
    if feat_vec is None:
        return DURATION

    # Reconstrói vetor na ordem do modelo de duração
    feat_order = _dur_model_features if _dur_model_features else _FEATURES
    feature_map = dict(zip(_FEATURES, feat_vec))
    X = [[feature_map.get(f, 0.0) for f in feat_order]]

    try:
        return int(_dur_model.predict(X)[0])
    except Exception:
        return DURATION
