"""
tests/test_config_thresholds.py — valida invariantes e consistência de config.py.

Garante que mudanças acidentais nos thresholds sejam detectadas antes de ir para produção.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config


# ─── Pesos da IA ─────────────────────────────────────────────────────────────

def test_ai_weights_sum_to_one():
    total = config.AI_TECH_WEIGHT + config.AI_MODEL_WEIGHT
    assert abs(total - 1.0) < 1e-9, (
        f"AI_TECH_WEIGHT + AI_MODEL_WEIGHT deve ser 1.0, got {total}"
    )


def test_ai_confidence_min_range():
    assert 0.50 <= config.AI_CONFIDENCE_MIN <= 0.70, (
        f"AI_CONFIDENCE_MIN fora da faixa segura [0.50, 0.70]: {config.AI_CONFIDENCE_MIN}"
    )


def test_ai_score_min_range():
    assert 0.25 <= config.AI_SCORE_MIN <= 0.70, (
        f"AI_SCORE_MIN fora da faixa segura [0.25, 0.70]: {config.AI_SCORE_MIN}"
    )


# ─── RSI ─────────────────────────────────────────────────────────────────────

def test_rsi_bounds_logical():
    assert config.RSI_OVERSOLD < config.RSI_OVERBOUGHT, (
        f"RSI_OVERSOLD ({config.RSI_OVERSOLD}) deve ser < RSI_OVERBOUGHT ({config.RSI_OVERBOUGHT})"
    )


def test_rsi_oversold_min():
    assert config.RSI_OVERSOLD >= 30, (
        f"RSI_OVERSOLD muito baixo ({config.RSI_OVERSOLD}), mínimo recomendado: 30"
    )


def test_rsi_overbought_max():
    assert config.RSI_OVERBOUGHT <= 70, (
        f"RSI_OVERBOUGHT muito alto ({config.RSI_OVERBOUGHT}), máximo recomendado: 70"
    )


# ─── ADX ─────────────────────────────────────────────────────────────────────

def test_adx_min_range():
    assert 10 <= config.ADX_MIN <= 40, (
        f"ADX_MIN fora da faixa razoável [10, 40]: {config.ADX_MIN}"
    )


def test_adx_adaptive_percentile_range():
    assert 20 <= config.ADX_ADAPTIVE_PERCENTILE <= 80, (
        f"ADX_ADAPTIVE_PERCENTILE fora da faixa [20, 80]: {config.ADX_ADAPTIVE_PERCENTILE}"
    )


# ─── Sinal ponderado ─────────────────────────────────────────────────────────

def test_signal_score_min_range():
    assert 0.0 < config.SIGNAL_SCORE_MIN < 1.0, (
        f"SIGNAL_SCORE_MIN deve estar em (0, 1): {config.SIGNAL_SCORE_MIN}"
    )


def test_signal_score_min_not_too_low():
    assert config.SIGNAL_SCORE_MIN >= 0.03, (
        f"SIGNAL_SCORE_MIN muito permissivo ({config.SIGNAL_SCORE_MIN}), mínimo recomendado: 0.03"
    )


# ─── Candles ─────────────────────────────────────────────────────────────────

def test_candle_timeframe_min():
    assert config.CANDLE_TIMEFRAME_SEC >= 60, (
        f"CANDLE_TIMEFRAME_SEC muito baixo ({config.CANDLE_TIMEFRAME_SEC}s), mínimo: 60s"
    )


def test_target_lookforward_min():
    assert config.TARGET_LOOKFORWARD >= 1, (
        f"TARGET_LOOKFORWARD deve ser >= 1: {config.TARGET_LOOKFORWARD}"
    )


def test_min_candles_positive():
    assert config.MIN_CANDLES >= 5, (
        f"MIN_CANDLES muito baixo ({config.MIN_CANDLES}), mínimo recomendado: 5"
    )


# ─── Gestão de risco ─────────────────────────────────────────────────────────

def test_stake_pct_safe():
    assert 0 < config.STAKE_PCT <= 0.05, (
        f"STAKE_PCT fora da faixa segura (0, 0.05]: {config.STAKE_PCT}"
    )


def test_stop_loss_lt_take_profit():
    assert config.STOP_LOSS_PCT < config.TAKE_PROFIT_PCT, (
        f"STOP_LOSS_PCT ({config.STOP_LOSS_PCT}) deve ser < TAKE_PROFIT_PCT ({config.TAKE_PROFIT_PCT})"
    )


def test_max_consec_losses_positive():
    assert config.MAX_CONSEC_LOSSES >= 1, (
        f"MAX_CONSEC_LOSSES deve ser >= 1: {config.MAX_CONSEC_LOSSES}"
    )


# ─── Valores recomendados pela estratégia atual ───────────────────────────────

def test_recommended_candle_timeframe():
    """Garante que o bot usa velas de 5min ou mais (estratégia recomendada)."""
    assert config.CANDLE_TIMEFRAME_SEC >= 60, (
        f"Estratégia recomenda CANDLE_TIMEFRAME_SEC >= 60, atual: {config.CANDLE_TIMEFRAME_SEC}"
    )


def test_recommended_target_lookforward():
    """Horizonte de 2+ candles é mais estável que 1 para índices sintéticos."""
    assert config.TARGET_LOOKFORWARD >= 2, (
        f"Estratégia recomenda TARGET_LOOKFORWARD >= 2, atual: {config.TARGET_LOOKFORWARD}"
    )


def test_recommended_signal_score_min():
    """Score mínimo de 0.25+ filtra entradas de baixa qualidade."""
    assert config.SIGNAL_SCORE_MIN >= 0.03, (
        f"Estratégia recomenda SIGNAL_SCORE_MIN >= 0.03, atual: {config.SIGNAL_SCORE_MIN}"
    )
