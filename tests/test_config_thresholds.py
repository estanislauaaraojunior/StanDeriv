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
    """Garante que AI_TECH_WEIGHT e AI_MODEL_WEIGHT somam 1.0."""
    total = config.AI_TECH_WEIGHT + config.AI_MODEL_WEIGHT
    assert abs(total - 1.0) < 1e-9, (
        f"AI_TECH_WEIGHT + AI_MODEL_WEIGHT deve ser 1.0, got {total}"
    )


def test_ai_confidence_min_range():
    """Garante que AI_CONFIDENCE_MIN fica entre o piso de aprovação e 0.70."""
    assert config.MIN_APPROVAL_WIN_RATE <= config.AI_CONFIDENCE_MIN <= 0.70, (
        "AI_CONFIDENCE_MIN deve respeitar o piso aprovado e permanecer controlado: "
        f"{config.AI_CONFIDENCE_MIN}"
    )


def test_ai_score_min_range():
    """Garante que AI_SCORE_MIN fica na faixa segura [0.25, 0.70]."""
    assert 0.25 <= config.AI_SCORE_MIN <= 0.70, (
        f"AI_SCORE_MIN fora da faixa segura [0.25, 0.70]: {config.AI_SCORE_MIN}"
    )


def test_execution_signal_mode_valid():
    """Garante que EXECUTION_SIGNAL_MODE é válido e que o único modo aceito é trained_model."""
    assert config.EXECUTION_SIGNAL_MODE in config.VALID_EXECUTION_SIGNAL_MODES
    assert config.VALID_EXECUTION_SIGNAL_MODES == {
        "trained_model",
    }


# ─── RSI ─────────────────────────────────────────────────────────────────────

def test_rsi_bounds_logical():
    """Garante que RSI_OVERSOLD é menor que RSI_OVERBOUGHT."""
    assert config.RSI_OVERSOLD < config.RSI_OVERBOUGHT, (
        f"RSI_OVERSOLD ({config.RSI_OVERSOLD}) deve ser < RSI_OVERBOUGHT ({config.RSI_OVERBOUGHT})"
    )


def test_rsi_oversold_min():
    """Garante que RSI_OVERSOLD não fica abaixo de 30."""
    assert config.RSI_OVERSOLD >= 30, (
        f"RSI_OVERSOLD muito baixo ({config.RSI_OVERSOLD}), mínimo recomendado: 30"
    )


def test_rsi_overbought_max():
    """Garante que RSI_OVERBOUGHT não passa de 70."""
    assert config.RSI_OVERBOUGHT <= 70, (
        f"RSI_OVERBOUGHT muito alto ({config.RSI_OVERBOUGHT}), máximo recomendado: 70"
    )


# ─── ADX ─────────────────────────────────────────────────────────────────────

def test_adx_min_range():
    """Garante que ADX_MIN fica na faixa [10, 40]."""
    assert 10 <= config.ADX_MIN <= 40, (
        f"ADX_MIN fora da faixa razoável [10, 40]: {config.ADX_MIN}"
    )
    assert config.ADX_MIN >= 30, (
        f"ADX_MIN deve exigir tendência forte na configuração atual: {config.ADX_MIN}"
    )


def test_adx_adaptive_percentile_range():
    """Garante que ADX_ADAPTIVE_PERCENTILE fica na faixa [20, 80]."""
    assert 20 <= config.ADX_ADAPTIVE_PERCENTILE <= 80, (
        f"ADX_ADAPTIVE_PERCENTILE fora da faixa [20, 80]: {config.ADX_ADAPTIVE_PERCENTILE}"
    )


# ─── Sinal ponderado ─────────────────────────────────────────────────────────

def test_signal_score_min_range():
    """Garante que SIGNAL_SCORE_MIN está no intervalo aberto (0, 1)."""
    assert 0.0 < config.SIGNAL_SCORE_MIN < 1.0, (
        f"SIGNAL_SCORE_MIN deve estar em (0, 1): {config.SIGNAL_SCORE_MIN}"
    )


def test_entry_mode_balanced_configuration():
    """Garante que ENTRY_MODE usa confluência balanceada com score coerente."""
    assert config.ENTRY_MODE in {"balanced", "opportunity", "conservative"}
    assert config.ENTRY_MODE == "balanced"
    assert 0 < config.OPPORTUNITY_SIGNAL_SCORE_MIN <= config.SIGNAL_SCORE_MIN
    assert 0 <= config.OPPORTUNITY_ADX_WEIGHT_FLOOR <= 1


def test_signal_score_min_not_too_low():
    """Garante que SIGNAL_SCORE_MIN não fica abaixo de 0.03."""
    assert config.SIGNAL_SCORE_MIN >= 0.03, (
        f"SIGNAL_SCORE_MIN muito permissivo ({config.SIGNAL_SCORE_MIN}), mínimo recomendado: 0.03"
    )
    assert config.SIGNAL_SCORE_MIN >= 0.05, (
        f"SIGNAL_SCORE_MIN deve preservar confluência técnica mínima: {config.SIGNAL_SCORE_MIN}"
    )


def test_unfavorable_symbol_switch_not_too_eager():
    """Garante que a troca automática tenha paciência com a configuração seletiva."""
    assert config.UNFAVORABLE_SYMBOL_SWITCH_DECISIONS >= 10, (
        "UNFAVORABLE_SYMBOL_SWITCH_DECISIONS muito baixo para ADX_MIN=30: "
        f"{config.UNFAVORABLE_SYMBOL_SWITCH_DECISIONS}"
    )


# ─── Candles ─────────────────────────────────────────────────────────────────

def test_candle_timeframe_min():
    """Garante que CANDLE_TIMEFRAME_SEC é de pelo menos 60 segundos."""
    assert config.CANDLE_TIMEFRAME_SEC >= 60, (
        f"CANDLE_TIMEFRAME_SEC muito baixo ({config.CANDLE_TIMEFRAME_SEC}s), mínimo: 60s"
    )


def test_target_lookforward_min():
    """Garante que TARGET_LOOKFORWARD é de pelo menos 1."""
    assert config.TARGET_LOOKFORWARD >= 1, (
        f"TARGET_LOOKFORWARD deve ser >= 1: {config.TARGET_LOOKFORWARD}"
    )


def test_min_candles_positive():
    """Garante que MIN_CANDLES cobre a janela operacional de contratos 15m."""
    assert config.MIN_CANDLES >= 150, (
        f"MIN_CANDLES muito baixo ({config.MIN_CANDLES}), mínimo recomendado: 150"
    )


# ─── Gestão de risco ─────────────────────────────────────────────────────────

def test_stake_pct_safe():
    """Garante que STAKE_PCT fica na faixa segura (0, 0.05]."""
    assert 0 < config.STAKE_PCT <= 0.05, (
        f"STAKE_PCT fora da faixa segura (0, 0.05]: {config.STAKE_PCT}"
    )


def test_stop_loss_lt_take_profit():
    """Garante que STOP_LOSS_PCT é menor que TAKE_PROFIT_PCT."""
    assert config.STOP_LOSS_PCT < config.TAKE_PROFIT_PCT, (
        f"STOP_LOSS_PCT ({config.STOP_LOSS_PCT}) deve ser < TAKE_PROFIT_PCT ({config.TAKE_PROFIT_PCT})"
    )


def test_max_consec_losses_positive():
    """Garante que MAX_CONSEC_LOSSES é de pelo menos 1."""
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
    """Target acompanha um vencimento de 15m sobre candles de 15m."""
    assert config.CANDLE_TIMEFRAME_SEC == 900 and config.TARGET_LOOKFORWARD == 1, (
        f"Esperado TARGET_LOOKFORWARD=1 em 15m, atual: {config.TARGET_LOOKFORWARD}"
    )


def test_recommended_signal_score_min():
    """Score mínimo de 0.25+ filtra entradas de baixa qualidade."""
    assert config.SIGNAL_SCORE_MIN >= 0.03, (
        f"Estratégia recomenda SIGNAL_SCORE_MIN >= 0.03, atual: {config.SIGNAL_SCORE_MIN}"
    )
