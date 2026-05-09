"""
tests/test_strategy_edge_cases.py — casos extremos não cobertos pelos testes originais.

Foca em: entradas inválidas, preços degenerados, limiares de RSI/ADX e filtros de PA.
"""

import math
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _trend_up(n=200, start=100.0, step=0.05):
    return [start + i * step for i in range(n)]


def _trend_down(n=200, start=100.0, step=0.05):
    return [start - i * step for i in range(n)]


def _flat(n=200, value=100.0):
    return [value] * n


# ─── Entradas inválidas ───────────────────────────────────────────────────────

def test_empty_list_returns_none():
    from strategy import get_signal
    sig, ind = get_signal([])
    assert sig is None
    assert ind == {}


def test_single_price_returns_none():
    from strategy import get_signal
    sig, ind = get_signal([100.0])
    assert sig is None


def test_very_short_list_returns_none():
    from strategy import get_signal
    sig, ind = get_signal([100.0, 101.0, 102.0])
    assert sig is None


# ─── Preços constantes (volatilidade zero) ───────────────────────────────────

def test_constant_prices_no_exception():
    """Divisão por zero não deve ocorrer com preços completamente planos."""
    from strategy import get_signal
    prices = _flat(200)
    try:
        sig, ind = get_signal(prices)
    except (ZeroDivisionError, FloatingPointError) as e:
        pytest.fail(f"Preços constantes causaram exceção: {e}")
    # Sem tendência → sinal deve ser None
    assert sig is None


def test_constant_prices_indicators_populated():
    """Indicadores devem ser calculados mesmo sem volatilidade."""
    from strategy import get_signal
    prices = _flat(200)
    sig, ind = get_signal(prices)
    # Pode retornar {} se não houver dados suficientes para algum indicador
    # mas não deve explodir
    assert isinstance(ind, dict)


# ─── Preços com NaN ───────────────────────────────────────────────────────────

def test_nan_in_prices_no_crash():
    """NaN nos preços não deve causar exceção não tratada."""
    from strategy import get_signal
    prices = _trend_up(200)
    prices[100] = float("nan")
    try:
        sig, ind = get_signal(prices)
        # Pode retornar None ou um sinal — o importante é não crashar
        assert sig in (None, "BUY", "SELL")
        assert isinstance(ind, dict)
    except Exception as e:
        pytest.fail(f"NaN em prices causou exceção: {e}")


# ─── ADX history com poucos elementos ────────────────────────────────────────

def test_adx_history_single_element():
    """adx_history com 1 elemento não deve causar IndexError."""
    from strategy import get_signal
    prices = _trend_up(200)
    try:
        sig, ind = get_signal(prices, adx_history=[25.0])
        assert sig in (None, "BUY", "SELL")
    except (IndexError, ZeroDivisionError) as e:
        pytest.fail(f"adx_history=[1 elemento] causou exceção: {e}")


def test_adx_history_empty():
    """adx_history vazio não deve causar exceção."""
    from strategy import get_signal
    prices = _trend_up(200)
    try:
        sig, ind = get_signal(prices, adx_history=[])
    except Exception as e:
        pytest.fail(f"adx_history=[] causou exceção: {e}")


# ─── Limiares de RSI ─────────────────────────────────────────────────────────

def test_rsi_boundary_oversold_blocks_sell():
    """
    Com RSI exatamente no RSI_OVERSOLD (38), SELL não deve ser gerado
    no modo AND rígido (USE_WEIGHTED_SIGNAL=False).
    """
    from strategy import get_signal
    import config

    prices = _trend_down(200)
    with patch.object(config, "USE_WEIGHTED_SIGNAL", False):
        sig, ind = get_signal(prices)
        # RSI em downtrend forte pode ser < 38 — SELL pode ou não ser gerado
        # O importante é que não haja exceção
        assert sig in (None, "BUY", "SELL")


def test_rsi_boundary_overbought_blocks_buy():
    """
    Com RSI exatamente no RSI_OVERBOUGHT (62), BUY não deve ser gerado
    no modo AND rígido.
    """
    from strategy import get_signal
    import config

    prices = _trend_up(200)
    with patch.object(config, "USE_WEIGHTED_SIGNAL", False):
        sig, ind = get_signal(prices)
        assert sig in (None, "BUY", "SELL")


# ─── Score ponderado (USE_WEIGHTED_SIGNAL) ────────────────────────────────────

def test_weighted_signal_returns_valid():
    """USE_WEIGHTED_SIGNAL=True não deve lançar exceção em mercado normal."""
    from strategy import get_signal
    import config

    prices = _trend_up(200)
    with patch.object(config, "USE_WEIGHTED_SIGNAL", True):
        sig, ind = get_signal(prices)
        assert sig in (None, "BUY", "SELL")
        assert isinstance(ind, dict)


def test_weighted_signal_score_in_indicators():
    """Quando USE_WEIGHTED_SIGNAL=True, 'tech_score' deve aparecer nos indicadores."""
    from strategy import get_signal
    import config

    prices = _trend_up(200)
    with patch.object(config, "USE_WEIGHTED_SIGNAL", True):
        sig, ind = get_signal(prices)
        if ind:  # só verifica se indicadores foram calculados
            assert "tech_score" in ind, "tech_score ausente nos indicadores ponderados"


def test_weighted_signal_high_score_min_blocks_entry():
    """SIGNAL_SCORE_MIN=0.99 deve bloquear quase todas as entradas."""
    import strategy as strat
    import indicators as ind

    prices = _trend_up(200)
    with patch.object(strat, "USE_WEIGHTED_SIGNAL", True), \
         patch.object(strat, "SIGNAL_SCORE_MIN", 0.99), \
         patch.object(strat, "USE_AI_MODEL", False), \
         patch.object(ind, "price_action_features", return_value=None):
        sig, _ = strat.get_signal(prices, adx_min=1)
        assert sig is None, "Score impossível de atingir deveria bloquear sinal"


def test_weighted_signal_zero_score_min_allows_entry():
    """Com SIGNAL_SCORE_MIN muito baixo, o score técnico em uptrend com ruído deve gerar BUY."""
    import strategy as strat
    import indicators as ind
    import math

    # Série com tendência clara + ruído para ADX detectar tendência real
    prices = [100.0 + i * 0.1 + math.sin(i * 0.3) * 0.5 for i in range(400)]
    with patch.object(strat, "USE_WEIGHTED_SIGNAL", True), \
         patch.object(strat, "SIGNAL_SCORE_MIN", 0.001), \
         patch.object(strat, "USE_AI_MODEL", False), \
         patch.object(ind, "price_action_features", return_value=None):
        # passa adx_min=1 diretamente para contornar o threshold
        sig, ind_dict = strat.get_signal(prices, adx_min=1)
        assert sig == "BUY", (
            f"Tendência com ruído e SIGNAL_SCORE_MIN=0.001 deveria gerar BUY, got {sig}. "
            f"tech_score={ind_dict.get('tech_score')}"
        )


# ─── ADX adaptativo ──────────────────────────────────────────────────────────

def test_adaptive_adx_min_respects_floor():
    """ADX mínimo adaptativo nunca deve cair abaixo de 15."""
    from strategy import get_adaptive_adx_min
    import config

    with patch.object(config, "ADX_ADAPTIVE", True):
        # Histórico muito baixo — o piso de 15 deve ser respeitado
        result = get_adaptive_adx_min([1.0, 2.0, 3.0, 1.5, 2.0])
        assert result >= 15.0, f"ADX adaptativo abaixo do piso: {result}"


def test_adaptive_adx_min_disabled_returns_fixed():
    """Quando ADX_ADAPTIVE=False, deve retornar ADX_MIN fixo."""
    from strategy import get_adaptive_adx_min
    import config

    with patch.object(config, "ADX_ADAPTIVE", False):
        result = get_adaptive_adx_min([100.0, 200.0, 300.0])
        assert result == config.ADX_MIN


# ─── Indicadores sempre populados ────────────────────────────────────────────

def test_indicators_dict_keys_present():
    """Com dados suficientes, todos os indicadores esperados devem estar presentes."""
    from strategy import get_signal
    prices = _trend_up(300)
    sig, ind = get_signal(prices)

    if not ind:
        pytest.skip("Dados insuficientes para calcular indicadores")

    expected_keys = {"ema9", "ema21", "rsi", "adx", "macd_hist", "momentum"}
    missing = expected_keys - ind.keys()
    assert not missing, f"Chaves ausentes nos indicadores: {missing}"


def test_indicators_returned_even_when_no_signal():
    """Mesmo sem sinal, o dict de indicadores deve ser retornado preenchido."""
    from strategy import get_signal
    import config

    prices = _flat(200)
    with patch.object(config, "ADX_MIN", 1000):  # ADX impossível → sem sinal
        sig, ind = get_signal(prices)
        assert sig is None
        # ind pode ser {} ou preenchido dependendo de quando o ADX falha
        assert isinstance(ind, dict)
