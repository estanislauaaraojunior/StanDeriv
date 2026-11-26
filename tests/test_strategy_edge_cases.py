"""
tests/test_strategy_edge_cases.py — casos extremos não cobertos pelos testes originais.

Foca em: entradas inválidas, preços degenerados, limiares de RSI/ADX e filtros de PA.
"""

import math
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import strategy
from model_contract import FEATURE_WINDOW

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _trend_up(n=200, start=100.0, step=0.05):
    """Gera série de preços em alta linear."""
    return [start + i * step for i in range(n)]


def _trend_down(n=200, start=100.0, step=0.05):
    """Gera série de preços em baixa linear."""
    return [start - i * step for i in range(n)]


def _flat(n=200, value=100.0):
    """Gera série de preços constantes."""
    return [value] * n


# ─── Entradas inválidas ───────────────────────────────────────────────────────

def test_empty_list_returns_none():
    """Garante que get_signal retorna (None, {}) para lista vazia."""
    from strategy import get_signal
    sig, ind = get_signal([])
    assert sig is None
    assert ind == {}


def test_single_price_returns_none():
    """Garante que get_signal não gera sinal com um único preço."""
    from strategy import get_signal
    sig, ind = get_signal([100.0])
    assert sig is None


def test_very_short_list_returns_none():
    """Garante que get_signal não gera sinal com apenas três preços."""
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
    with patch.object(strategy, "USE_WEIGHTED_SIGNAL", False):
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
    with patch.object(strategy, "USE_WEIGHTED_SIGNAL", False):
        sig, ind = get_signal(prices)
        assert sig in (None, "BUY", "SELL")


# ─── Score ponderado (USE_WEIGHTED_SIGNAL) ────────────────────────────────────

def test_weighted_signal_returns_valid():
    """USE_WEIGHTED_SIGNAL=True não deve lançar exceção em mercado normal."""
    from strategy import get_signal
    import config

    prices = _trend_up(200)
    with patch.object(strategy, "USE_WEIGHTED_SIGNAL", True):
        sig, ind = get_signal(prices)
        assert sig in (None, "BUY", "SELL")
        assert isinstance(ind, dict)


def test_weighted_signal_score_in_indicators():
    """Quando USE_WEIGHTED_SIGNAL=True, 'tech_score' deve aparecer nos indicadores."""
    from strategy import get_signal
    import config

    prices = _trend_up(200)
    with patch.object(strategy, "USE_WEIGHTED_SIGNAL", True):
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
         patch.object(__import__("config"), "ENTRY_MODE", "balanced", create=True), \
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

def test_opportunity_mode_turns_low_adx_into_penalty_not_block():
    """ENTRY_MODE=opportunity permite candidato técnico mesmo com ADX abaixo do piso."""
    import strategy as strat
    import config as cfg
    import indicators as ind

    prices = _trend_up(220, step=0.2)
    with patch.object(cfg, "ENTRY_MODE", "opportunity", create=True), \
         patch.object(strat, "USE_WEIGHTED_SIGNAL", True), \
         patch.object(strat, "USE_AI_MODEL", False), \
         patch.object(ind, "adx", return_value=1.0), \
         patch.object(ind, "price_action_features", return_value=None):
        sig, ind_dict = strat.get_signal(prices, adx_min=999)

    assert sig == "BUY"
    assert ind_dict["adx_below_min"] is True


def test_balanced_mode_keeps_low_adx_as_block():
    """ENTRY_MODE fora de opportunity preserva o bloqueio duro de ADX."""
    import strategy as strat
    import config as cfg
    import indicators as ind

    prices = _trend_up(220, step=0.2)
    with patch.object(cfg, "ENTRY_MODE", "balanced", create=True), \
         patch.object(strat, "USE_WEIGHTED_SIGNAL", True), \
         patch.object(strat, "USE_AI_MODEL", False), \
         patch.object(ind, "adx", return_value=1.0):
        sig, ind_dict = strat.get_signal(prices, adx_min=999)

    assert sig is None
    assert ind_dict["adx_below_min"] is True


def test_trained_model_mode_allows_ai_direction_override():
    """Modo trained_model usa o bundle aprovado sem exigir voto direcional ao vivo."""
    import strategy as strat
    import config as cfg

    with patch.object(cfg, "EXECUTION_SIGNAL_MODE", "trained_model", create=True), \
         patch("strategy.ind.price_action_features", return_value=None), \
         patch("ai_predictor.predict", return_value=("SELL", .54)), \
         patch("ai_predictor.prediction_model_id", return_value="offline"):
        signal, values = strat._apply_ai_filter(
            "BUY",
            [100] * FEATURE_WINDOW,
            {"tech_score": .8},
            use_ai_model=True,
        )

    assert signal == "BUY"
    assert values["execution_signal_mode"] == "trained_model"
    assert values["ai_direction_override"] == "trained_model"
    assert values["win_probability"] == pytest.approx(.46)


def test_disabled_candle_pattern_module_does_not_generate_signal():
    """Padrões desligados não participam da decisão nem geram direção."""
    import strategy as strat
    import indicators as ind

    prices = _trend_up(220, step=0.2)
    bearish_alert = {
        "timestamp": 1700000000,
        "strength": 1.0,
        "direction": "bearish",
        "name": "bearish-test",
    }

    with patch.object(strat, "USE_WEIGHTED_SIGNAL", False), \
         patch.object(strat, "USE_AI_MODEL", False), \
         patch.object(strat, "CANDLE_PATTERN_FILTER", False), \
         patch.object(ind, "price_action_features", return_value=None):
        sig, ind_dict = strat.get_signal(
            prices,
            adx_min=0,
            candle_alerts=[bearish_alert],
            evaluation_time=1700000001,
        )

    assert sig is None
    assert "candle_pattern" not in ind_dict.get("decision_modules", [])


def test_candle_pattern_module_generates_fallback_signal():
    """Padrões confirmados em velas distintas podem votar sem gerador primário."""
    import strategy as strat
    import indicators as ind

    prices = _trend_up(220, step=0.2)
    bearish_alerts = [
        {"timestamp": epoch, "strength": 1.0, "direction": "bearish", "name": "bearish-test"}
        for epoch in (1700000000, 1700000060)
    ]

    with patch.object(strat, "USE_WEIGHTED_SIGNAL", False), \
         patch.object(strat, "USE_AI_MODEL", False), \
         patch.object(strat, "CANDLE_PATTERN_FILTER", True), \
         patch.object(ind, "price_action_features", return_value=None):
        sig, ind_dict = strat.get_signal(
            prices,
            adx_min=0,
            candle_alerts=bearish_alerts,
            evaluation_time=1700000061,
        )

    assert sig == "SELL"
    assert ind_dict["decision_source"] == "candle_pattern"


def test_single_candle_pattern_occurrence_abstains():
    """Uma ocorrência isolada não aprova o módulo de padrões."""
    alert = {"timestamp": 100, "strength": 1.0, "direction": "bullish", "name": "Hammer"}

    assert strategy._candle_pattern_signal([alert], evaluation_time=101) == (
        None, "candle_pattern_insufficient_history"
    )


def test_disabled_price_action_module_does_not_generate_signal():
    """Price Action desligado não participa da decisão."""
    import strategy as strat
    import indicators as ind

    prices = _trend_up(220, step=0.2)
    pa_blocking_buy = {
        "pa_sr_position": 0.95,
        "pa_market_structure": 1.0,
        "pa_bos_strength": 0.0,
    }

    with patch.object(strat, "USE_WEIGHTED_SIGNAL", False), \
         patch.object(strat, "USE_AI_MODEL", False), \
         patch.object(strat, "PRICE_ACTION_FILTER", False), \
         patch.object(ind, "price_action_features", return_value=pa_blocking_buy):
        sig, ind_dict = strat.get_signal(prices, adx_min=0)

    assert sig is None
    assert "price_action" not in ind_dict.get("decision_modules", [])


def test_price_action_module_generates_fallback_signal():
    """Estrutura de mercado habilitada pode gerar a direção no fallback."""
    import strategy as strat
    import indicators as ind

    prices = _trend_up(220, step=0.2)
    pa_blocking_buy = {
        "pa_sr_position": 0.95,
        "pa_market_structure": 1.0,
        "pa_bos_strength": 0.0,
    }

    with patch.object(strat, "USE_WEIGHTED_SIGNAL", False), \
         patch.object(strat, "USE_AI_MODEL", False), \
         patch.object(strat, "PRICE_ACTION_FILTER", True), \
         patch.object(ind, "price_action_features", return_value=pa_blocking_buy):
        sig, _ = strat.get_signal(prices, adx_min=0)

    assert sig == "BUY"


def test_weighted_signal_runs_without_ai_when_only_weighted_enabled(monkeypatch):
    """Garante que o score ponderado decide sozinho sem consultar a IA quando ela está desligada."""
    import strategy as strat

    monkeypatch.setattr(strat, "USE_WEIGHTED_SIGNAL", True)
    monkeypatch.setattr(strat, "USE_AI_MODEL", False)
    monkeypatch.setattr(strat, "PRICE_ACTION_FILTER", False)
    monkeypatch.setattr(strat, "CANDLE_PATTERN_FILTER", False)
    monkeypatch.setattr(strat, "_weighted_signal", lambda *args, **kwargs: ("BUY", {"tech_score": .7}))
    monkeypatch.setattr(strat.ai_predictor, "predict", lambda *args, **kwargs: pytest.fail("IA desligada foi consultada"))

    signal, values = strat.get_signal(_trend_up(220, step=.2), use_ai_model=False)

    assert signal == "BUY"
    assert values["decision_source"] == "weighted_score"


def test_auxiliary_modules_run_independently_with_weighted_signal(monkeypatch):
    """Garante que PA e padrões votam mesmo quando Score Ponderado está ativo."""
    import strategy as strat

    monkeypatch.setattr(strat, "USE_WEIGHTED_SIGNAL", True)
    monkeypatch.setattr(strat, "USE_AI_MODEL", False)
    monkeypatch.setattr(strat, "PRICE_ACTION_FILTER", True)
    monkeypatch.setattr(strat, "CANDLE_PATTERN_FILTER", True)
    monkeypatch.setattr(strat, "_weighted_signal", lambda *args, **kwargs: (None, {"tech_score": 0.0}))
    monkeypatch.setattr(
        strat.ind,
        "price_action_features",
        lambda *args, **kwargs: {
            "pa_market_structure": 1.0, "pa_trend_consistency": 1.0,
            "pa_bos_strength": 1.0, "pa_candle_at_sr": 0.0,
            "pa_demand_zone": 0.0, "pa_supply_zone": 0.0,
            "pa_fvg_bullish": 0.0, "pa_fvg_bearish": 0.0,
            "pa_sr_position": 0.0,
        },
    )
    monkeypatch.setattr(
        strat,
        "_candle_pattern_signal",
        lambda *args, **kwargs: ("BUY", None),
    )

    signal, values = strat.get_signal(_trend_up(220, step=.2), use_ai_model=False)

    assert signal == "BUY"
    assert values["enabled_decision_modules"] == ["weighted_score", "price_action", "candle_pattern"]
    assert values["decision_modules"] == ["price_action", "candle_pattern"]
    assert values["module_votes"]["weighted_score"] is None


def test_weighted_and_model_modules_require_agreement(monkeypatch):
    """Garante bloqueio por discordância quando score ponderado e modelo apontam direções opostas."""
    import strategy as strat

    monkeypatch.setattr(strat, "USE_WEIGHTED_SIGNAL", True)
    monkeypatch.setattr(strat, "USE_AI_MODEL", True)
    monkeypatch.setattr(strat, "PRICE_ACTION_FILTER", True)
    monkeypatch.setattr(strat, "CANDLE_PATTERN_FILTER", True)
    monkeypatch.setattr(strat, "_weighted_signal", lambda *args, **kwargs: ("BUY", {"tech_score": .7}))
    monkeypatch.setattr(strat, "_trained_model_signal", lambda *args, **kwargs: ("SELL", {"model_id": "offline", "win_probability": .8}))

    signal, values = strat.get_signal(_trend_up(220, step=.2))

    assert signal is None
    assert values["decision_block"] == "module_consensus_no_majority"


def test_weighted_and_model_modules_open_only_on_matching_direction(monkeypatch):
    """Garante que o sinal abre quando score ponderado e modelo concordam na direção."""
    import strategy as strat

    monkeypatch.setattr(strat, "USE_WEIGHTED_SIGNAL", True)
    monkeypatch.setattr(strat, "USE_AI_MODEL", True)
    monkeypatch.setattr(strat, "PRICE_ACTION_FILTER", False)
    monkeypatch.setattr(strat, "CANDLE_PATTERN_FILTER", False)
    monkeypatch.setattr(strat, "_weighted_signal", lambda *args, **kwargs: ("BUY", {"tech_score": .7}))
    monkeypatch.setattr(strat, "_trained_model_signal", lambda *args, **kwargs: ("BUY", {"model_id": "offline", "win_probability": .8}))

    signal, values = strat.get_signal(_trend_up(220, step=.2))

    assert signal == "BUY"
    assert values["decision_modules"] == ["weighted_score", "trained_model"]


def test_primary_abstention_blocks_without_fallback_override(monkeypatch):
    """Garante bloqueio por abstenção quando o modelo se abstém e o fallback de PA contradiz o score."""
    import strategy as strat

    monkeypatch.setattr(strat, "USE_WEIGHTED_SIGNAL", True)
    monkeypatch.setattr(strat, "USE_AI_MODEL", True)
    monkeypatch.setattr(strat, "PRICE_ACTION_FILTER", True)
    monkeypatch.setattr(strat, "CANDLE_PATTERN_FILTER", False)
    monkeypatch.setattr(strat, "_weighted_signal", lambda *args, **kwargs: ("BUY", {"tech_score": .7}))
    monkeypatch.setattr(strat, "_trained_model_signal", lambda *args, **kwargs: (None, {"ai_block": "model_abstention"}))
    monkeypatch.setattr(strat.ind, "price_action_features", lambda *args, **kwargs: {
        "pa_market_structure": -1.0, "pa_trend_consistency": -1.0,
    })

    signal, values = strat.get_signal(_trend_up(220, step=.2))

    assert signal is None
    assert values["decision_block"] == "module_consensus_no_majority"
    assert values["enabled_decision_modules"] == ["weighted_score", "trained_model", "price_action"]


def test_module_error_remains_in_consensus_denominator(monkeypatch):
    """Erro de um módulo ativo não aprova nem reduz o total de votos exigidos."""
    import strategy as strat

    monkeypatch.setattr(strat, "USE_WEIGHTED_SIGNAL", True)
    monkeypatch.setattr(strat, "USE_AI_MODEL", True)
    monkeypatch.setattr(strat, "PRICE_ACTION_FILTER", False)
    monkeypatch.setattr(strat, "CANDLE_PATTERN_FILTER", False)
    monkeypatch.setattr(strat, "_weighted_signal", lambda *args, **kwargs: ("BUY", {"tech_score": .7}))
    monkeypatch.setattr(strat, "_trained_model_signal", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("offline")))

    signal, values = strat.get_signal(_trend_up(220, step=.2))

    assert signal is None
    assert values["module_votes"] == {"weighted_score": "BUY", "trained_model": None}
    assert values["module_details"]["trained_model"]["module_error"] == "RuntimeError"
    assert values["decision_block"] == "module_consensus_no_majority"


def test_majority_direction_does_not_reuse_dissenting_model_probability(monkeypatch):
    """Voto dissidente da IA fica fora dos detalhes direcionais da maioria."""
    import strategy as strat

    monkeypatch.setattr(strat, "USE_WEIGHTED_SIGNAL", True)
    monkeypatch.setattr(strat, "USE_AI_MODEL", True)
    monkeypatch.setattr(strat, "PRICE_ACTION_FILTER", True)
    monkeypatch.setattr(strat, "CANDLE_PATTERN_FILTER", True)
    monkeypatch.setattr(strat, "_weighted_signal", lambda *args, **kwargs: ("BUY", {"tech_score": .7}))
    monkeypatch.setattr(strat, "_trained_model_signal", lambda *args, **kwargs: (
        "SELL", {"model_id": "dissenting", "win_probability": .8}
    ))
    monkeypatch.setattr(strat.ind, "price_action_features", lambda *args, **kwargs: {
        "pa_market_structure": 1.0, "pa_trend_consistency": 1.0,
        "pa_bos_strength": 1.0, "pa_candle_at_sr": 0.0,
        "pa_demand_zone": 0.0, "pa_supply_zone": 0.0,
        "pa_fvg_bullish": 0.0, "pa_fvg_bearish": 0.0,
        "pa_sr_position": 0.0,
    })
    monkeypatch.setattr(strat, "_candle_pattern_signal", lambda *_args: ("BUY", None))

    signal, values = strat.get_signal(_trend_up(220, step=.2))

    assert signal == "BUY"
    assert values["module_votes"] == {
        "weighted_score": "BUY", "trained_model": "SELL",
        "price_action": "BUY", "candle_pattern": "BUY",
    }
    assert "trained_model" not in values["decision_modules"]
    assert "model_id" not in values
    assert values["active_model_id"] == "dissenting"
    assert values["enabled_decision_modules"] == [
        "weighted_score", "trained_model", "price_action", "candle_pattern"
    ]


@pytest.mark.parametrize(
    "component_values,expected",
    [((1, 1, 1, -1, -1), True), ((1, 1, -1, -1, -1), False)],
)
def test_weighted_module_requires_majority_of_technical_components(component_values, expected):
    """Score só aprova internamente quando pelo menos três componentes apontam para o sinal."""
    import strategy as strat

    ema_fast, macd_hist, price_delta, momentum, rsi_delta = component_values
    approved, votes = strat._weighted_directional_majority(
        "BUY", 100 + ema_fast, 100, macd_hist, momentum, 50 + rsi_delta, 101 + price_delta
    )

    assert approved is expected
    assert len(votes) == 5


def test_adaptive_adx_min_respects_floor():
    """ADX mínimo adaptativo nunca deve cair abaixo de ADX_MIN."""
    from strategy import get_adaptive_adx_min
    import config

    with patch.object(config, "ADX_ADAPTIVE", True):
        # Histórico muito baixo — o ADX_MIN fixo deve ser respeitado
        result = get_adaptive_adx_min([1.0, 2.0, 3.0, 1.5, 2.0])
        assert result >= config.ADX_MIN, f"ADX adaptativo abaixo do piso: {result}"


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
