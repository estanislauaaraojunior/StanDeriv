"""Score Bollinger e isolamento dos módulos de decisão, sem rede ou modelos."""
import math

import pytest

import config
import strategy


def weighted(price=100.0, bands=(102.0, 100.0, 98.0), **overrides):
    values = dict(ema9=price, ema21=price, macd_hist=0.0, mom=0.0,
                  adx_val=40.0, rsi_val=50.0, price=price,
                  indicators={}, prices=[price] * 100, use_ai_model=False,
                  apply_aux_filters=False, bollinger_bands=bands)
    values.update(overrides)
    return strategy._weighted_signal(**values)


@pytest.fixture(autouse=True)
def balanced(monkeypatch):
    monkeypatch.setattr(config, "ENTRY_MODE", "balanced")
    monkeypatch.setattr(strategy, "SIGNAL_SCORE_MIN", 0.06)


@pytest.mark.parametrize("price,score,signal", [
    (100, 0, None), (102, 1, "BUY"), (98, -1, "SELL"),
    (104, 1, "BUY"), (96, -1, "SELL"),
    (100.4, .2, None), (99.6, -.2, None),
])
def test_position_clipping_and_entry_threshold(price, score, signal):
    actual, details = weighted(price)
    assert actual == signal
    assert details["bb_score"] == pytest.approx(score)
    assert details["tech_score"] == pytest.approx(.15 * score)


@pytest.mark.parametrize("bands", [
    (100, 100, 100), (98, 100, 102), (102, 103, 98),
    (math.nan, 100, 98), (102, 100, math.inf), None,
])
def test_invalid_bands_block_weighted_module(bands):
    signal, details = weighted(bands=bands)
    assert signal is None
    assert details["decision_block"] == "bollinger_invalid"
    assert "tech_score" not in details


def test_composite_weights_and_adx_multiplier():
    signal, details = weighted(
        price=102, ema9=101, ema21=100, macd_hist=.01,
        mom=1, rsi_val=75, adx_val=30,
    )
    # EMA, preço e momentum saturados; MACD normalizado = .01 / .02.
    expected = (.25 + .20 * .5 + .15 + .15 + .15 + .10 * .5) * .75
    assert signal == "BUY"
    assert details["tech_score"] == pytest.approx(expected, abs=5e-5)


@pytest.mark.parametrize("price,expected", [(102, "BUY"), (98, "SELL")])
def test_threshold_is_inclusive(monkeypatch, price, expected):
    monkeypatch.setattr(strategy, "SIGNAL_SCORE_MIN", .15)
    assert weighted(price)[0] == expected
    monkeypatch.setattr(strategy, "SIGNAL_SCORE_MIN", .15001)
    assert weighted(price)[0] is None


def test_opportunity_adx_floor_is_preserved(monkeypatch):
    monkeypatch.setattr(config, "ENTRY_MODE", "opportunity")
    monkeypatch.setattr(config, "OPPORTUNITY_ADX_WEIGHT_FLOOR", .35)
    monkeypatch.setattr(config, "OPPORTUNITY_SIGNAL_SCORE_MIN", .05)
    signal, details = weighted(price=102, adx_val=0)
    assert signal == "BUY"
    assert details["tech_score"] == .0525


def test_balanced_adx_gate_still_blocks_weighted_score(monkeypatch):
    monkeypatch.setattr(strategy, "USE_WEIGHTED_SIGNAL", True)
    monkeypatch.setattr(strategy, "PRICE_ACTION_FILTER", False)
    monkeypatch.setattr(strategy, "CANDLE_PATTERN_FILTER", False)
    monkeypatch.setattr(strategy.ind, "adx", lambda *a: 29)
    signal, details = strategy.get_signal([100 + i * .05 for i in range(300)], adx_min=30, use_ai_model=False)
    assert signal is None
    assert details["module_details"]["weighted_score"]["decision_block"] == "adx_below_min"


def test_public_path_reuses_unrounded_bands_and_preserves_five_votes(monkeypatch):
    monkeypatch.setattr(strategy, "USE_WEIGHTED_SIGNAL", True)
    monkeypatch.setattr(strategy, "PRICE_ACTION_FILTER", False)
    monkeypatch.setattr(strategy, "CANDLE_PATTERN_FILTER", False)
    prices = [100 + i * .05 for i in range(300)]
    original = strategy.ind.bollinger
    calls = []

    def tracked(*args):
        result = original(*args)
        calls.append(result)
        return result

    monkeypatch.setattr(strategy.ind, "bollinger", tracked)
    signal, details = strategy.get_signal(prices, use_ai_model=False)
    assert signal == "BUY"
    assert len(calls) == 1
    upper, mid, lower = calls[0]
    expected = min(1, (prices[-1] - mid) / ((upper - lower) / 2))
    assert details["bb_score"] == round(expected, 4)
    assert set(details["technical_component_votes"]) == {
        "ema", "macd", "price_vs_ema", "momentum", "rsi",
    }


def test_invalid_bollinger_does_not_block_other_modules(monkeypatch):
    monkeypatch.setattr(strategy, "USE_WEIGHTED_SIGNAL", True)
    monkeypatch.setattr(strategy, "PRICE_ACTION_FILTER", True)
    monkeypatch.setattr(strategy, "CANDLE_PATTERN_FILTER", True)
    monkeypatch.setattr(strategy.ind, "bollinger", lambda *args: (100, 100, 100))
    monkeypatch.setattr(strategy, "_trained_model_signal", lambda *a, **k: ("BUY", {"model_id": "offline"}))
    monkeypatch.setattr(strategy, "_price_action_signal", lambda *a: ("BUY", .5))
    monkeypatch.setattr(strategy, "_candle_pattern_signal", lambda *a: ("BUY", None))
    signal, details = strategy.get_signal([100 + i * .05 for i in range(300)], use_ai_model=True)
    assert signal == "BUY"
    assert details["module_votes"] == {
        "weighted_score": None, "trained_model": "BUY",
        "price_action": "BUY", "candle_pattern": "BUY",
    }
    assert details["module_details"]["weighted_score"]["decision_block"] == "bollinger_invalid"


def test_disabled_weighted_module_does_not_evaluate_new_score(monkeypatch):
    monkeypatch.setattr(strategy, "USE_WEIGHTED_SIGNAL", False)
    monkeypatch.setattr(strategy, "PRICE_ACTION_FILTER", False)
    monkeypatch.setattr(strategy, "CANDLE_PATTERN_FILTER", False)
    monkeypatch.setattr(strategy, "_trained_model_signal", lambda *a, **k: ("SELL", {"model_id": "offline"}))

    def forbidden(*args, **kwargs):
        pytest.fail("Score Ponderado desabilitado foi avaliado")

    monkeypatch.setattr(strategy, "_weighted_signal", forbidden)
    signal, details = strategy.get_signal([100 + i * .05 for i in range(300)], use_ai_model=True)
    assert signal == "SELL"
    assert details["module_votes"] == {"trained_model": "SELL"}
    assert "bb_score" not in details
