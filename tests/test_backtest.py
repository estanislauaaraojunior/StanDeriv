"""Testes determinísticos para o motor de backtesting offline."""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

os.environ.setdefault("DERIV_TOKEN", "test_token")

import strategy
from backtest import (
    BacktestEngine,
    BacktestSettings,
    SignalContext,
    Tick,
    load_ticks,
    write_report,
)


def _ticks(prices, start=1_700_000_040, step=60, symbol="R_TEST"):
    return [
        Tick(epoch=start + index * step, price=float(price), symbol=symbol)
        for index, price in enumerate(prices)
    ]


def _settings(**overrides):
    values = {
        "initial_balance": 1000.0,
        "payout_ratio": 0.8,
        "stake_pct": 0.1,
        "timeframe_sec": 60,
        "min_candles": 2,
        "price_buffer_size": 20,
        "entry_candle_interval": 1,
        "duration": 1,
        "duration_unit": "m",
        "use_ai": False,
        "dynamic_duration": False,
        "use_candle_patterns": False,
    }
    values.update(overrides)
    return BacktestSettings(**values)


def _first_buy_then_none():
    used = False

    def provider(_context: SignalContext):
        nonlocal used
        if used:
            return None, {"adx": 20.0}
        used = True
        return "BUY", {"adx": 20.0}

    return provider


def test_load_ticks_accepts_headerless_csv_and_isolates_symbol(tmp_path):
    source = tmp_path / "ticks.csv"
    source.write_text(
        "1700000000,2023-11-14T00:00:00,R_10,100.0\n"
        "1700000001,2023-11-14T00:00:01,R_25,200.0\n"
        "1700000002,2023-11-14T00:00:02,R_10,101.0\n",
        encoding="utf-8",
    )

    ticks, symbol, quality = load_ticks(source, symbol="R_10")

    assert symbol == "R_10"
    assert [tick.price for tick in ticks] == [100.0, 101.0]
    assert quality["symbols_found"] == {"R_10": 2, "R_25": 1}


def test_load_ticks_rejects_unknown_symbol(tmp_path):
    source = tmp_path / "ticks.csv"
    source.write_text(
        "epoch,datetime,symbol,price\n"
        "1700000000,x,R_10,100.0\n"
        "1700000001,x,R_10,101.0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="não encontrado"):
        load_ticks(source, symbol="R_100")


def test_payout_and_costs_are_applied_to_balance():
    settings = _settings(cost_rate=0.01, fixed_cost=1.0)
    engine = BacktestEngine(
        _ticks([100, 101, 102, 104, 105, 106, 107, 108]),
        "R_TEST",
        settings,
        signal_provider=_first_buy_then_none(),
    )

    report = engine.run()
    trade = report.trades[0]

    assert trade.stake == 100.0
    assert trade.gross_profit == 80.0
    assert trade.costs == 2.0
    assert trade.net_profit == 78.0
    assert report.metrics["final_balance"] == 1078.0
    assert report.metrics["return_pct"] == 7.8


def test_tie_is_a_contract_loss():
    engine = BacktestEngine(
        _ticks([100, 101, 102, 102, 103, 104, 105, 106]),
        "R_TEST",
        _settings(),
        signal_provider=_first_buy_then_none(),
    )

    report = engine.run()

    assert report.trades[0].outcome == "TIE"
    assert report.trades[0].net_profit == -100.0
    assert report.metrics["ties"] == 1
    assert report.metrics["losses"] == 1


def test_tick_duration_uses_exact_future_tick_offset():
    engine = BacktestEngine(
        _ticks([100, 101, 102, 99, 105, 106, 107, 108], step=1),
        "R_TEST",
        _settings(timeframe_sec=1, duration=2, duration_unit="t"),
        signal_provider=_first_buy_then_none(),
    )

    trade = engine.run().trades[0]

    assert trade.entry_price == 102.0
    assert trade.exit_price == 105.0
    assert trade.expiry_epoch - trade.entry_epoch == 2


def test_daily_stop_blocks_new_entries_after_loss():
    def always_buy(_context):
        return "BUY", {"adx": 20.0}

    engine = BacktestEngine(
        _ticks([110, 109, 108, 107, 106, 105, 104, 103, 102, 101]),
        "R_TEST",
        _settings(stop_loss_pct=0.05),
        signal_provider=always_buy,
    )

    report = engine.run()

    assert report.metrics["trades"] == 1
    assert report.metrics["risk_blocks"]["daily_stop_loss"] > 0
    assert report.metrics["max_drawdown_pct"] == 10.0


def test_pause_uses_historical_clock_after_consecutive_losses():
    def always_buy(_context):
        return "BUY", {"adx": 20.0}

    prices = list(range(130, 100, -1))
    engine = BacktestEngine(
        _ticks(prices),
        "R_TEST",
        _settings(
            stop_loss_pct=1.0,
            max_consecutive_losses=2,
            pause_base_sec=180,
        ),
        signal_provider=always_buy,
    )

    report = engine.run()

    assert report.metrics["trades"] >= 2
    assert report.metrics["risk_blocks"]["pause"] >= 2
    assert report.metrics["max_consecutive_losses"] >= 2


def test_signal_context_never_contains_future_prices():
    seen_a = []
    seen_b = []

    def provider_for(target):
        used = False

        def provider(context):
            nonlocal used
            target.append(list(context.prices))
            if not used:
                used = True
                return "BUY", {"adx": 20.0}
            return None, {"adx": 20.0}

        return provider

    common_prefix = [100, 101, 102]
    report_a = BacktestEngine(
        _ticks(common_prefix + [110, 111, 112, 113, 114]),
        "R_TEST",
        _settings(),
        signal_provider=provider_for(seen_a),
    ).run()
    report_b = BacktestEngine(
        _ticks(common_prefix + [90, 89, 88, 87, 86]),
        "R_TEST",
        _settings(),
        signal_provider=provider_for(seen_b),
    ).run()

    assert seen_a[0] == seen_b[0] == [100.0, 101.0]
    assert report_a.trades[0].entry_price == report_b.trades[0].entry_price == 102.0
    assert report_a.trades[0].outcome == "WIN"
    assert report_b.trades[0].outcome == "LOSS"


def test_technical_mode_does_not_call_ai_predictor():
    prices = [100.0 + index * 0.05 for index in range(200)]
    with patch.object(strategy.ai_predictor, "predict") as predict:
        strategy.get_signal(prices, adx_min=0, use_ai_model=False)
    predict.assert_not_called()


def test_candle_filter_uses_historical_evaluation_time():
    alert = {
        "name": "Bearish Engulfing",
        "direction": "bearish",
        "strength": 0.9,
        "timestamp": 1_000,
    }

    recent = strategy._candle_pattern_filter(
        "BUY",
        [alert],
        evaluation_time=1_200,
    )
    expired = strategy._candle_pattern_filter(
        "BUY",
        [alert],
        evaluation_time=1_400,
    )

    assert recent is not None
    assert expired is None


def test_report_is_written_as_valid_json_and_csv(tmp_path):
    report = BacktestEngine(
        _ticks([100, 101, 102, 103, 104, 105, 106, 107]),
        "R_TEST",
        _settings(),
        signal_provider=_first_buy_then_none(),
    ).run(source="fixture.csv")

    json_path, csv_path = write_report(report, tmp_path)

    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["metrics"]["trades"] == 1
    assert loaded["source"] == "fixture.csv"
    assert len(loaded["model_artifacts"]["feature_schema"]) == 27
    assert loaded["model_artifacts"]["strategy_sha256"]
    assert len(csv_path.read_text(encoding="utf-8").splitlines()) == 2
