"""Testes determinísticos para o motor de backtesting offline."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

os.environ.setdefault("DERIV_TOKEN", "test_token")

import strategy
from backtest import (
    BacktestEngine,
    BacktestSettings,
    HistoricalRiskManager,
    SignalContext,
    Tick,
    load_ticks,
    write_report,
    build_candles,
)


def _ticks(prices, start=1_700_000_040, step=60, symbol="R_TEST"):
    """Converte uma sequência de preços em Ticks espaçados por step segundos."""
    return [
        Tick(epoch=start + index * step, price=float(price), symbol=symbol)
        for index, price in enumerate(prices)
    ]


def _settings(**overrides):
    """Cria BacktestSettings determinísticas (sem IA) com sobrescritas opcionais."""
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
    """Cria um provedor de sinal que retorna BUY uma vez e depois None."""
    used = False

    def provider(_context: SignalContext):
        """Retorna BUY na primeira chamada e None nas seguintes."""
        nonlocal used
        if used:
            return None, {"adx": 20.0}
        used = True
        return "BUY", {"adx": 20.0}

    return provider


def test_load_ticks_accepts_headerless_csv_and_isolates_symbol(tmp_path):
    """Garante que load_ticks lê CSV sem cabeçalho e filtra apenas o símbolo pedido."""
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
    """Garante que load_ticks levanta ValueError para símbolo ausente no CSV."""
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
    """Garante que payout e custos hipotéticos são refletidos no lucro líquido e no saldo final."""
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
    """Garante que empate de preço é contabilizado como perda do stake."""
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
    """Garante que contrato em ticks expira exatamente no tick futuro indicado pela duração."""
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
    """Garante que o stop loss diário bloqueia novas entradas após a perda."""
    def always_buy(_context):
        """Sempre retorna BUY."""
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
    """Garante que a pausa por perdas consecutivas usa o relógio histórico dos ticks."""
    def always_buy(_context):
        """Sempre retorna BUY."""
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
    """Garante que o contexto do sinal não contém preços futuros: séries com mesmo prefixo geram a mesma entrada."""
    seen_a = []
    seen_b = []

    def provider_for(target):
        """Cria provedor que registra os preços vistos em target e compra uma vez."""
        used = False

        def provider(context):
            """Registra os preços do contexto e retorna BUY apenas na primeira chamada."""
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
    """Garante que get_signal com use_ai_model=False não consulta o preditor de IA."""
    prices = [100.0 + index * 0.05 for index in range(200)]
    with patch.object(strategy.ai_predictor, "predict") as predict:
        strategy.get_signal(prices, adx_min=0, use_ai_model=False)
    predict.assert_not_called()


def test_candle_filter_uses_historical_evaluation_time():
    """Garante que o filtro de padrões de vela usa o horário de avaliação e expira alertas antigos."""
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
    """Garante que write_report grava JSON válido com artefatos do modelo e um CSV de trades."""
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


@pytest.mark.parametrize('ratio,expected',[(.5999,0),(.60,1)])
def test_backtest_respects_payout_floor(ratio,expected):
    """Garante que o backtest não opera com payout abaixo de 0.60."""
    engine=BacktestEngine(_ticks([100,101,102,104,105,106,107,108]),'R_TEST',
                         _settings(payout_ratio=ratio),signal_provider=_first_buy_then_none())
    result=engine.run()
    assert len(result.trades)==expected


def _epoch(value):
    """Converte data ISO em epoch UTC."""
    return int(datetime.fromisoformat(value).replace(tzinfo=timezone.utc).timestamp())


@pytest.mark.parametrize("profit,reason", [(-100, "daily_stop_loss"), (100, "daily_take_profit")])
def test_daily_limits_reset_at_explicit_timezone_midnight(profit, reason):
    """Garante que stop e take profit diários bloqueiam e são zerados na meia-noite do fuso configurado."""
    risk = HistoricalRiskManager(_settings(
        risk_timezone="America/Sao_Paulo", stop_loss_pct=.05, take_profit_pct=.05,
    ))
    before = _epoch("2024-01-02T02:59:00")
    risk.settle(before, profit)
    assert risk.can_trade(before) == (False, reason)
    assert risk.current_day.isoformat() == "2024-01-01"
    assert risk.can_trade(before + 60) == (True, "")
    assert risk.current_day.isoformat() == "2024-01-02"
    assert risk.daily_start_balance == 1000 + profit
    assert risk.daily_profit == 0


def test_midnight_preserves_pending_pause():
    """Garante que a virada do dia preserva a pausa pendente por perdas consecutivas."""
    risk = HistoricalRiskManager(_settings(max_consecutive_losses=1, pause_base_sec=180))
    before = _epoch("2024-01-01T23:59:00")
    risk.settle(before, -100)
    assert risk.can_trade(before + 60) == (False, "pause")
    assert risk.daily_profit == 0
    assert risk.consecutive_losses == 0
    assert risk.can_trade(before + 180) == (True, "")


def test_zero_net_result_preserves_loss_streak():
    """Garante que resultado líquido zero não zera a sequência de perdas."""
    risk = HistoricalRiskManager(_settings())
    risk.settle(1_700_000_040, -10)
    risk.settle(1_700_000_041, 0)
    assert risk.consecutive_losses == 1


def test_contract_crossing_midnight_is_charged_to_settlement_day():
    """Garante que contrato que atravessa a meia-noite é contabilizado no dia da liquidação."""
    report = BacktestEngine(
        _ticks([110, 109, 108, 107, 106, 105, 104, 103],
               start=_epoch("2024-01-01T23:57:00")),
        "R_TEST", _settings(stop_loss_pct=.05),
        signal_provider=lambda context: ("BUY", {}),
    ).run()
    assert len(report.trades) == 1
    assert report.trades[0].entry_epoch == _epoch("2024-01-01T23:59:00")
    assert report.trades[0].expiry_epoch == _epoch("2024-01-02T00:00:00")
    assert report.metrics["by_settlement_day"] == {
        "2024-01-02": {"trades": 1, "net_profit": -100.0},
    }
    assert report.metrics["risk_blocks"]["daily_stop_loss"] > 0


def test_global_loss_streak_survives_daily_reset():
    """Garante que a maior sequência de perdas é global e sobrevive à virada diária."""
    report = BacktestEngine(
        _ticks(list(range(120, 108, -1)), start=_epoch("2024-01-01T23:55:00")),
        "R_TEST", _settings(stop_loss_pct=1, max_consecutive_losses=100),
        signal_provider=lambda context: ("BUY", {}),
    ).run()
    assert len(report.metrics["by_settlement_day"]) == 2
    assert report.metrics["max_consecutive_losses"] == len(report.trades)


@pytest.mark.parametrize("zone", ["Invalid/Timezone", "", "../UTC"])
def test_invalid_risk_timezone_is_rejected(zone):
    """Garante que fuso de risco inválido faz validate levantar ValueError."""
    with pytest.raises(ValueError, match="risk_timezone"):
        _settings(risk_timezone=zone).validate()


@pytest.mark.parametrize("direction,prices", [("BUY", [100, 101, 102, 104, 105]),
                                             ("SELL", [105, 104, 103, 102, 101])])
def test_seconds_duration_and_winning_directions(direction, prices):
    """Garante que contratos de 1 segundo vencem em BUY e SELL com expiração um segundo após a entrada."""
    report = BacktestEngine(
        _ticks(prices, step=1), "R_TEST",
        _settings(timeframe_sec=1, duration=1, duration_unit="s"),
        signal_provider=lambda context: (direction, {}),
    ).run()
    assert report.trades[0].outcome == "WIN"
    assert report.trades[0].expiry_epoch == report.trades[0].entry_epoch + 1
    assert report.metrics["max_consecutive_wins"] == len(report.trades)


def test_missing_file_and_insufficient_candles(tmp_path):
    """Garante erros claros para arquivo inexistente e para velas insuficientes."""
    with pytest.raises(FileNotFoundError):
        load_ticks(tmp_path / "missing.csv")
    with pytest.raises(ValueError, match="candles insuficientes"):
        BacktestEngine(_ticks([100, 101], step=1), "R_TEST", _settings()).run()


def test_last_incomplete_candle_is_not_used():
    """Garante que build_candles descarta a última vela incompleta."""
    ticks = _ticks([100, 101, 102, 999])
    candles = build_candles(ticks, 60)
    assert [c.close for c in candles] == [100, 101, 102]


def test_repeated_runs_are_deterministic_and_do_not_touch_operational_files(tmp_path, monkeypatch):
    """Garante que execuções repetidas são idênticas e não tocam arquivos, rede, IA ou risco operacionais."""
    import config
    import socket
    import ai_predictor
    import risk_manager

    monkeypatch.chdir(tmp_path)
    paths = [tmp_path / name for name in (
        "risk_state.json", "state.json", "operacoes_log.csv", "risk_accounts.json",
        "model.pkl", "model_metrics.json", "operations.csv",
    )]
    for path in paths:
        path.write_bytes(b"operational sentinel")
    before = {p: p.read_bytes() for p in paths}
    config_before = (config.get_active_symbol(), config.DURATION, config.USE_AI_MODEL)
    def forbidden(*args, **kwargs):
        """Falha o teste se o backtest acessar recursos operacionais."""
        raise AssertionError("backtest accessed operational resources")
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(ai_predictor, "predict", forbidden)
    monkeypatch.setattr(ai_predictor, "status", forbidden)
    monkeypatch.setattr(risk_manager.RiskManager, "__init__", forbidden)
    engine = BacktestEngine(
        _ticks([100 + i * .01 for i in range(150)], step=1), "R_TEST",
        _settings(timeframe_sec=1, min_candles=100, price_buffer_size=100),
    )
    reports = [engine.run().to_dict() for _ in range(2)]
    for report in reports:
        report.pop("created_at")
    assert reports[0] == reports[1]
    assert reports[0]["model_artifacts"]["ticks_sha256"]
    assert reports[0]["metrics"]["simulation_assumptions"]["risk_timezone"] == "UTC"
    assert {p: p.read_bytes() for p in paths} == before
    assert (config.get_active_symbol(), config.DURATION, config.USE_AI_MODEL) == config_before


def test_cli_help_without_credentials_or_dotenv(tmp_path):
    """Garante que backtest.py --help funciona sem credenciais nem .env."""
    import shutil
    import subprocess
    import sys

    root = Path(__file__).resolve().parent.parent
    for name in ("backtest.py", "config.py", "ai_predictor.py", "strategy.py",
                 "indicators.py", "feature_engine.py", "model_contract.py", "market_data.py"):
        shutil.copy2(root / name, tmp_path / name)
    env = {key: value for key, value in os.environ.items()
           if key not in {"DERIV_TOKEN", "DERIV_APP_ID", "PYTHONPATH"}}
    result = subprocess.run(
        [sys.executable, str(tmp_path / "backtest.py"), "--help"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "--risk-timezone" in result.stdout


@pytest.mark.parametrize('probability,payout,accepted', [
    (.56, .70, False), (.56, .85, True),
    (.625, .60, True), (.6249, .60, False),
    (.45, 1.0, False), (.450001, 1.0, False),
    (.500001, 1.0, True),
    (None, .85, False), (float('nan'), .85, False),
])
def test_ai_backtest_shares_executor_economic_gate(monkeypatch, probability, payout, accepted):
    """Garante que o backtest com IA aceita ou bloqueia propostas pelo mesmo gate econômico do executor."""
    import ai_predictor
    import config
    from model_contract import profitable_proposal
    symbol = config.get_active_symbol()
    monkeypatch.setattr(ai_predictor, 'status', lambda: {'ready': True})
    monkeypatch.setattr(ai_predictor, '_metadata', {'last_seen_epoch': 0})
    report = BacktestEngine(
        _ticks(range(100, 150), symbol=symbol), symbol,
        _settings(use_ai=True, timeframe_sec=config.CANDLE_TIMEFRAME_SEC,
              duration=config.DURATION, duration_unit=config.DURATION_UNIT,
                  payout_ratio=payout),
        signal_provider=lambda ctx: ('BUY', {'win_probability': probability}),
    ).run()
    assert bool(report.trades) is accepted
    assert profitable_proposal(1, 1 + payout, probability) is accepted
    if not accepted:
        assert report.metrics['risk_blocks']['unprofitable_proposal'] > 0


def test_ai_backtest_deducts_hypothetical_costs_before_accepting(monkeypatch):
    """Garante que custos hipotéticos são deduzidos antes de aceitar a proposta no backtest com IA."""
    import ai_predictor
    import config
    symbol = config.get_active_symbol()
    monkeypatch.setattr(ai_predictor, 'status', lambda: {'ready': True})
    monkeypatch.setattr(ai_predictor, '_metadata', {'last_seen_epoch': 0})
    report = BacktestEngine(
        _ticks(range(100, 150), symbol=symbol), symbol,
        _settings(use_ai=True, timeframe_sec=config.CANDLE_TIMEFRAME_SEC,
              duration=config.DURATION, duration_unit=config.DURATION_UNIT,
                  payout_ratio=.60, fixed_cost=.01),
        signal_provider=lambda ctx: ('BUY', {'win_probability': .625}),
    ).run()
    assert not report.trades
    assert report.metrics['risk_blocks']['unprofitable_proposal'] > 0
