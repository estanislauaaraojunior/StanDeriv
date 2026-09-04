#!/usr/bin/env python3
"""Backtesting offline do StanDeriv, sem conexão com a API da Deriv.

O motor reproduz o fluxo operacional relevante: fecha uma vela de tempo,
calcula o sinal apenas com o histórico disponível, entra no primeiro tick da
vela seguinte e liquida o contrato no primeiro tick que alcança a expiração.

Por segurança contra leakage, modelos de IA e duração dinâmica ficam
desabilitados por padrão. Habilite-os somente ao testar um período que não foi
usado no treinamento dos artefatos persistidos.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import ai_predictor
import indicators as ind
import strategy
from config import (
    AI_MODEL_PATH,
    CANDIDATE_DURATIONS,
    CANDLE_PATTERN_MAX_AGE_SEC,
    CANDLE_TIMEFRAME_SEC,
    DURATION,
    DURATION_MODEL_PATH,
    DURATION_UNIT,
    ENTRY_CANDLE_INTERVAL,
    MAX_CONSEC_LOSSES,
    MIN_CANDLES,
    PAUSE_BASE_SEC,
    PAUSE_SCALE_FACTOR,
    PRICE_BUFFER_SIZE,
    PRIORITIZE_DURATION,
    STAKE_PCT,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    TICKS_CSV,
    TRANSFORMER_MODEL_PATH,
    get_active_symbol,
)
from feature_engine import FEATURES


@dataclass(frozen=True)
class Tick:
    epoch: int
    price: float
    symbol: str


@dataclass(frozen=True)
class Candle:
    epoch: int
    open: float
    high: float
    low: float
    close: float
    first_tick_index: int


@dataclass(frozen=True)
class SignalContext:
    prices: list[float]
    adx_min: float
    adx_history: list[float]
    candle_alerts: list[dict]
    evaluation_epoch: int


@dataclass
class BacktestSettings:
    initial_balance: float = 1000.0
    payout_ratio: float = 0.85
    cost_rate: float = 0.0
    fixed_cost: float = 0.0
    stake_pct: float = STAKE_PCT
    min_stake: float = 0.35
    stop_loss_pct: float = STOP_LOSS_PCT
    take_profit_pct: float = TAKE_PROFIT_PCT
    max_consecutive_losses: int = MAX_CONSEC_LOSSES
    pause_base_sec: int = PAUSE_BASE_SEC
    pause_scale_factor: float = PAUSE_SCALE_FACTOR
    timeframe_sec: int = CANDLE_TIMEFRAME_SEC
    min_candles: int = MIN_CANDLES
    price_buffer_size: int = PRICE_BUFFER_SIZE
    entry_candle_interval: int = ENTRY_CANDLE_INTERVAL
    duration: int = (
        PRIORITIZE_DURATION
        if PRIORITIZE_DURATION in CANDIDATE_DURATIONS
        else DURATION
    )
    duration_unit: str = DURATION_UNIT
    use_ai: bool = False
    dynamic_duration: bool = False
    use_candle_patterns: bool = True

    def validate(self) -> None:
        if self.initial_balance <= 0:
            raise ValueError("initial_balance deve ser positivo")
        if not 0 <= self.payout_ratio <= 10:
            raise ValueError("payout_ratio deve estar entre 0 e 10")
        if self.cost_rate < 0 or self.fixed_cost < 0:
            raise ValueError("custos não podem ser negativos")
        if not 0 < self.stake_pct <= 1:
            raise ValueError("stake_pct deve estar em (0, 1]")
        if self.min_stake <= 0:
            raise ValueError("min_stake deve ser positivo")
        if not 0 < self.stop_loss_pct <= 1:
            raise ValueError("stop_loss_pct deve estar em (0, 1]")
        if not 0 < self.take_profit_pct:
            raise ValueError("take_profit_pct deve ser positivo")
        if self.max_consecutive_losses < 1:
            raise ValueError("max_consecutive_losses deve ser positivo")
        if self.pause_base_sec < 0 or self.pause_scale_factor < 1:
            raise ValueError("configuração de pausa inválida")
        if self.timeframe_sec < 1 or self.min_candles < 1:
            raise ValueError("timeframe_sec e min_candles devem ser positivos")
        if self.price_buffer_size < self.min_candles:
            raise ValueError("price_buffer_size deve comportar o aquecimento")
        if self.entry_candle_interval < 1:
            raise ValueError("entry_candle_interval deve ser positivo")
        if self.duration < 1:
            raise ValueError("duration deve ser positiva")
        if self.duration_unit not in {"m", "s", "t"}:
            raise ValueError("duration_unit deve ser 'm', 's' ou 't'")


@dataclass
class TradeRecord:
    trade_id: int
    symbol: str
    direction: str
    signal_epoch: int
    entry_epoch: int
    expiry_epoch: int
    entry_price: float
    exit_price: float
    duration: int
    duration_unit: str
    stake: float
    payout_ratio: float
    gross_profit: float
    costs: float
    net_profit: float
    outcome: str
    balance_before: float
    balance_after: float
    indicators: dict = field(default_factory=dict)


@dataclass
class BacktestReport:
    created_at: str
    source: str
    symbol: str
    settings: dict
    data_quality: dict
    model_artifacts: dict
    metrics: dict
    trades: list[TradeRecord]

    def to_dict(self, include_trades: bool = True) -> dict:
        result = {
            "created_at": self.created_at,
            "source": self.source,
            "symbol": self.symbol,
            "settings": self.settings,
            "data_quality": self.data_quality,
            "model_artifacts": self.model_artifacts,
            "metrics": self.metrics,
        }
        if include_trades:
            result["trades"] = [asdict(trade) for trade in self.trades]
        return result


SignalProvider = Callable[[SignalContext], tuple[Optional[str], dict]]
DurationProvider = Callable[[SignalContext], int]


class HistoricalRiskManager:
    """Espelho determinístico das regras de risco, guiado pelo tempo histórico."""

    def __init__(self, settings: BacktestSettings) -> None:
        self.settings = settings
        self.balance = settings.initial_balance
        self.daily_start_balance = settings.initial_balance
        self.daily_profit = 0.0
        self.current_day = None
        self.consecutive_losses = 0
        self.max_consecutive_losses_seen = 0
        self.pause_until_epoch = 0

    def _advance_day(self, epoch: int) -> None:
        day = datetime.fromtimestamp(epoch, timezone.utc).date()
        if day != self.current_day:
            self.current_day = day
            self.daily_start_balance = self.balance
            self.daily_profit = 0.0
            self.consecutive_losses = 0

    def get_stake(self) -> float:
        return max(round(self.balance * self.settings.stake_pct, 2), self.settings.min_stake)

    def can_trade(self, epoch: int) -> tuple[bool, str]:
        self._advance_day(epoch)
        if epoch < self.pause_until_epoch:
            return False, "pause"
        daily_pct = (
            self.daily_profit / self.daily_start_balance
            if self.daily_start_balance > 0
            else 0.0
        )
        if daily_pct <= -self.settings.stop_loss_pct:
            return False, "daily_stop_loss"
        if daily_pct >= self.settings.take_profit_pct:
            return False, "daily_take_profit"
        stake = self.get_stake()
        costs = stake * self.settings.cost_rate + self.settings.fixed_cost
        if stake + costs > self.balance:
            return False, "insufficient_balance"
        return True, ""

    def settle(self, epoch: int, net_profit: float) -> tuple[float, float]:
        self._advance_day(epoch)
        balance_before = self.balance
        self.balance = round(self.balance + net_profit, 10)
        self.daily_profit = round(self.daily_profit + net_profit, 10)

        if net_profit < 0:
            self.consecutive_losses += 1
            self.max_consecutive_losses_seen = max(
                self.max_consecutive_losses_seen,
                self.consecutive_losses,
            )
            if self.consecutive_losses >= self.settings.max_consecutive_losses:
                extra = self.consecutive_losses - self.settings.max_consecutive_losses
                pause = int(
                    self.settings.pause_base_sec
                    * (self.settings.pause_scale_factor ** extra)
                )
                self.pause_until_epoch = epoch + min(pause, 7200)
        else:
            self.consecutive_losses = 0

        return balance_before, self.balance


def _parse_tick_row(row: list[str]) -> Optional[Tick]:
    if not row:
        return None
    first = row[0].strip().lower()
    if first in {"epoch", "timestamp", "time"}:
        return None
    try:
        if len(row) >= 4:
            epoch = int(float(row[0]))
            symbol = row[2].strip()
            price = float(row[3])
        elif len(row) >= 2:
            epoch = int(float(row[0]))
            symbol = ""
            price = float(row[1])
        else:
            return None
    except (TypeError, ValueError):
        return None
    if epoch <= 0 or not math.isfinite(price) or price <= 0:
        return None
    return Tick(epoch=epoch, price=price, symbol=symbol)


def load_ticks(
    path: str | Path,
    symbol: Optional[str] = None,
    max_ticks: Optional[int] = None,
) -> tuple[list[Tick], str, dict]:
    """Carrega CSV com ou sem cabeçalho e isola um único símbolo."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"arquivo de ticks não encontrado: {source}")

    parsed: list[Tick] = []
    invalid_rows = 0
    with source.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.reader(handle):
            tick = _parse_tick_row(row)
            if tick is None:
                if row and row[0].strip().lower() not in {"epoch", "timestamp", "time"}:
                    invalid_rows += 1
                continue
            parsed.append(tick)

    if not parsed:
        raise ValueError("nenhum tick válido encontrado")

    counts = Counter(tick.symbol for tick in parsed if tick.symbol)
    preferred = symbol or get_active_symbol()
    if counts:
        if symbol and symbol not in counts:
            raise ValueError(f"símbolo '{symbol}' não encontrado no arquivo")
        selected = preferred if preferred in counts else counts.most_common(1)[0][0]
        selected_ticks = [tick for tick in parsed if tick.symbol == selected]
    else:
        selected = symbol or preferred
        selected_ticks = [
            Tick(tick.epoch, tick.price, selected) for tick in parsed
        ]

    selected_ticks.sort(key=lambda tick: tick.epoch)
    if max_ticks is not None:
        if max_ticks < 2:
            raise ValueError("max_ticks deve ser pelo menos 2")
        selected_ticks = selected_ticks[-max_ticks:]

    if len(selected_ticks) < 2:
        raise ValueError("são necessários pelo menos dois ticks do símbolo")

    quality = {
        "rows_valid": len(parsed),
        "rows_invalid": invalid_rows,
        "ticks_selected": len(selected_ticks),
        "symbols_found": dict(sorted(counts.items())),
        "first_epoch": selected_ticks[0].epoch,
        "last_epoch": selected_ticks[-1].epoch,
    }
    return selected_ticks, selected, quality


def build_candles(ticks: list[Tick], timeframe_sec: int) -> list[Candle]:
    """Agrega com indicators.py e associa cada vela ao seu primeiro tick."""
    raw = [{"epoch": tick.epoch, "price": tick.price} for tick in ticks]
    aggregated = ind.ticks_to_candles_by_time(raw, timeframe_sec)
    first_tick_by_bucket: dict[int, int] = {}
    for index, tick in enumerate(ticks):
        bucket = (tick.epoch // timeframe_sec) * timeframe_sec
        first_tick_by_bucket.setdefault(bucket, index)

    return [
        Candle(
            epoch=int(item["epoch"]),
            open=float(item["open"]),
            high=float(item["high"]),
            low=float(item["low"]),
            close=float(item["close"]),
            first_tick_index=first_tick_by_bucket[int(item["epoch"])],
        )
        for item in aggregated
        if int(item["epoch"]) in first_tick_by_bucket
    ]


def _resolve_expiry_index(
    ticks: list[Tick],
    epochs: list[int],
    entry_index: int,
    duration: int,
    duration_unit: str,
) -> Optional[int]:
    if duration_unit == "t":
        expiry_index = entry_index + duration
    else:
        multiplier = 60 if duration_unit == "m" else 1
        target_epoch = ticks[entry_index].epoch + duration * multiplier
        expiry_index = bisect.bisect_left(epochs, target_epoch, lo=entry_index + 1)
    return expiry_index if expiry_index < len(ticks) else None


def _artifact_fingerprint(path: str) -> Optional[str]:
    artifact = Path(path)
    if not artifact.exists() or not artifact.is_file():
        return None
    digest = hashlib.sha256()
    with artifact.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_iso(epoch: int) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


class BacktestEngine:
    def __init__(
        self,
        ticks: list[Tick],
        symbol: str,
        settings: Optional[BacktestSettings] = None,
        signal_provider: Optional[SignalProvider] = None,
        duration_provider: Optional[DurationProvider] = None,
    ) -> None:
        self.ticks = ticks
        self.symbol = symbol
        self.settings = settings or BacktestSettings()
        self.settings.validate()
        self.signal_provider = signal_provider or self._default_signal
        self.duration_provider = duration_provider or self._default_duration

    def _default_signal(self, context: SignalContext) -> tuple[Optional[str], dict]:
        return strategy.get_signal(
            context.prices,
            adx_min=context.adx_min,
            adx_history=context.adx_history,
            candle_alerts=context.candle_alerts,
            use_ai_model=self.settings.use_ai,
            evaluation_time=context.evaluation_epoch,
        )

    def _default_duration(self, context: SignalContext) -> int:
        if self.settings.dynamic_duration:
            return ai_predictor.predict_duration(context.prices)
        return self.settings.duration

    def run(self, source: str = "") -> BacktestReport:
        candles = build_candles(self.ticks, self.settings.timeframe_sec)
        if len(candles) <= self.settings.min_candles:
            raise ValueError(
                "candles insuficientes para aquecimento e ao menos uma entrada: "
                f"{len(candles)} <= {self.settings.min_candles}"
            )

        risk = HistoricalRiskManager(self.settings)
        epochs = [tick.epoch for tick in self.ticks]
        closes: list[float] = []
        adx_history: deque[float] = deque(maxlen=500)
        alerts: deque[dict] = deque(maxlen=20)
        trades: list[TradeRecord] = []
        pending: Optional[TradeRecord] = None
        candles_since_last_entry = self.settings.entry_candle_interval
        signals_generated = 0
        skipped_without_expiry = 0
        blocked_reasons: Counter[str] = Counter()
        equity_curve = [self.settings.initial_balance]

        def settle_pending() -> None:
            nonlocal pending
            if pending is None:
                return
            before, after = risk.settle(pending.expiry_epoch, pending.net_profit)
            pending.balance_before = round(before, 10)
            pending.balance_after = round(after, 10)
            trades.append(pending)
            equity_curve.append(after)
            pending = None

        for index, candle in enumerate(candles):
            closes.append(candle.close)
            entry_tick_index = bisect.bisect_left(
                epochs,
                candle.epoch + self.settings.timeframe_sec,
                lo=candle.first_tick_index + 1,
            )
            if entry_tick_index >= len(self.ticks):
                continue
            evaluation_epoch = self.ticks[entry_tick_index].epoch

            if pending is not None and evaluation_epoch >= pending.expiry_epoch:
                settle_pending()

            self._update_candle_alerts(
                candles[max(0, index - 99) : index + 1],
                alerts,
                evaluation_epoch,
            )

            if len(closes) < self.settings.min_candles or pending is not None:
                continue

            allowed, reason = risk.can_trade(evaluation_epoch)
            if not allowed:
                blocked_reasons[reason] += 1
                continue

            candles_since_last_entry += 1
            adaptive_interval = self.settings.entry_candle_interval
            if risk.consecutive_losses >= 2:
                adaptive_interval += 1
            if candles_since_last_entry < adaptive_interval:
                continue

            prices = closes[-self.settings.price_buffer_size :]
            adx_min = strategy.get_adaptive_adx_min(list(adx_history))
            context = SignalContext(
                prices=list(prices),
                adx_min=adx_min,
                adx_history=list(adx_history),
                candle_alerts=list(alerts),
                evaluation_epoch=evaluation_epoch,
            )
            signal, indicators = self.signal_provider(context)
            if indicators.get("adx") is not None:
                adx_history.append(float(indicators["adx"]))
            if signal not in {"BUY", "SELL"}:
                continue

            signals_generated += 1
            duration = int(self.duration_provider(context))
            if duration < 1:
                raise ValueError("provedor de duração retornou valor inválido")
            expiry_index = _resolve_expiry_index(
                self.ticks,
                epochs,
                entry_tick_index,
                duration,
                self.settings.duration_unit,
            )
            if expiry_index is None:
                skipped_without_expiry += 1
                continue

            stake = risk.get_stake()
            entry_tick = self.ticks[entry_tick_index]
            expiry_tick = self.ticks[expiry_index]
            won = (
                expiry_tick.price > entry_tick.price
                if signal == "BUY"
                else expiry_tick.price < entry_tick.price
            )
            tied = expiry_tick.price == entry_tick.price
            gross_profit = stake * self.settings.payout_ratio if won else -stake
            costs = stake * self.settings.cost_rate + self.settings.fixed_cost
            net_profit = gross_profit - costs
            pending = TradeRecord(
                trade_id=len(trades) + 1,
                symbol=self.symbol,
                direction=signal,
                signal_epoch=evaluation_epoch,
                entry_epoch=entry_tick.epoch,
                expiry_epoch=expiry_tick.epoch,
                entry_price=entry_tick.price,
                exit_price=expiry_tick.price,
                duration=duration,
                duration_unit=self.settings.duration_unit,
                stake=round(stake, 10),
                payout_ratio=self.settings.payout_ratio,
                gross_profit=round(gross_profit, 10),
                costs=round(costs, 10),
                net_profit=round(net_profit, 10),
                outcome="TIE" if tied else ("WIN" if won else "LOSS"),
                balance_before=round(risk.balance, 10),
                balance_after=round(risk.balance, 10),
                indicators=dict(indicators),
            )
            candles_since_last_entry = 0

        if pending is not None:
            settle_pending()

        metrics = self._calculate_metrics(
            trades,
            equity_curve,
            signals_generated,
            skipped_without_expiry,
            blocked_reasons,
            risk,
        )
        artifacts = {
            "strategy_sha256": _artifact_fingerprint(strategy.__file__),
            "feature_schema": list(FEATURES),
        }
        if self.settings.use_ai:
            artifacts["direction_model_sha256"] = _artifact_fingerprint(AI_MODEL_PATH)
            artifacts["transformer_model_sha256"] = _artifact_fingerprint(
                TRANSFORMER_MODEL_PATH
            )
        if self.settings.dynamic_duration:
            artifacts["duration_model_sha256"] = _artifact_fingerprint(
                DURATION_MODEL_PATH
            )

        return BacktestReport(
            created_at=datetime.now(timezone.utc).isoformat(),
            source=source,
            symbol=self.symbol,
            settings=asdict(self.settings),
            data_quality={
                "ticks": len(self.ticks),
                "candles_closed": len(candles),
                "first_tick": _utc_iso(self.ticks[0].epoch),
                "last_tick": _utc_iso(self.ticks[-1].epoch),
            },
            model_artifacts=artifacts,
            metrics=metrics,
            trades=trades,
        )

    def _update_candle_alerts(
        self,
        candles: list[Candle],
        alerts: deque[dict],
        evaluation_epoch: int,
    ) -> None:
        if not self.settings.use_candle_patterns or len(candles) < 3:
            return
        raw = [
            {
                "open": candle.open,
                "high": candle.high,
                "low": candle.low,
                "close": candle.close,
            }
            for candle in candles[-100:]
        ]
        for pattern in ind.detect_candle_patterns(raw):
            if float(pattern.get("strength", 0.0)) < 0.5:
                continue
            alerts.append(
                {
                    "name": pattern["name"],
                    "direction": pattern["direction"],
                    "strength": float(pattern["strength"]),
                    "timestamp": evaluation_epoch,
                }
            )
        while alerts and (
            evaluation_epoch - int(alerts[0]["timestamp"])
            > CANDLE_PATTERN_MAX_AGE_SEC
        ):
            alerts.popleft()

    def _calculate_metrics(
        self,
        trades: list[TradeRecord],
        equity_curve: list[float],
        signals_generated: int,
        skipped_without_expiry: int,
        blocked_reasons: Counter[str],
        risk: HistoricalRiskManager,
    ) -> dict:
        wins = sum(trade.outcome == "WIN" for trade in trades)
        losses = sum(trade.outcome in {"LOSS", "TIE"} for trade in trades)
        gross_wins = sum(max(trade.net_profit, 0.0) for trade in trades)
        gross_losses = abs(sum(min(trade.net_profit, 0.0) for trade in trades))
        net_profit = sum(trade.net_profit for trade in trades)

        peak = equity_curve[0]
        max_drawdown_abs = 0.0
        max_drawdown_pct = 0.0
        for equity in equity_curve:
            peak = max(peak, equity)
            drawdown = peak - equity
            drawdown_pct = drawdown / peak if peak > 0 else 0.0
            max_drawdown_abs = max(max_drawdown_abs, drawdown)
            max_drawdown_pct = max(max_drawdown_pct, drawdown_pct)

        durations = Counter(str(trade.duration) for trade in trades)
        directions = Counter(trade.direction for trade in trades)
        n_trades = len(trades)
        return {
            "trades": n_trades,
            "wins": wins,
            "losses": losses,
            "ties": sum(trade.outcome == "TIE" for trade in trades),
            "win_rate_pct": round(wins / n_trades * 100, 4) if n_trades else 0.0,
            "initial_balance": round(self.settings.initial_balance, 10),
            "final_balance": round(risk.balance, 10),
            "net_profit": round(net_profit, 10),
            "return_pct": round(
                net_profit / self.settings.initial_balance * 100,
                4,
            ),
            "gross_wins": round(gross_wins, 10),
            "gross_losses": round(gross_losses, 10),
            "profit_factor": (
                round(gross_wins / gross_losses, 6)
                if gross_losses > 0
                else None
            ),
            "expectancy": round(net_profit / n_trades, 10) if n_trades else 0.0,
            "total_staked": round(sum(trade.stake for trade in trades), 10),
            "total_costs": round(sum(trade.costs for trade in trades), 10),
            "max_drawdown_abs": round(max_drawdown_abs, 10),
            "max_drawdown_pct": round(max_drawdown_pct * 100, 4),
            "max_consecutive_losses": risk.max_consecutive_losses_seen,
            "signals_generated": signals_generated,
            "signals_without_expiry": skipped_without_expiry,
            "risk_blocks": dict(sorted(blocked_reasons.items())),
            "trades_by_direction": dict(sorted(directions.items())),
            "trades_by_duration": dict(sorted(durations.items())),
        }


def write_report(report: BacktestReport, output_dir: str | Path) -> tuple[Path, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "backtest_report.json"
    csv_path = destination / "backtest_trades.csv"
    json_path.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    fields = [field.name for field in TradeRecord.__dataclass_fields__.values()]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for trade in report.trades:
            row = asdict(trade)
            row["indicators"] = json.dumps(row["indicators"], ensure_ascii=False)
            writer.writerow(row)
    return json_path, csv_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backtesting offline do StanDeriv")
    parser.add_argument("--input", default=TICKS_CSV, help="CSV de ticks")
    parser.add_argument("--symbol", help="símbolo isolado; padrão: ativo ou mais frequente")
    parser.add_argument("--output-dir", default="backtest_reports")
    parser.add_argument("--balance", type=float, default=1000.0)
    parser.add_argument("--payout", type=float, default=0.85, help="lucro líquido por stake em uma vitória")
    parser.add_argument("--cost-rate", type=float, default=0.0, help="custo proporcional por contrato")
    parser.add_argument("--fixed-cost", type=float, default=0.0, help="custo fixo por contrato")
    parser.add_argument(
        "--duration",
        type=int,
        default=(
            PRIORITIZE_DURATION
            if PRIORITIZE_DURATION in CANDIDATE_DURATIONS
            else DURATION
        ),
    )
    parser.add_argument("--duration-unit", choices=("m", "s", "t"), default=DURATION_UNIT)
    parser.add_argument("--timeframe", type=int, default=CANDLE_TIMEFRAME_SEC, help="vela em segundos")
    parser.add_argument("--min-candles", type=int, default=MIN_CANDLES)
    parser.add_argument("--max-ticks", type=int)
    parser.add_argument("--with-ai", action="store_true", help="habilita modelos; exige período fora da amostra")
    parser.add_argument("--dynamic-duration", action="store_true", help="habilita modelo de duração; exige período fora da amostra")
    parser.add_argument("--no-candle-patterns", action="store_true")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    settings = BacktestSettings(
        initial_balance=args.balance,
        payout_ratio=args.payout,
        cost_rate=args.cost_rate,
        fixed_cost=args.fixed_cost,
        timeframe_sec=args.timeframe,
        min_candles=args.min_candles,
        duration=args.duration,
        duration_unit=args.duration_unit,
        use_ai=args.with_ai,
        dynamic_duration=args.dynamic_duration,
        use_candle_patterns=not args.no_candle_patterns,
    )
    try:
        ticks, symbol, quality = load_ticks(args.input, args.symbol, args.max_ticks)
        engine = BacktestEngine(ticks, symbol, settings)
        report = engine.run(source=str(Path(args.input).resolve()))
        report.data_quality.update(quality)
        json_path, csv_path = write_report(report, args.output_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[BACKTEST] Erro: {exc}")
        return 1

    metrics = report.metrics
    print(f"[BACKTEST] Símbolo: {symbol} | Trades: {metrics['trades']}")
    print(
        f"[BACKTEST] Win rate: {metrics['win_rate_pct']:.2f}% | "
        f"Retorno: {metrics['return_pct']:+.2f}% | "
        f"Drawdown máx.: {metrics['max_drawdown_pct']:.2f}%"
    )
    print(f"[BACKTEST] Relatório JSON: {json_path}")
    print(f"[BACKTEST] Operações CSV: {csv_path}")
    if args.with_ai or args.dynamic_duration:
        print(
            "[BACKTEST] Atenção: métricas com modelos só são válidas se este "
            "período não participou do treinamento."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
