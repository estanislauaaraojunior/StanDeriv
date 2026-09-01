"""
backtester.py — motor de backtesting offline do StanDeriv.

Simula a operação do bot sobre uma série histórica de ticks (ticks.csv gerado
por collector.py), reutilizando os mesmos componentes de decisão do fluxo ao
vivo — sem qualquer conexão com a Deriv:

  - indicators.ticks_to_candles_by_time  → agrega ticks em velas de tempo
    (mesma agregação usada por executor.py e dataset_builder.py)
  - strategy.get_signal / get_adaptive_adx_min → geração de sinais
    (inclui o filtro de IA quando USE_AI_MODEL=True, igual ao bot ao vivo)
  - ai_predictor.predict_duration        → duração dinâmica do contrato
  - risk_manager.RiskManager             → stake, stops, pausas e log de risco

IMPORTANTE — limitações do backtest:
  - O payout é uma estimativa fixa (BACKTEST_PAYOUT_RATIO em config.py); a
    Deriv só informa o payout real por proposal ao vivo, que varia por ativo,
    duração e condições de mercado.
  - Não modela slippage, requotes, latência de rede ou indisponibilidade de
    duração/ativo — fatores que existem na execução real (ver executor.py).
  - Resultados de backtest são apenas uma referência RELATIVA para comparar
    estratégias/modelos entre si. Não garantem lucro, acurácia ou retorno
    em operação ao vivo.

Uso via CLI:
    python backtester.py --ticks ticks.csv --balance 1000
    python backtester.py --ticks ticks.csv --balance 1000 --compare-ai
    python backtester.py --ticks ticks.csv --balance 1000 --output backtest_reports/run1.json

Uso programático:
    from backtester import run_backtest
    result = run_backtest("ticks.csv", initial_balance=1000.0)
    print(result.summary())
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

import indicators as ind
import risk_manager as rm_mod
import strategy as strat_mod
import ai_predictor
from risk_manager import RiskManager
from strategy import get_signal, get_adaptive_adx_min
from dataset_builder import _resolve_active_symbol
from config import (
    TICKS_CSV, SYMBOL, CANDLE_TIMEFRAME_SEC,
    MIN_CANDLES, ENTRY_CANDLE_INTERVAL,
    DURATION, DURATION_UNIT,
    BACKTEST_PAYOUT_RATIO, BACKTEST_REPORTS_DIR,
)

# Mesmo tamanho de buffer de candles usado pelo bot ao vivo (executor.py)
_CANDLE_BUFFER_MAXLEN = 500
_ADX_HISTORY_MAXLEN = 500


@dataclass
class Trade:
    """Registro de uma operação simulada."""
    entry_epoch: int
    exit_epoch: int
    direction: str
    entry_price: float
    exit_price: float
    stake: float
    duration: int
    duration_unit: str
    profit: float
    result: str          # "WIN" | "LOSS"
    balance_after: float


@dataclass
class BacktestResult:
    """Resultado agregado de um backtest."""
    symbol: str
    initial_balance: float
    final_balance: float
    trades: list = field(default_factory=list)   # list[Trade]
    candles_used: int = 0
    use_ai_model: bool = False

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def wins(self) -> int:
        return sum(1 for t in self.trades if t.result == "WIN")

    @property
    def losses(self) -> int:
        return sum(1 for t in self.trades if t.result == "LOSS")

    @property
    def win_rate_pct(self) -> float:
        if not self.trades:
            return 0.0
        return round(self.wins / len(self.trades) * 100, 2)

    @property
    def total_return_pct(self) -> float:
        if self.initial_balance <= 0:
            return 0.0
        return round((self.final_balance - self.initial_balance) / self.initial_balance * 100, 2)

    @property
    def profit_factor(self) -> float:
        gross_win = sum(t.profit for t in self.trades if t.profit > 0)
        gross_loss = abs(sum(t.profit for t in self.trades if t.profit < 0))
        if gross_loss == 0:
            return round(gross_win, 2) if gross_win > 0 else 0.0
        return round(gross_win / gross_loss, 2)

    @property
    def max_drawdown_pct(self) -> float:
        """Maior queda percentual do saldo em relação ao pico anterior."""
        peak = self.initial_balance
        max_dd = 0.0
        for t in self.trades:
            peak = max(peak, t.balance_after)
            if peak > 0:
                dd = (peak - t.balance_after) / peak * 100
                max_dd = max(max_dd, dd)
        return round(max_dd, 2)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "use_ai_model": self.use_ai_model,
            "candles_used": self.candles_used,
            "initial_balance": self.initial_balance,
            "final_balance": self.final_balance,
            "total_trades": self.total_trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate_pct": self.win_rate_pct,
            "total_return_pct": self.total_return_pct,
            "profit_factor": self.profit_factor,
            "max_drawdown_pct": self.max_drawdown_pct,
            "trades": [
                {
                    "entry_epoch": t.entry_epoch, "exit_epoch": t.exit_epoch,
                    "direction": t.direction,
                    "entry_price": t.entry_price, "exit_price": t.exit_price,
                    "stake": t.stake, "duration": t.duration,
                    "duration_unit": t.duration_unit,
                    "profit": t.profit, "result": t.result,
                    "balance_after": t.balance_after,
                }
                for t in self.trades
            ],
        }

    def summary(self) -> str:
        return (
            f"[BACKTEST] symbol={self.symbol} ai_model={self.use_ai_model} "
            f"candles={self.candles_used}\n"
            f"  Trades       : {self.total_trades} (W:{self.wins} / L:{self.losses})\n"
            f"  Win rate     : {self.win_rate_pct:.2f}%\n"
            f"  Saldo        : {self.initial_balance:.2f} -> {self.final_balance:.2f} "
            f"({self.total_return_pct:+.2f}%)\n"
            f"  Profit factor: {self.profit_factor:.2f}\n"
            f"  Max drawdown : {self.max_drawdown_pct:.2f}%\n"
            "  (estimativa offline — payout fixo, sem slippage; não garante resultado ao vivo)"
        )


def _load_candles(ticks_path: str, symbol: str = None) -> tuple[list, list, str]:
    """
    Lê ticks.csv e agrega em velas de CANDLE_TIMEFRAME_SEC segundos, na mesma
    granularidade usada pelo bot ao vivo (executor.py / dataset_builder.py).

    Retorna (closes, epochs, símbolo_resolvido).
    """
    if not os.path.exists(ticks_path):
        raise FileNotFoundError(
            f"Arquivo de ticks não encontrado: '{ticks_path}'. "
            "Execute collector.py primeiro para gerar histórico."
        )

    df = pd.read_csv(ticks_path)
    active_symbol = symbol or _resolve_active_symbol(df)

    if "symbol" in df.columns:
        df = df[df["symbol"] == active_symbol]

    if "price" in df.columns:
        prices_raw = df["price"].astype(float).tolist()
    elif df.shape[1] == 2:
        prices_raw = df.iloc[:, 1].astype(float).tolist()
    else:
        prices_raw = df.iloc[:, 3].astype(float).tolist()

    if "epoch" in df.columns:
        epochs_raw = df["epoch"].astype(int).tolist()
    else:
        epochs_raw = list(range(len(prices_raw)))

    if len(epochs_raw) < 2:
        return [], [], active_symbol

    if epochs_raw[-1] < epochs_raw[0]:
        pairs = sorted(zip(epochs_raw, prices_raw), key=lambda x: x[0])
        epochs_raw = [e for e, _ in pairs]
        prices_raw = [p for _, p in pairs]

    ticks_list = [{"epoch": e, "price": p} for e, p in zip(epochs_raw, prices_raw)]
    candles = ind.ticks_to_candles_by_time(ticks_list, CANDLE_TIMEFRAME_SEC)
    closes = [c["close"] for c in candles]
    candle_epochs = [int(c["epoch"]) for c in candles]
    return closes, candle_epochs, active_symbol


def _duration_to_candles(duration: int, unit: str) -> int:
    """
    Converte a duração do contrato (na unidade de config.DURATION_UNIT) para
    número de velas de CANDLE_TIMEFRAME_SEC segundos à frente.

    Aproximação: contratos em "t" (ticks) são tratados como 1 vela por tick,
    já que o backtester opera sobre velas agregadas, não ticks brutos.
    """
    if unit == "s":
        candles = round(duration / CANDLE_TIMEFRAME_SEC)
    elif unit == "m":
        candles = round(duration * 60 / CANDLE_TIMEFRAME_SEC)
    else:  # "t" (ticks) — aproximação 1:1 em velas
        candles = duration
    return max(1, int(candles))


def run_backtest(
    ticks_path: str = TICKS_CSV,
    initial_balance: float = 1000.0,
    symbol: str = None,
    use_ai_model: bool = None,
    payout_ratio: float = BACKTEST_PAYOUT_RATIO,
) -> BacktestResult:
    """
    Executa o backtest completo sobre o histórico de ticks e retorna o resultado.

    Args:
        ticks_path:   caminho do CSV de ticks (formato do collector.py).
        initial_balance: saldo inicial simulado.
        symbol:       símbolo a filtrar; se None, resolve automaticamente.
        use_ai_model: força USE_AI_MODEL=True/False durante a simulação
                      (None = mantém o valor atual de config.py).
        payout_ratio: payout assumido para contratos vencedores (0.85 = 85%).
    """
    closes, epochs, active_symbol = _load_candles(ticks_path, symbol)

    min_needed = MIN_CANDLES + 1
    if len(closes) < min_needed:
        raise ValueError(
            f"Velas insuficientes para backtest: {len(closes)} < {min_needed} "
            f"necessárias (MIN_CANDLES={MIN_CANDLES})."
        )

    # Isola o RiskManager de qualquer estado/arquivo de log ao vivo
    tmp_dir = tempfile.mkdtemp(prefix="standeriv_backtest_")
    tmp_log = os.path.join(tmp_dir, "backtest_operations.csv")
    old_ops_log = rm_mod.OPERATIONS_LOG
    old_use_ai = strat_mod.USE_AI_MODEL
    rm_mod.OPERATIONS_LOG = tmp_log
    if use_ai_model is not None:
        strat_mod.USE_AI_MODEL = use_ai_model

    try:
        risk = RiskManager(initial_balance=initial_balance)
        trades: list[Trade] = []

        adx_history: list = []
        candles_since_last_entry = ENTRY_CANDLE_INTERVAL
        i = MIN_CANDLES - 1   # índice da última vela (0-based) já "fechada"
        n = len(closes)

        while i < n - 1:
            window_start = max(0, i + 1 - _CANDLE_BUFFER_MAXLEN)
            candle_closes = closes[window_start:i + 1]

            candles_since_last_entry += 1

            adaptive_interval = ENTRY_CANDLE_INTERVAL
            if risk.consecutive_losses >= 2:
                adaptive_interval += 1

            if not risk.can_trade() or candles_since_last_entry < adaptive_interval:
                i += 1
                continue

            adx_min = get_adaptive_adx_min(adx_history)
            signal, indicators = get_signal(
                candle_closes, adx_min=adx_min, adx_history=adx_history,
            )

            if indicators.get("adx"):
                adx_history.append(indicators["adx"])
                if len(adx_history) > _ADX_HISTORY_MAXLEN:
                    adx_history.pop(0)

            if signal not in ("BUY", "SELL"):
                i += 1
                continue

            stake = risk.get_stake()
            duration = ai_predictor.predict_duration(candle_closes) or DURATION
            candles_ahead = _duration_to_candles(duration, DURATION_UNIT)
            exit_idx = i + candles_ahead
            if exit_idx >= n:
                break   # dados insuficientes para resolver o contrato — encerra

            entry_price = closes[i]
            exit_price = closes[exit_idx]
            win = (exit_price > entry_price) if signal == "BUY" else (exit_price < entry_price)
            profit = round(stake * payout_ratio, 2) if win else -stake

            risk.record_result(active_symbol, signal, stake, duration, profit, indicators)

            trades.append(Trade(
                entry_epoch=epochs[i], exit_epoch=epochs[exit_idx],
                direction=signal,
                entry_price=entry_price, exit_price=exit_price,
                stake=stake, duration=duration, duration_unit=DURATION_UNIT,
                profit=profit, result="WIN" if win else "LOSS",
                balance_after=risk.balance,
            ))

            candles_since_last_entry = 0
            i = exit_idx  # próxima avaliação só após o contrato atual encerrar

        return BacktestResult(
            symbol=active_symbol,
            initial_balance=initial_balance,
            final_balance=risk.balance,
            trades=trades,
            candles_used=n,
            use_ai_model=strat_mod.USE_AI_MODEL,
        )
    finally:
        rm_mod.OPERATIONS_LOG = old_ops_log
        strat_mod.USE_AI_MODEL = old_use_ai


def compare_ai_filter(
    ticks_path: str = TICKS_CSV,
    initial_balance: float = 1000.0,
    symbol: str = None,
    payout_ratio: float = BACKTEST_PAYOUT_RATIO,
) -> dict:
    """Executa o mesmo histórico com e sem o filtro de IA para comparação direta."""
    without_ai = run_backtest(
        ticks_path, initial_balance, symbol, use_ai_model=False, payout_ratio=payout_ratio,
    )
    with_ai = run_backtest(
        ticks_path, initial_balance, symbol, use_ai_model=True, payout_ratio=payout_ratio,
    )
    return {"without_ai_model": without_ai, "with_ai_model": with_ai}


def _save_report(result: BacktestResult, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    payload = result.to_dict()
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtesting offline do StanDeriv")
    parser.add_argument("--ticks", default=TICKS_CSV, help="Caminho do ticks.csv")
    parser.add_argument("--balance", type=float, default=1000.0, help="Saldo inicial simulado")
    parser.add_argument("--symbol", default=None, help="Símbolo a filtrar (padrão: auto-detectado)")
    parser.add_argument("--payout", type=float, default=BACKTEST_PAYOUT_RATIO, help="Payout assumido (0-1)")
    parser.add_argument("--compare-ai", action="store_true", help="Compara com/sem filtro de IA")
    parser.add_argument("--output", default=None, help="Salva relatório JSON neste caminho")
    args = parser.parse_args()

    if args.compare_ai:
        results = compare_ai_filter(args.ticks, args.balance, args.symbol, args.payout)
        for label, res in results.items():
            print(f"\n=== {label} ===")
            print(res.summary())
        if args.output:
            combined = {k: v.to_dict() for k, v in results.items()}
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(combined, f, indent=2)
            print(f"\n[BACKTEST] Relatório salvo em '{args.output}'")
        return

    result = run_backtest(args.ticks, args.balance, args.symbol, payout_ratio=args.payout)
    print(result.summary())
    if args.output:
        _save_report(result, args.output)
        print(f"\n[BACKTEST] Relatório salvo em '{args.output}'")


if __name__ == "__main__":
    main()
