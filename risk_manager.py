"""
risk_manager.py — controle de risco profissional.

Responsabilidades:
  - Sizing pelo %risco fixo do saldo (sem martingale)
  - Stop diário e take profit diário
  - Pausa automática após N losses consecutivos
  - Log completo de cada operação em CSV
"""

import csv
import os
import time
from datetime import datetime, date
from config import (
    STAKE_PCT, STOP_LOSS_PCT, TAKE_PROFIT_PCT,
    MAX_CONSEC_LOSSES, OPERATIONS_LOG, ADX_MIN,
    PAUSE_BASE_SEC, PAUSE_SCALE_FACTOR, RESUME_ON_WIN,
    DRIFT_WINDOW, DRIFT_WIN_RATE_MIN,
)


class RiskManager:
    """
    Controla se o bot pode operar e com qual stake.

    Uso:
        rm = RiskManager(initial_balance=1000.0)
        if rm.can_trade():
            stake = rm.get_stake()
            # ... executa operação ...
            rm.record_result(symbol, direction, stake, duration, profit, indicators)
    """

    def __init__(self, initial_balance: float):
        self.balance: float = initial_balance
        self._initial_balance: float = initial_balance

        # Controle diário
        self._today: date = date.today()
        self._daily_start_balance: float = initial_balance
        self._daily_profit: float = 0.0

        # Controle de sequência de perdas
        self._consec_losses: int = 0
        self._pause_until: float = 0.0  # epoch timestamp
        self.consecutive_losses: int = 0  # exposto para cadência adaptativa

        # P13: Janela deslizante de resultados para detecção de drift
        self._recent_results: list = []

        # Inicializa arquivo de log (append se existir com conteúdo, senão cria com cabeçalho)
        _header = [
            "timestamp", "symbol", "direction", "stake", "duration",
            "result", "profit",
            "balance_before", "balance_after",
            "ema9", "ema21", "rsi", "adx", "macd_hist",
            "ai_confidence", "ai_score",
            "consec_losses",
            "drawdown_pct", "win_rate_recent", "market_condition",
        ]
        if os.path.exists(OPERATIONS_LOG) and os.path.getsize(OPERATIONS_LOG) > 0:
            # Arquivo existe com conteúdo → append sem cabeçalho
            self._log_file = open(OPERATIONS_LOG, "a", newline="")
            self._csv_writer = csv.writer(self._log_file)
        else:
            # Arquivo novo → criar com cabeçalho
            self._log_file = open(OPERATIONS_LOG, "w", newline="")
            self._csv_writer = csv.writer(self._log_file)
            self._csv_writer.writerow(_header)

    # ──────────────────────────────────────────────
    #  Verificações
    # ──────────────────────────────────────────────

    def is_paused(self) -> bool:
        """True se estiver em período de pausa por losses consecutivos."""
        if time.time() < self._pause_until:
            remaining = int(self._pause_until - time.time())
            print(f"[RISCO] Bot pausado — retorna em {remaining // 60}m {remaining % 60}s")
            return True
        return False

    def can_trade(self) -> bool:
        """True se todas as condições de risco permitirem nova entrada."""
        self._reset_daily_if_needed()

        if self.is_paused():
            return False

        daily_pnl_pct = (
            self._daily_profit / self._daily_start_balance
            if self._daily_start_balance > 0 else 0.0
        )

        if daily_pnl_pct <= -STOP_LOSS_PCT:
            print(
                f"[RISCO] Stop diário atingido: {daily_pnl_pct * 100:.2f}% "
                f"(limite: -{STOP_LOSS_PCT * 100:.0f}%)"
            )
            return False

        if daily_pnl_pct >= TAKE_PROFIT_PCT:
            print(
                f"[RISCO] Take profit diário atingido: {daily_pnl_pct * 100:.2f}% "
                f"(meta: +{TAKE_PROFIT_PCT * 100:.0f}%)"
            )
            return False

        return True

    # ──────────────────────────────────────────────
    #  Sizing
    # ──────────────────────────────────────────────

    def get_stake(self) -> float:
        """Retorna o stake da próxima operação (STAKE_PCT % do saldo atual)."""
        stake = round(self.balance * STAKE_PCT, 2)
        return max(stake, 0.35)  # mínimo aceitável pela Deriv (~$0.35)

    # ──────────────────────────────────────────────
    #  Registro de resultado
    # ──────────────────────────────────────────────

    def record_result(
        self,
        symbol: str,
        direction: str,
        stake: float,
        duration: int,
        profit: float,
        indicators: dict,
    ) -> None:
        """
        Atualiza saldo, controles de risco e salva log.

        Args:
            profit: variação no saldo (positivo = lucro, negativo = prejuízo)
        """
        self._reset_daily_if_needed()

        balance_before = self.balance
        self.balance = round(self.balance + profit, 2)
        self._daily_profit = round(self._daily_profit + profit, 2)

        if profit < 0.0:
            self._consec_losses += 1
        else:
            # P8: Win durante pausa → retomar antecipadamente
            if RESUME_ON_WIN and time.time() < self._pause_until:
                self._pause_until = 0.0
                print("[RISCO] Win detectado durante pausa — retomando operações")
            self._consec_losses = 0

        self.consecutive_losses = self._consec_losses

        # P8: Pausa escalável — base * scale_factor ^ (losses_extras), cap 2h
        if self._consec_losses >= MAX_CONSEC_LOSSES:
            extra = self._consec_losses - MAX_CONSEC_LOSSES
            pause_sec = int(PAUSE_BASE_SEC * (PAUSE_SCALE_FACTOR ** extra))
            pause_sec = min(pause_sec, 7200)  # cap de 2 horas
            self._pause_until = time.time() + pause_sec
            print(
                f"[RISCO] {self._consec_losses} losses consecutivos — "
                f"pausando por {pause_sec // 60} min"
            )

        # P13: Detectar drift do modelo
        self._recent_results.append(1 if profit >= 0.0 else 0)
        if len(self._recent_results) > DRIFT_WINDOW:
            self._recent_results.pop(0)
        self._check_drift()

        result_str = "WIN" if profit >= 0.0 else "LOSS"
        print(
            f"[TRADE] {result_str:4s} | Profit: {profit:+.2f} USD | "
            f"Saldo: {self.balance:.2f} | PnL hoje: {self._daily_profit:+.2f}"
        )

        # P15: Colunas extras no log
        drawdown_pct     = round((self.balance - self._initial_balance) / self._initial_balance * 100, 2)
        win_rate_recent  = (
            round(sum(self._recent_results) / len(self._recent_results) * 100, 1)
            if self._recent_results else ""
        )
        market_condition = "trending" if indicators.get("adx", 0) >= ADX_MIN else "lateral"

        with open(OPERATIONS_LOG, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                symbol, direction, stake, duration,
                result_str, profit,
                balance_before, self.balance,
                indicators.get("ema9",           ""),
                indicators.get("ema21",          ""),
                indicators.get("rsi",            ""),
                indicators.get("adx",            ""),
                indicators.get("macd_hist",      ""),
                indicators.get("ai_confidence",  ""),
                indicators.get("ai_score",       ""),
                self._consec_losses,
                drawdown_pct, win_rate_recent, market_condition,
            ])

    # ──────────────────────────────────────────────
    #  Interno
    # ──────────────────────────────────────────────

    def _reset_daily_if_needed(self) -> None:
        today = date.today()
        if today != self._today:
            print(f"[RISCO] Novo dia — resetando contadores diários")
            self._today = today
            self._daily_start_balance = self.balance
            self._daily_profit = 0.0
            self._consec_losses = 0

    def _check_drift(self) -> None:
        """P13: Alerta quando o win rate recente cai abaixo do limiar configurado."""
        if len(self._recent_results) < DRIFT_WINDOW:
            return
        wr = sum(self._recent_results) / len(self._recent_results)
        if wr < DRIFT_WIN_RATE_MIN:
            print(
                f"[DRIFT] ALERTA: win rate dos últimos {DRIFT_WINDOW} trades = "
                f"{wr:.1%} (abaixo de {DRIFT_WIN_RATE_MIN:.0%}) "
                "→ considere retreinar o modelo"
            )
