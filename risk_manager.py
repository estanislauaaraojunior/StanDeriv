"""
risk_manager.py — controle de risco profissional.

Responsabilidades:
  - Sizing pelo %risco fixo do saldo (sem martingale)
  - Stop diário e take profit diário
  - Pausa automática após N losses consecutivos
  - Log completo de cada operação em CSV
"""

import csv
import json
import logging
import os
import time
from datetime import datetime, date
from config import (
    STAKE_PCT, STOP_LOSS_PCT, TAKE_PROFIT_PCT,
    MAX_CONSEC_LOSSES, OPERATIONS_LOG, ADX_MIN,
    PAUSE_BASE_SEC, PAUSE_SCALE_FACTOR, RESUME_ON_WIN,
    DRIFT_WINDOW, DRIFT_WIN_RATE_MIN,
)

logger = logging.getLogger(__name__)


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

        # Tenta restaurar estado salvo da sessão anterior
        self._load_saved_state()

        # Garante que o arquivo de log existe com cabeçalho (sem manter handle aberto)
        if not (os.path.exists(OPERATIONS_LOG) and os.path.getsize(OPERATIONS_LOG) > 0):
            with open(OPERATIONS_LOG, "w", newline="") as f:
                csv.writer(f).writerow([
                    "timestamp", "symbol", "direction", "stake", "duration",
                    "result", "profit",
                    "balance_before", "balance_after",
                    "ema9", "ema21", "rsi", "adx", "macd_hist",
                    "ai_confidence", "ai_score",
                    "consec_losses",
                    "drawdown_pct", "win_rate_recent", "market_condition",
                ])

    # ──────────────────────────────────────────────
    #  Verificações
    # ──────────────────────────────────────────────

    def is_paused(self) -> bool:
        """True se estiver em período de pausa por losses consecutivos."""
        if time.time() < self._pause_until:
            remaining = int(self._pause_until - time.time())
            logger.info("Bot pausado — retorna em %dm %ds", remaining // 60, remaining % 60)
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
            logger.warning(
                "Stop diário atingido: %.2f%% (limite: -%.0f%%)",
                daily_pnl_pct * 100, STOP_LOSS_PCT * 100,
            )
            return False

        if daily_pnl_pct >= TAKE_PROFIT_PCT:
            logger.info(
                "Take profit diário atingido: %.2f%% (meta: +%.0f%%)",
                daily_pnl_pct * 100, TAKE_PROFIT_PCT * 100,
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
                logger.info("Win detectado durante pausa — retomando operações")
            self._consec_losses = 0

        self.consecutive_losses = self._consec_losses

        # P8: Pausa escalável — base * scale_factor ^ (losses_extras), cap 2h
        if self._consec_losses >= MAX_CONSEC_LOSSES:
            extra = self._consec_losses - MAX_CONSEC_LOSSES
            pause_sec = int(PAUSE_BASE_SEC * (PAUSE_SCALE_FACTOR ** extra))
            pause_sec = min(pause_sec, 7200)  # cap de 2 horas
            self._pause_until = time.time() + pause_sec
            logger.warning(
                "%d losses consecutivos — pausando por %d min",
                self._consec_losses, pause_sec // 60,
            )

        # P13: Detectar drift do modelo
        self._recent_results.append(1 if profit >= 0.0 else 0)
        if len(self._recent_results) > DRIFT_WINDOW:
            self._recent_results.pop(0)
        self._check_drift()

        result_str = "WIN" if profit >= 0.0 else "LOSS"
        logger.info(
            "%s | Profit: %+.2f USD | Saldo: %.2f | PnL hoje: %+.2f",
            result_str, profit, self.balance, self._daily_profit,
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

        self._save_risk_state()

    # ──────────────────────────────────────────────
    #  Interno
    # ──────────────────────────────────────────────

    def _reset_daily_if_needed(self) -> None:
        today = date.today()
        if today != self._today:
            logger.info("Novo dia — resetando contadores diários")
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
            logger.warning(
                "[DRIFT] ALERTA: win rate dos últimos %d trades = %.1f%% "
                "(abaixo de %.0f%%) → considere retreinar o modelo",
                DRIFT_WINDOW, wr * 100, DRIFT_WIN_RATE_MIN * 100,
            )

    # ──────────────────────────────────────────────
    #  Exposição de estado para o dashboard
    # ──────────────────────────────────────────────

    def to_state_dict(self) -> dict:
        """Retorna estado atual do RiskManager como dict serializável em JSON."""
        daily_pnl_pct = (
            self._daily_profit / self._daily_start_balance
            if self._daily_start_balance > 0 else 0.0
        )
        wr = (
            round(sum(self._recent_results) / len(self._recent_results) * 100, 1)
            if self._recent_results else 0.0
        )
        drift = len(self._recent_results) >= DRIFT_WINDOW and wr / 100 < DRIFT_WIN_RATE_MIN
        pause_remaining = max(0, int(self._pause_until - time.time()))
        return {
            # Para o dashboard
            "is_paused":           pause_remaining > 0,
            "pause_remaining_sec": pause_remaining,
            "daily_pnl":           round(self._daily_profit, 2),
            "daily_pnl_pct":       round(daily_pnl_pct * 100, 2),
            "consec_losses":       self._consec_losses,
            "drift_detected":      drift,
            "win_rate_recent":     wr,
            # Para restauração de estado
            "date":                date.today().isoformat(),
            "balance":             round(self.balance, 2),
            "pause_until_epoch":   self._pause_until,
            "recent_results":      self._recent_results[-DRIFT_WINDOW:],
        }

    def _save_risk_state(self) -> None:
        """Persiste estado atual em risk_state.json para leitura pelo dashboard."""
        try:
            state_path = os.path.join(os.path.dirname(OPERATIONS_LOG), "risk_state.json")
            with open(state_path, "w") as f:
                json.dump(self.to_state_dict(), f)
        except Exception:
            pass

    def _load_saved_state(self) -> None:
        """Restaura estado salvo de execução anterior (consec_losses, pause_until, recent_results)."""
        try:
            state_path = os.path.join(os.path.dirname(OPERATIONS_LOG), "risk_state.json")
            if not os.path.exists(state_path):
                return
            with open(state_path) as f:
                saved = json.load(f)
            # Só restaura se for do mesmo dia
            saved_date_str = saved.get("date", "")
            if saved_date_str != date.today().isoformat():
                return
            self._consec_losses = int(saved.get("consec_losses", 0))
            self.consecutive_losses = self._consec_losses
            self._pause_until = float(saved.get("pause_until_epoch", 0.0))
            self._recent_results = list(saved.get("recent_results", []))
            self._daily_profit = float(saved.get("daily_pnl", 0.0))
            saved_balance = float(saved.get("balance", 0.0))
            if saved_balance > 0:
                self.balance = saved_balance
            logger.info(
                "Estado restaurado: consec_losses=%d, saldo=%.2f",
                self._consec_losses, self.balance,
            )
        except Exception:
            pass
