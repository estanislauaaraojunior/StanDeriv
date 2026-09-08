"""Risco persistente por conta, reconciliação de saldo e diário idempotente."""
import csv
import json
import math
import os
import threading
import time
import uuid
from datetime import date, datetime
from pathlib import Path
from config import (STAKE_PCT, STOP_LOSS_PCT, TAKE_PROFIT_PCT, MAX_CONSEC_LOSSES,
                    OPERATIONS_LOG, ADX_MIN, PAUSE_BASE_SEC, PAUSE_SCALE_FACTOR,
                    RESUME_ON_WIN, DRIFT_WINDOW, DRIFT_WIN_RATE_MIN)
from model_contract import write_json_atomic

FIELDS = ['timestamp', 'symbol', 'direction', 'stake', 'duration', 'result', 'profit', 'balance_before',
          'balance_after', 'ema9', 'ema21', 'rsi', 'adx', 'macd_hist', 'ai_confidence', 'ai_score',
          'consec_losses', 'drawdown_pct', 'win_rate_recent', 'market_condition',
          'contract_id', 'duration_unit', 'entry_price', 'exit_price', 'entry_epoch', 'exit_epoch',
          'payout', 'buy_price', 'model_id', 'ai_direction', 'tech_score', 'account_key']


class RiskManager:
    def __init__(self, initial_balance):
        if not math.isfinite(initial_balance) or initial_balance <= 0:
            raise ValueError('saldo inicial inválido')
        self._authenticated_initial_balance = float(initial_balance)
        self._lock = threading.RLock()
        self.balance = self._initial_balance = self._daily_start_balance = self._peak_balance = float(initial_balance)
        self._today = date.today()
        self._daily_profit = 0.
        self._consec_losses = self.consecutive_losses = 0
        self._pause_until = 0.
        self._recent_results = []
        self._account_key = ''
        self._processed_contracts = set()
        self._outbox = []
        self._path = Path(OPERATIONS_LOG).parent / 'risk_state.json'
        self._load_saved_state()
        self._ensure_log()
        self._flush_outbox()

    def _ensure_log(self):
        path = Path(OPERATIONS_LOG)
        if not path.exists() or not path.stat().st_size:
            with path.open('w', newline = '') as f:
                csv.writer(f).writerow(FIELDS)
        else:
            with path.open(newline = '') as f:
                self._log_fields = next(csv.reader(f))
        if not hasattr(self, '_log_fields'):
            self._log_fields = FIELDS

    def bind_account(self, account_key, balance, * , preserve_local_balance = False):
        """Saldo autenticado prevalece. Reconexão com contrato pendente não conta lucro duas vezes."""
        if not account_key or not math.isfinite(balance) or balance < 0:
            raise ValueError('identidade ou saldo autenticado inválido')
        with self._lock:
            if self._account_key and self._account_key != account_key:
                # Não mistura risco de duas contas; restaura histórico próprio se existir.
                self._save_risk_state()
                archive = self._path.with_name('risk_accounts.json')
                accounts = json.loads(archive.read_text()) if archive.exists() else {}
                accounts[self._account_key] = self.to_state_dict()
                write_json_atomic(archive, accounts)
                saved = accounts.get(account_key)
                self._daily_profit = 0.
                self._consec_losses = self.consecutive_losses = 0
                self._pause_until = 0.
                self._recent_results = []
                self._initial_balance = self._daily_start_balance = self._peak_balance = balance
                self._processed_contracts = set()
                self._outbox = []
                if saved:
                    self._restore(saved)
            self._account_key = account_key
            self.synchronize_balance(balance, preserve_local_balance = preserve_local_balance)
            self._save_risk_state()

    def synchronize_balance(self, balance, * , preserve_local_balance = False):
        if not math.isfinite(balance) or balance < 0:
            raise ValueError('saldo inválido')
        if preserve_local_balance:
            return
        # Ajusta a base por depósitos/retiradas sem apagar o PnL diário já realizado.
        self.balance = float(balance)
        self._daily_start_balance = max(self.balance - self._daily_profit, 0.)
        if not self._account_key or self._initial_balance <= 0:
            self._initial_balance = self._daily_start_balance
        self._peak_balance = max(self._peak_balance, self.balance)

    def _reset_daily_if_needed(self):
        if date.today() != self._today:
            self._today = date.today()
            self._daily_start_balance = self.balance
            self._daily_profit = 0.
            self._consec_losses = self.consecutive_losses = 0
            # Não elimina IDs nem pausa pendente na virada do dia.

    def is_paused(self):
        return time.time() < self._pause_until

    def can_trade(self):
        self._reset_daily_if_needed()
        if self.is_paused() or self.get_stake() > self.balance:
            return False
        pct = self._daily_profit / self._daily_start_balance if self._daily_start_balance > 0 else 0.
        return - STOP_LOSS_PCT < pct < TAKE_PROFIT_PCT

    def get_stake(self):
        return max(round(self.balance * STAKE_PCT, 2), .35)

    def record_result(self, symbol, direction, stake, duration, profit, indicators, contract_id = None, details = None):
        if not math.isfinite(profit):
            raise ValueError('resultado inválido')
        with self._lock:
            self._flush_outbox()
            key = str(contract_id) if contract_id is not None else str(uuid.uuid4())
            if key in self._processed_contracts:
                return False
            self._reset_daily_if_needed()
            before = self.balance
            self.balance = round(before + profit, 2)
            self._daily_profit = round(self._daily_profit + profit, 2)
            self._peak_balance = max(self._peak_balance, self.balance)
            if profit < 0:
                self._consec_losses += 1
            elif profit > 0:
                self._consec_losses = 0
                if RESUME_ON_WIN:
                    self._pause_until = 0.
            self.consecutive_losses = self._consec_losses
            if profit < 0 and self._consec_losses >= MAX_CONSEC_LOSSES:
                seconds = min(PAUSE_BASE_SEC * PAUSE_SCALE_FACTOR ** min(self._consec_losses - MAX_CONSEC_LOSSES, 20), 7200)
                self._pause_until = time.time() + seconds
            if profit != 0:
                self._recent_results = (self._recent_results + [int(profit > 0)])[-DRIFT_WINDOW:]
            self._processed_contracts.add(key)
            state = self.to_state_dict()
            row = dict(timestamp = datetime.now().isoformat(), symbol = symbol, direction = direction, stake = stake,
                     duration = duration, result = 'WIN' if profit > 0 else ('LOSS' if profit < 0 else 'VOID'),
                     profit = profit, balance_before = before, balance_after = self.balance, consec_losses = self._consec_losses,
                     drawdown_pct = state['drawdown_pct'], win_rate_recent = state['win_rate_recent'],
                     market_condition = 'trending' if indicators.get('adx', 0) >= ADX_MIN else 'lateral',
                     contract_id = key, account_key = self._account_key)
            row.update({k: indicators.get(k, '') for k in ['ema9', 'ema21', 'rsi', 'adx', 'macd_hist', 'ai_confidence',
                        'ai_score', 'model_id', 'ai_direction', 'tech_score']})
            row.update(details or {})
            self._outbox.append(row)
            # Estado + resultado pendente são uma gravação atômica antes do CSV.
            self._save_risk_state()
            self._flush_outbox()
            print(f"[TRADE] {row['result']} | Profit: {profit:+.2f} USD | Saldo: {self.balance:.2f}")
            self._check_drift()
            return True

    def _flush_outbox(self):
        if not self._outbox:
            return
        with open(OPERATIONS_LOG, newline = '') as f:
            existing = {r.get('contract_id') for r in csv.DictReader(f)}
        if 'contract_id' not in self._log_fields:
            raise ValueError('log legado não suporta idempotência; use novo OPERATIONS_LOG')
        with open(OPERATIONS_LOG, 'a', newline = '') as f:
            w = csv.DictWriter(f, self._log_fields, extrasaction = 'ignore')
            for row in self._outbox:
                if row['contract_id'] not in existing:
                    w.writerow(row)
                    existing.add(row['contract_id'])
            f.flush()
            os.fsync(f.fileno())
        self._outbox = []
        self._save_risk_state()

    def _check_drift(self):
        if self.to_state_dict()['drift_detected']:
            print('[DRIFT] Taxa de acerto recente abaixo do limite; avaliar modelo fora da amostra')

    def to_state_dict(self):
        wr = 100 * sum(self._recent_results) / len(self._recent_results) if self._recent_results else 0.
        return dict(is_paused = self.is_paused(), pause_remaining_sec = max(0, int(self._pause_until - time.time())),
                    daily_pnl = self._daily_profit, daily_pnl_pct = 100 * self._daily_profit / self._daily_start_balance if self._daily_start_balance else 0.,
                    consec_losses = self._consec_losses, drift_detected = len(self._recent_results) >= DRIFT_WINDOW and wr / 100 < DRIFT_WIN_RATE_MIN,
                    win_rate_recent = round(wr, 1), date = self._today.isoformat(), balance = self.balance,
                    pause_until_epoch = self._pause_until, recent_results = self._recent_results,
                    daily_start_balance = self._daily_start_balance, initial_balance = self._initial_balance,
                    peak_balance = self._peak_balance, drawdown_pct = round(100 * (self._peak_balance - self.balance) / self._peak_balance, 2) if self._peak_balance else 0.,
                    account_key = self._account_key, processed_contracts = sorted(self._processed_contracts), outbox = self._outbox)

    def _save_risk_state(self):
        write_json_atomic(self._path, self.to_state_dict())

    def _restore(self, saved):
        self._account_key = saved.get('account_key', '')
        self._processed_contracts = set(saved.get('processed_contracts', []))
        self._outbox = saved.get('outbox', [])
        saved_balance = float(saved.get('balance', self.balance))
        if not math.isfinite(saved_balance) or saved_balance < 0:
            raise ValueError('saldo persistido inválido')
        self.balance = saved_balance
        self._initial_balance = float(saved.get('initial_balance', saved_balance))
        self._peak_balance = max(float(saved.get('peak_balance', saved_balance)), saved_balance)
        self._pause_until = float(saved.get('pause_until_epoch', 0.))
        if saved.get('date') == date.today().isoformat():
            self._consec_losses = self.consecutive_losses = int(saved.get('consec_losses', 0))
            self._recent_results = list(saved.get('recent_results', []))
            self._daily_profit = float(saved.get('daily_pnl', 0.))
            self._daily_start_balance = float(saved.get('daily_start_balance', self.balance - self._daily_profit))
        else:
            self._daily_start_balance = self.balance

    def _load_saved_state(self):
        if self._path.exists():
            # Corrupção de estado não deve apagar stops silenciosamente.
            self._restore(json.loads(self._path.read_text()))
