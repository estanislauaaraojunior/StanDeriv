"""
executor.py — cliente WebSocket da Deriv e execução de ordens.

Responsabilidades:

  - Gerenciar ciclo de vida da conexão WebSocket
  - Autorizar token e subscrever fluxo de ticks
  - Chamar strategy.get_signal() a cada tick
  - Enviar proposals de compra/venda via API Deriv
  - Rastrear contratos abertos e registrar resultado via RiskManager

Fluxo de uma operação:

  tick → get_signal() → send_proposal → handle_proposal (buy)
       → handle_contract (result)
"""

import json
import os
import re
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

import websocket
import deriv_session

from config import (
    APP_ID,
    TOKEN,
    SYMBOL,
    DURATION,
    DURATION_UNIT,
    BASIS,
    CURRENCY,
    MIN_CANDLES,
    ENTRY_TICK_INTERVAL,
    ENTRY_CANDLE_INTERVAL,
    PRICE_BUFFER_SIZE,
    PROPOSAL_TIMEOUT_SEC,
    HEARTBEAT_TIMEOUT_SEC,
    CANDLE_NOTIFY,
    PA_SR_TOLERANCE,
    CANDLE_TIMEFRAME_SEC,
    TICKS_CSV,
)

from risk_manager import RiskManager
from strategy import get_signal, get_adaptive_adx_min
import ai_predictor
import config
import math
import hashlib
import secrets
from model_contract import (FEATURE_WINDOW, MAX_TICK_GAP_SEC, duration_seconds,
                            profitable_proposal, valid_payout_proposal,
                            write_json_atomic, acquire_execution_lock)
from market_data import read_records, canonicalize
import indicators as ind


_STATE_JSON = Path(__file__).resolve().parent / "state.json"
_BOT_STATUS_JSON = Path(__file__).resolve().parent / "bot_status.json"


class DerivBot:
    """
    Bot de trading para a plataforma Deriv via WebSocket API.

    Args:
        risk_manager: instância de RiskManager já inicializada
        demo:
            True  → exibe avisos de modo demo
            False → opera em conta real
                     (requer TOKEN de conta real)
    """

    def __init__(
        self,
        risk_manager: RiskManager,
        demo: bool = True,
        auth_mode: str = "legacy",
        account_id: str = "",
        initial_ws_url: str = "",
    ) -> None:
        self.symbol = SYMBOL
        self.risk_manager = risk_manager
        self.demo = demo
        self.auth_mode = auth_mode
        self.account_id = account_id
        self._next_ws_url = initial_ws_url
        self._last_status_write: float = 0.0

        self._ws_url = (
            f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}"
        )

        # P12: Buffer de ticks brutos.
        self._prices: deque = deque(maxlen=PRICE_BUFFER_SIZE)

        # Buffer de candles de tempo.
        self._candle_closes: deque = deque(maxlen=500)
        self._candle_epochs: deque = deque(maxlen=500)
        self._candles: deque = deque(maxlen=500)

        # Vela em construção.
        self._cur_candle_open: Optional[float] = None
        self._cur_candle_high: float = 0.0
        self._cur_candle_low: float = 0.0
        self._cur_candle_close: float = 0.0
        self._cur_candle_epoch: int = 0

        # Sinaliza que uma nova vela acabou de ser fechada neste tick.
        self._new_candle_closed: bool = False
        self._last_waiting_candle_epoch: int = -1

        # Contador de velas fechadas desde a última entrada.
        self._candles_since_last_entry: int = ENTRY_CANDLE_INTERVAL

        # P10: Histórico de ADX para cálculo adaptativo.
        self._adx_history: deque = deque(maxlen=500)

        # Alertas de padrões de vela.
        self._candle_alerts: deque = deque(maxlen=20)
        self._candle_tick_counter: int = 0

        # Estado da operação em andamento.
        self._in_trade: bool = False
        self._pending_direction: str = ""
        self._pending_stake: float = 0.0
        self._pending_duration: int = DURATION
        self._pending_indicators: dict = {}

        self._open_contract_id: Optional[str] = None

        self._entry_price: float = 0.0
        self._entry_epoch: int = 0

        self._last_tick_price: float = 0.0
        self._last_tick_epoch: int = 0
        self._last_contract_attempt_epoch: int = 0
        self._decision_history: deque = deque(maxlen=120)
        self._unfavorable_decision_streak: int = 0
        self._unfavorable_switch_requested: bool = False

        # P1: Timestamp da última proposal enviada.
        self._pending_timestamp: float = 0.0

        # P7: Timestamp da abertura do contrato.
        self._buy_timestamp: float = 0.0

        # P13: Timer de poll ativo.
        self._contract_poll_timer: Optional[threading.Timer] = None

        # P9: Watchdog de heartbeat.
        self._watchdog: Optional[threading.Timer] = None

        # Cadência de entradas: começa já no limite.
        self._ticks_since_last_entry: int = ENTRY_TICK_INTERVAL

        # Histórico de candles já carregado via API.
        self._api_history_loaded: bool = False

        self._ws: Optional[websocket.WebSocketApp] = None
        self._phase = 'idle'
        self._proposal_req_id = None
        self._buy_req_id = None
        self._portfolio_req_id = None
        self._request_counter = secrets.randbelow(1_000_000_000)
        self._pending_payout = None
        self._pending_buy_price = None
        self._proposal_id = None
        self._proposal_retry_count = 0
        self._proposal_api_error_streak = 0
        self._auth_verified = False
        self._state_lock = threading.RLock()
        self._restore_active_contract()


    def _uses_bearer_auth(self) -> bool:
        return self.auth_mode == "bearer" and bool(self.account_id)

    def _get_ws_url(self) -> str:
        if not self._uses_bearer_auth():
            return self._ws_url

        if self._next_ws_url:
            ws_url = self._next_ws_url
            self._next_ws_url = ""
            return ws_url

        return deriv_session.get_options_ws_url(
            self.account_id,
            app_id=APP_ID,
            token=TOKEN,
            attempts=3,
        )

    def _next_request_id(self):
        self._request_counter += 1
        return self._request_counter

    def _restore_active_contract(self):
        if not _STATE_JSON.exists():
            return
        saved = json.loads(_STATE_JSON.read_text()).get('active_contract')
        if not saved:
            return
        self._phase = saved.get('phase', 'reconciling')
        self._in_trade = True
        self._open_contract_id = str(saved['contract_id']) if saved.get('contract_id') else None
        self._pending_direction = saved.get('direction', '')
        self._pending_stake = float(saved.get('stake', 0))
        self._pending_duration = int(saved.get('duration', config.DURATION))
        self._pending_indicators = saved.get('indicators', {})
        self._pending_timestamp = float(saved.get('pending_timestamp', 0))
        self._buy_timestamp = float(saved.get('buy_timestamp', 0))
        self._pending_payout = saved.get('payout')
        self._pending_buy_price = saved.get('buy_price')
        self._proposal_req_id = saved.get('proposal_req_id')
        self._buy_req_id = saved.get('buy_req_id')
        self._saved_account_key = saved.get('account_key', '')
        self._contract_symbol = saved.get('symbol', self.symbol)
        # Uma proposta sem compra pode ser abandonada após reinício. Uma compra incerta não.
        if self._phase == 'proposal':
            self._clear_pending()

    def _set_active_contract_state(self, clear=False):
        state = json.loads(_STATE_JSON.read_text()) if _STATE_JSON.exists() else {}
        state['active_contract'] = None if clear else dict(
            has_active=True, phase=self._phase, contract_id=self._open_contract_id,
            symbol=getattr(self,'_contract_symbol',self.symbol), direction=self._pending_direction,
            stake=self._pending_stake, duration=self._pending_duration, duration_unit=config.DURATION_UNIT,
            buy_timestamp=self._buy_timestamp, pending_timestamp=self._pending_timestamp,
            entry_epoch=self._entry_epoch, entry_price=self._entry_price,
            payout=self._pending_payout, buy_price=self._pending_buy_price,
            indicators=self._pending_indicators, proposal_req_id=self._proposal_req_id,
            buy_req_id=self._buy_req_id, account_key=self.risk_manager._account_key)
        write_json_atomic(_STATE_JSON,state)

    def _write_runtime_status(self, *, tick_epoch: int = 0, force: bool = False, **updates) -> None:
        now = time.time()
        if not force and now - self._last_status_write < 5:
            return
        self._last_status_write = now
        try:
            status = json.loads(_BOT_STATUS_JSON.read_text()) if _BOT_STATUS_JSON.exists() else {}
        except Exception:
            status = {}
        status.update(
            status="running",
            running=True,
            pid=os.getpid(),
            symbol=self.symbol,
            last_loop_at=time.strftime("%Y-%m-%dT%H:%M:%S%z", time.gmtime(now)),
        )
        if tick_epoch:
            status["last_tick_epoch"] = int(tick_epoch)
            status["last_tick_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.gmtime(int(tick_epoch)))
        status.update(updates)
        try:
            write_json_atomic(_BOT_STATUS_JSON, status)
        except Exception:
            pass

    def _decision_counts(self) -> dict:
        counts: dict[str, int] = {}
        for item in self._decision_history:
            reason = str(item.get("reason", "unknown") or "unknown")
            counts[reason] = counts.get(reason, 0) + 1
        return counts

    def _record_decision(
        self,
        *,
        reason: str,
        signal: Optional[str] = None,
        indicators: Optional[dict] = None,
        tick_epoch: int = 0,
        force: bool = True,
    ) -> None:
        event = {
            "epoch": int(tick_epoch or self._last_tick_epoch or time.time()),
            "symbol": self.symbol,
            "reason": reason,
            "signal": signal,
        }
        if indicators:
            event["indicators"] = {
                key: indicators.get(key)
                for key in (
                    "ema9",
                    "ema21",
                    "rsi",
                    "adx",
                    "macd_hist",
                    "ai_direction",
                    "ai_confidence",
                    "ai_score",
                    "win_probability",
                    "tech_score",
                    "model_id",
                    "candle_pattern_block",
                    "ai_block",
                    "decision_source",
                    "decision_modules",
                    "enabled_decision_modules",
                    "decision_block",
                    "pa_score",
                )
                if key in indicators
            }
        self._decision_history.append(event)
        self._update_unfavorable_market_streak(reason, event, indicators)
        no_entry_for = 0
        if self._last_contract_attempt_epoch:
            no_entry_for = max(0, event["epoch"] - self._last_contract_attempt_epoch)
        elif self._entry_epoch:
            no_entry_for = max(0, event["epoch"] - self._entry_epoch)
        self._write_runtime_status(
            tick_epoch=event["epoch"],
            force=force,
            event="decision",
            last_decision=event,
            decision_counts=self._decision_counts(),
            no_entry_for_sec=no_entry_for,
        )

    def _update_unfavorable_market_streak(
        self,
        reason: str,
        event: dict,
        indicators: Optional[dict],
    ) -> None:
        unfavorable = set(
            getattr(config, "UNFAVORABLE_SYMBOL_SWITCH_REASONS", ())
        )
        threshold = int(
            getattr(config, "UNFAVORABLE_SYMBOL_SWITCH_DECISIONS", 0) or 0
        )

        if threshold <= 0 or reason not in unfavorable:
            if reason in ("signal_ready", "proposal_sent", "active_contract", "risk_block"):
                self._unfavorable_decision_streak = 0
                self._unfavorable_switch_requested = False
            return

        self._unfavorable_decision_streak += 1
        if (
            self._unfavorable_switch_requested
            or self._unfavorable_decision_streak < threshold
            or self._in_trade
            or self._market_switch_pending()
        ):
            return

        self._unfavorable_switch_requested = True
        callback = getattr(self, "on_market_unfavorable", None)
        if callback:
            callback(
                self.symbol,
                reason,
                self._unfavorable_decision_streak,
                indicators or {},
            )

    def _risk_allows_trade(self):
        return (not getattr(config, "ENFORCE_RISK_GATE", True)) or self.risk_manager.can_trade()

    def _record_api_error(
        self,
        *,
        error: dict,
        phase: str,
        req_id=None,
    ) -> None:
        code = str(error.get("code") or "UnknownError")
        message = str(error.get("message") or "Erro desconhecido")
        details = {
            "code": code,
            "message": message,
            "phase": phase,
            "req_id": req_id,
            "symbol": self.symbol,
            "direction": self._pending_direction,
            "duration": self._pending_duration,
            "duration_unit": config.DURATION_UNIT,
        }
        print(
            "[BOT] Requisição rejeitada pela API "
            f"(fase={phase}, req_id={req_id}, code={code}): {message}"
        )
        self._write_runtime_status(
            tick_epoch=self._last_tick_epoch,
            force=True,
            event="api_error",
            last_api_error=details,
        )

    def _proposal_error_switch_threshold(self) -> int:
        return int(getattr(config, "PROPOSAL_API_ERROR_SWITCH_THRESHOLD", 3) or 0)

    def _request_switch_after_repeated_proposal_errors(self, reason: str) -> None:
        threshold = self._proposal_error_switch_threshold()
        if threshold <= 0:
            return
        self._proposal_api_error_streak += 1
        if (
            self._proposal_api_error_streak < threshold
            or self._market_switch_pending()
            or self._open_contract_id
        ):
            return
        callback = getattr(self, "on_market_unfavorable", None)
        if callback:
            callback(
                self.symbol,
                reason,
                self._proposal_api_error_streak,
                self._pending_indicators or {},
            )

    @staticmethod
    def _payout_limit_from_error(error: dict) -> Optional[tuple[float, float]]:
        message = str(error.get("message") or "")
        match = re.search(
            r"maximum payout of\s+([0-9]+(?:\.[0-9]+)?).*?"
            r"Current payout is\s+([0-9]+(?:\.[0-9]+)?)",
            message,
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        maximum, current = (float(match.group(1)), float(match.group(2)))
        if not (math.isfinite(maximum) and math.isfinite(current)) or maximum <= 0 or current <= 0:
            return None
        return maximum, current

    def _retry_proposal_with_payout_cap(self, error: dict) -> bool:
        parsed = self._payout_limit_from_error(error)
        if (
            parsed is None
            or self._proposal_retry_count >= 1
            or self._pending_stake <= 0
            or self._ws is None
            or self._phase != "proposal"
        ):
            return False
        maximum, current = parsed
        adjusted = math.floor(self._pending_stake * maximum / current * 0.98 * 100) / 100
        if adjusted < 0.35 or adjusted >= self._pending_stake:
            return False
        self._proposal_retry_count += 1
        self._pending_stake = adjusted
        self._pending_timestamp = time.time()
        self._proposal_req_id = self._next_request_id()
        self._set_active_contract_state()
        proposal = dict(
            proposal=1,
            amount=self._pending_stake,
            basis=BASIS,
            contract_type="CALL" if self._pending_direction == "BUY" else "PUT",
            currency=CURRENCY,
            duration=self._pending_duration,
            duration_unit=DURATION_UNIT,
            req_id=self._proposal_req_id,
        )
        proposal["underlying_symbol" if self._uses_bearer_auth() else "symbol"] = self.symbol
        self._ws.send(json.dumps(proposal))
        self._record_decision(
            reason="proposal_retry_payout_cap",
            signal=self._pending_direction or None,
            indicators=self._pending_indicators,
            tick_epoch=self._last_tick_epoch,
        )
        return True

    def _handle_api_error(self, data: dict) -> None:
        error = data.get("error") or {}
        req_id = data.get("req_id")
        phase = self._phase
        self._record_api_error(error=error, phase=phase, req_id=req_id)

        if phase == "proposal" and req_id == self._proposal_req_id:
            from payout_quotes import market_unavailable

            if market_unavailable(error):
                callback = getattr(self, "on_market_unavailable", None)
                if callback:
                    callback(self.symbol)
            if self._retry_proposal_with_payout_cap(error):
                return
            if self._payout_limit_from_error(error) is not None:
                self._request_switch_after_repeated_proposal_errors("proposal_payout_cap_rejected")
            self._record_decision(
                reason="proposal_rejected_by_api",
                signal=self._pending_direction or None,
                indicators=self._pending_indicators,
                tick_epoch=self._last_tick_epoch,
            )
            self._clear_pending()
            return

        if phase == "buy" and req_id == self._buy_req_id:
            # Resultado financeiro incerto: não repetir a compra automaticamente.
            self._phase = "reconciling"
            self._set_active_contract_state()
            self._record_decision(
                reason="buy_rejected_or_uncertain",
                signal=self._pending_direction or None,
                indicators=self._pending_indicators,
                tick_epoch=self._last_tick_epoch,
            )
            return

        self._record_decision(
            reason="api_error_unmatched",
            tick_epoch=self._last_tick_epoch,
        )

    def _decision_reason(self, signal: Optional[str], indicators: dict) -> str:
        if signal in ("BUY", "SELL"):
            return "signal_ready"
        if indicators.get("decision_block"):
            return str(indicators["decision_block"])
        if indicators.get("candle_pattern_block"):
            return "candle_conflict"
        if indicators.get("ai_block"):
            return "ai_disagreement_or_low_confidence"
        if indicators.get("adx_falling"):
            return "adx_falling"
        if indicators:
            if indicators.get("ai_confidence") is None and indicators.get("model_id"):
                return "missing_ai_confidence"
            return "no_technical_signal"
        return "insufficient_indicator_data"

    def _clear_pending(self):
        self._in_trade=False
        self._phase='idle'
        self._open_contract_id=None
        self._proposal_req_id=self._buy_req_id=None
        self._proposal_retry_count=0
        self._pending_timestamp=self._buy_timestamp=0.
        self._entry_price=0.
        self._entry_epoch=0
        self._set_active_contract_state(clear=True)

    # ─────────────────────────────────────────────────────────
    # API pública
    # ─────────────────────────────────────────────────────────

    def run(self) -> None:
        """Inicia o bot. Bloqueia até encerramento."""
        # Impede duas instâncias financeiras no mesmo workspace, inclusive via CLI.
        if not hasattr(self,'_execution_lock'):
            self._execution_lock=acquire_execution_lock(_STATE_JSON.with_name('executor.lock'))
        self._preload_candles_from_csv()

        mode = "DEMO" if self.demo else "REAL 💸"

        print(f"\n{'=' * 55}")
        print(f"  Deriv Trading Bot — Modo: {mode}")
        print(f"  Símbolo : {self.symbol}")
        print(f"  Duração : {DURATION} {DURATION_UNIT}")
        print(
            f"  Saldo   : "
            f"{self.risk_manager.balance:.2f} USD"
        )
        print(
            f"  Stake   : "
            f"{self.risk_manager.get_stake():.2f} USD "
            f"(1% do saldo)"
        )
        print(f"{'=' * 55}\n")
        self._write_runtime_status(force=True, event="started")

        while not self._market_switch_ready():
            self._write_runtime_status(event="loop")
            try:
                ws = websocket.WebSocketApp(
                    self._get_ws_url(),
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )

                self._ws = ws

                ws.run_forever(reconnect=5)
                time.sleep(1)

            except deriv_session.DerivAPIError as exc:
                code = f" ({exc.code})" if exc.code else ""

                print(
                    f"[BOT] Falha ao obter WebSocket Options: "
                    f"{exc}{code} — tentando novamente em 5s"
                )

                time.sleep(5)

    def _merge_candles(self, candles):
        merged = {int(c['epoch']): c for c in self._candles}
        for candle in candles:
            c = {k:float(candle[k]) for k in ('open','high','low','close')}
            c['epoch']=int(candle['epoch'])
            if (c['epoch'] % CANDLE_TIMEFRAME_SEC or
                    any(not math.isfinite(c[k]) or c[k] <= 0 for k in ('open','high','low','close')) or
                    not c['low'] <= min(c['open'],c['close']) <= max(c['open'],c['close']) <= c['high']):
                raise ValueError('candle inválido')
            merged[c['epoch']]=c
        ordered=[merged[k] for k in sorted(merged)][-500:]
        # Indicadores não atravessam intervalos ausentes (mercado fechado/desconexão).
        for i in range(len(ordered)-1,0,-1):
            if ordered[i]['epoch']-ordered[i-1]['epoch'] != CANDLE_TIMEFRAME_SEC:
                ordered=ordered[i:]
                break
        self._candles=deque(ordered,maxlen=500)
        self._candle_closes=deque((c['close'] for c in ordered),maxlen=500)
        self._candle_epochs=deque((c['epoch'] for c in ordered),maxlen=500)

    def _warm_adx(self):
        self._adx_history.clear()
        closes=list(self._candle_closes)
        for end in range(29,len(closes)+1):
            value=ind.adx(closes[max(0,end-FEATURE_WINDOW):end],config.ADX_PERIOD)
            if value is not None:
                self._adx_history.append(value)

    def _preload_candles_from_csv(self):
        if not Path(TICKS_CSV).exists():
            return
        records, rejected=read_records(TICKS_CSV)
        records,conflicts,_=canonicalize(records)
        if rejected or conflicts:
            print('[BOT] Histórico local inválido; aguardando OHLC autenticado')
            return
        cutoff=int(time.time())//CANDLE_TIMEFRAME_SEC*CANDLE_TIMEFRAME_SEC
        candles=[]
        for row in records:
            if row['symbol']==self.symbol and row['kind']=='candle' and row['timeframe']==CANDLE_TIMEFRAME_SEC and row['epoch']<cutoff:
                candles.append(dict(epoch=row['epoch'],open=row['open'],high=row['high'],low=row['low'],close=row['price']))
        self._merge_candles(candles)
        self._warm_adx()

    def _on_open(self, ws) -> None:
        print("[BOT] Conectado à Deriv WebSocket API")
        self._write_runtime_status(force=True, event="ws_open")

        if self._uses_bearer_auth():
            accounts=deriv_session.get_options_accounts(app_id=APP_ID,token=TOKEN)
            account=next((a for a in accounts if str(a.get('account_id') or a.get('id'))==self.account_id),None)
            if not account or account.get('account_type') != ('demo' if self.demo else 'real'):
                ws.close()
                raise ValueError('conta Bearer não corresponde à conta e ao modo selecionados')
            self._verify_account(hashlib.sha256(self.account_id.encode()).hexdigest(),float(account['balance']))
            self._start_authenticated_stream(ws)
        else:
            ws.send(json.dumps({"authorize": TOKEN}))

    def _market_switch_pending(self):
        return getattr(self, 'market_switch_requested', lambda: False)()

    def _market_switch_ready(self):
        return self._market_switch_pending() and not self._in_trade

    def _on_message(self, ws, message: str) -> None:
        if self._market_switch_ready():
            ws.close()
            return
        try:
            data = json.loads(message)
        except (TypeError, json.JSONDecodeError) as exc:
            print(f"[BOT] Mensagem WebSocket inválida: {exc}")
            return

        if "error" in data:
            self._handle_api_error(data)
            return

        msg_type = data.get("msg_type", "")

        if msg_type == "authorize":
            self._handle_authorize(
                ws,
                data["authorize"],
            )

        elif msg_type == "tick":
            self._handle_tick(
                ws,
                data["tick"],
            )

        elif msg_type == "proposal":
            self._handle_proposal(
                ws,
                data["proposal"],
                req_id=data.get("req_id"),
            )

        elif msg_type == "buy":
            self._handle_buy(
                ws,
                data["buy"],
                req_id=data.get("req_id"),
            )

        elif msg_type == "candles":
            self._handle_api_candles(
                ws,
                data.get("candles", []),
            )

        elif msg_type == "proposal_open_contract":
            self._handle_contract_update(
                data["proposal_open_contract"]
            )

        elif msg_type == "portfolio":
            self._handle_portfolio(
                ws,
                data.get("portfolio", {}),
                req_id=data.get("req_id"),
            )

    def _on_error(self, ws, error) -> None:
        print(f"[BOT] Erro WebSocket: {error}")

    def _on_close(self, ws, code, msg) -> None:
        self._auth_verified=False
        print(
            f"[BOT] Conexão encerrada "
            f"(código: {code}) — tentando reconectar..."
        )

    # ─────────────────────────────────────────────────────────
    # Lógica de negócio
    # ─────────────────────────────────────────────────────────

    def _handle_authorize(self, ws, auth):
        virtual=auth.get('is_virtual')
        if virtual not in (0,1,False,True) or bool(virtual)!=self.demo or not auth.get('loginid'):
            self._auth_verified=False
            ws.close()
            raise ValueError('conta autenticada não corresponde ao modo solicitado')
        key=hashlib.sha256(str(auth['loginid']).encode()).hexdigest()
        self._verify_account(key,float(auth['balance']))
        self._start_authenticated_stream(ws)

    def _verify_account(self,key,balance):
        saved=getattr(self,'_saved_account_key','')
        if self._in_trade and (not saved or saved!=key):
            raise ValueError('contrato pendente sem identidade de conta compatível; reconciliação necessária')
        self.risk_manager.bind_account(key,balance,preserve_local_balance=self._in_trade)
        self._auth_verified=True

    def _start_authenticated_stream(self, ws) -> None:
        if not self._auth_verified:
            return
        if self._in_trade and not self._open_contract_id:
            print('[BOT] Compra sem confirmação: entradas bloqueadas até reconciliação')
            self._request_portfolio_reconciliation(ws)
        # Re-subscrever ao contrato ativo em caso de reconexão.
        if (
            self._in_trade
            and self._open_contract_id
        ):
            ws.send(
                json.dumps(
                    {
                        "proposal_open_contract": 1,
                        "contract_id": int(
                            self._open_contract_id
                        ),
                        "subscribe": 1,
                    }
                )
            )

            print(
                "[BOT] Re-subscrevendo ao contrato ativo: "
                f"{self._open_contract_id}"
            )

        # Primeira conexão: solicita candles históricos.
        print(
            f"[BOT] Baixando 100 candles históricos "
            f"de {self.symbol} ({CANDLE_TIMEFRAME_SEC}s)..."
        )

        ws.send(
            json.dumps(
                {
                    "ticks_history": self.symbol,
                    "adjust_start_time": 1,
                    "count": 100,
                    "end": "latest",
                    "granularity": CANDLE_TIMEFRAME_SEC,
                    "style": "candles",
                }
            )
        )

    def _request_portfolio_reconciliation(self, ws) -> None:
        self._portfolio_req_id = self._next_request_id()
        ws.send(json.dumps({"portfolio": 1, "req_id": self._portfolio_req_id}))

    def _handle_portfolio(self, ws, portfolio, req_id=None) -> None:
        if not self._in_trade or self._open_contract_id:
            return
        if self._portfolio_req_id is not None and req_id != self._portfolio_req_id:
            return

        contracts = portfolio.get("contracts")
        if contracts is None:
            contracts = portfolio.get("open_contracts")
        if not isinstance(contracts, list):
            contracts = []

        expected_symbol = getattr(self, "_contract_symbol", self.symbol)
        expected_type = "CALL" if self._pending_direction == "BUY" else "PUT"
        matched = None
        for contract in contracts:
            if not isinstance(contract, dict):
                continue
            contract_id = str(contract.get("contract_id") or "")
            if not contract_id.isdigit():
                continue
            symbol = str(
                contract.get("symbol")
                or contract.get("underlying")
                or contract.get("underlying_symbol")
                or ""
            )
            contract_type = str(contract.get("contract_type") or "")
            if symbol and symbol != expected_symbol:
                continue
            if contract_type and contract_type != expected_type:
                continue
            matched = contract
            break

        if matched:
            self._open_contract_id = str(matched["contract_id"])
            self._phase = "open"
            self._in_trade = True
            self._buy_timestamp = self._buy_timestamp or time.time()
            self._set_active_contract_state()
            ws.send(
                json.dumps(
                    {
                        "proposal_open_contract": 1,
                        "contract_id": int(self._open_contract_id),
                        "subscribe": 1,
                    }
                )
            )
            self._record_decision(
                reason="buy_reconciled_from_portfolio",
                signal=self._pending_direction or None,
                indicators=self._pending_indicators,
                tick_epoch=self._last_tick_epoch,
            )
            return

        self._record_decision(
            reason="uncertain_buy_not_in_portfolio",
            signal=self._pending_direction or None,
            indicators=self._pending_indicators,
            tick_epoch=self._last_tick_epoch,
        )
        self._clear_pending()

    def _handle_api_candles(self, ws, candles):
        cutoff=int(time.time())//CANDLE_TIMEFRAME_SEC*CANDLE_TIMEFRAME_SEC
        closed=[c for c in candles if int(c['epoch'])+CANDLE_TIMEFRAME_SEC<=cutoff]
        self._merge_candles(closed)
        self._warm_adx()
        self._api_history_loaded=True
        self._cur_candle_open=None
        current=next((c for c in candles if int(c['epoch'])==cutoff),None)
        if current is not None:
            values=[float(current[k]) for k in ('open','high','low','close')]
            if all(math.isfinite(v) and v>0 for v in values):
                self._cur_candle_epoch=cutoff
                self._cur_candle_open,self._cur_candle_high,self._cur_candle_low,self._cur_candle_close=values
        self._candle_alerts.clear()
        ws.send(json.dumps({'ticks':self.symbol,'subscribe':1}))

    def _handle_tick(self, ws, tick: dict) -> None:
        if tick.get('symbol') != self.symbol or not self._auth_verified:
            return
        try:
            price=float(tick['quote'])
            epoch=int(tick['epoch'])
        except (KeyError,ValueError,TypeError):
            return
        if not math.isfinite(price) or price<=0 or epoch<=self._last_tick_epoch:
            return
        if self._last_tick_price and abs(price/self._last_tick_price-1)>=config.TICK_SPIKE_THRESHOLD:
            return
        if self._last_tick_epoch and epoch-self._last_tick_epoch>MAX_TICK_GAP_SEC:
            self._cur_candle_open=None
            self._candles.clear()
            self._candle_closes.clear()
            self._candle_epochs.clear()
            self._adx_history.clear()
            self._candle_alerts.clear()

        self._last_tick_price = price

        try:
            self._last_tick_epoch = int(
                tick.get("epoch", 0) or 0
            )
        except Exception:
            self._last_tick_epoch = 0

        self._prices.append(price)

        epoch = self._last_tick_epoch

        # P9: reiniciar watchdog a cada tick.
        self._reset_watchdog()
        self._write_runtime_status(tick_epoch=epoch, event="tick")

        if self._phase=='proposal' and time.time()-self._pending_timestamp>PROPOSAL_TIMEOUT_SEC:
            self._clear_pending()
        elif self._phase=='buy' and time.time()-self._pending_timestamp>PROPOSAL_TIMEOUT_SEC:
            self._phase='reconciling'
            self._set_active_contract_state()
        if self._open_contract_id and self._buy_timestamp and time.time()-self._buy_timestamp>duration_seconds(self._pending_duration,DURATION_UNIT)+10:
            if self._contract_poll_timer is None:
                self._poll_contract_result(self._open_contract_id)

        # ── Montar vela de tempo ──────────────────────────────

        self._new_candle_closed = False

        if (
            self._cur_candle_open is None
            and epoch > 0
        ):
            if epoch % CANDLE_TIMEFRAME_SEC > MAX_TICK_GAP_SEC:
                return  # esperar início observado, sem fabricar OHLC de uma vela parcial
            # Primeira vela: alinhar ao início da janela.
            self._cur_candle_epoch = (
                epoch // CANDLE_TIMEFRAME_SEC
            ) * CANDLE_TIMEFRAME_SEC

            self._cur_candle_open = price
            self._cur_candle_high = price
            self._cur_candle_low = price
            self._cur_candle_close = price

        elif (
            epoch > 0
            and self._cur_candle_open is not None
        ):
            if (
                epoch
                < self._cur_candle_epoch
                + CANDLE_TIMEFRAME_SEC
            ):
                # Tick dentro da vela corrente.
                self._cur_candle_high = max(
                    self._cur_candle_high,
                    price,
                )

                self._cur_candle_low = min(
                    self._cur_candle_low,
                    price,
                )

                self._cur_candle_close = price

            else:
                self._merge_candles([dict(epoch=self._cur_candle_epoch,open=self._cur_candle_open,
                    high=self._cur_candle_high,low=self._cur_candle_low,close=self._cur_candle_close)])

                self._new_candle_closed = True
                self._candle_tick_counter = 0

                # Iniciar nova vela alinhada.
                self._cur_candle_epoch = (
                    epoch // CANDLE_TIMEFRAME_SEC
                ) * CANDLE_TIMEFRAME_SEC

                self._cur_candle_open = price
                self._cur_candle_high = price
                self._cur_candle_low = price
                self._cur_candle_close = price

                # Verificar padrões da vela fechada.
                self._check_candle_patterns(price)

        n_candles=len(self._candle_closes)
        if n_candles<MIN_CANDLES:
            return
        self._evaluation_adx_history=list(self._adx_history)
        if self._new_candle_closed:
            value=ind.adx(list(self._candle_closes)[-FEATURE_WINDOW:],config.ADX_PERIOD)
            if value is not None:
                self._adx_history.append(value)

        # Não abrir nova ordem se há contrato ativo.
        if self._in_trade:
            if self._new_candle_closed:
                self._record_decision(
                    reason="active_contract",
                    tick_epoch=self._last_tick_epoch,
                )
            return

        if not self._risk_allows_trade():
            if self._new_candle_closed:
                self._record_decision(
                    reason="risk_block",
                    tick_epoch=self._last_tick_epoch,
                )
            return

        # Cadência: só avaliar sinal quando nova vela fecha.
        if self._new_candle_closed:
            self._candles_since_last_entry += 1

        # Intervalo adaptativo após perdas consecutivas.
        adaptive_interval = ENTRY_CANDLE_INTERVAL

        if hasattr(
            self.risk_manager,
            "consecutive_losses",
        ):
            if self.risk_manager.consecutive_losses >= 2:
                adaptive_interval = (
                    ENTRY_CANDLE_INTERVAL + 1
                )

        # Só avaliar depois do intervalo e de uma vela fechada.
        if (
            getattr(config, "ENFORCE_ENTRY_CADENCE", True)
            and (
                self._candles_since_last_entry < adaptive_interval
                or not self._new_candle_closed
            )
        ):
            remain = max(
                0,
                adaptive_interval
                - self._candles_since_last_entry,
            )

            if not self._new_candle_closed:
                if self._last_waiting_candle_epoch != self._cur_candle_epoch:
                    self._last_waiting_candle_epoch = self._cur_candle_epoch
                    print(
                        f"\r[BOT] {price:.4f} | "
                        f"Aguardando fechamento da vela... "
                        f"(candles ok: {n_candles})   ",
                        end="",
                        flush=True,
                    )
            else:
                self._record_decision(
                    reason="cadence",
                    tick_epoch=self._last_tick_epoch,
                )
                print(
                    f"\r[BOT] {price:.4f} | "
                    f"Próxima entrada em {remain} vela(s)   ",
                    end="",
                    flush=True,
                )

            return

        # P10: ADX mínimo adaptativo.
        candle_closes = list(
            self._candle_closes
        )

        adx_min = get_adaptive_adx_min(
            self._evaluation_adx_history
        )

        signal, indicators = get_signal(
            candle_closes,
            adx_min=adx_min,
            adx_history=self._evaluation_adx_history,
            candle_alerts=list(self._candle_alerts),
            candles=list(self._candles),
            evaluation_time=self._last_tick_epoch,
        )

        self._print_status(
            price,
            indicators,
            signal,
        )

        self._record_decision(
            reason=self._decision_reason(signal, indicators),
            signal=signal,
            indicators=indicators,
            tick_epoch=self._last_tick_epoch,
        )

        if signal in ("BUY", "SELL"):
            sent = self._send_proposal(
                ws,
                signal,
                indicators,
            )
            if sent:
                self._candles_since_last_entry = 0

    def _send_proposal(self,ws,direction,indicators):
        if self._market_switch_pending() or self._in_trade or not self._auth_verified or not self._risk_allows_trade():
            self._record_decision(reason="proposal_precheck_block", signal=direction, indicators=indicators)
            return False
        if direction not in ('BUY','SELL'):
            self._record_decision(reason="invalid_direction", signal=direction, indicators=indicators)
            return False
        if time.time()-self._last_tick_epoch>MAX_TICK_GAP_SEC:
            self._record_decision(reason="stale_tick", signal=direction, indicators=indicators)
            return False
        duration=ai_predictor.predict_duration(list(self._candle_closes))
        if duration!=config.DURATION or DURATION_UNIT!=config.DURATION_UNIT:
            self._record_decision(reason="duration_mismatch", signal=direction, indicators=indicators)
            return False
        model_selected = (
            'trained_model' in indicators.get('decision_modules', [])
            or ('decision_modules' not in indicators and bool(indicators.get('model_id')))
        )
        if model_selected and not indicators.get('model_id'):
            self._record_decision(reason="missing_model_id", signal=direction, indicators=indicators)
            return False
        probability=indicators.get('win_probability')
        if model_selected and (
            probability is None
            or not math.isfinite(probability)
            or not config.AI_CONFIDENCE_MIN < probability <= 1
        ):
            self._record_decision(reason="invalid_win_probability", signal=direction, indicators=indicators)
            return False
        self._pending_payout=self._pending_buy_price=None
        self._proposal_retry_count=0
        self._pending_direction=direction
        self._pending_stake=self.risk_manager.get_stake()
        self._pending_duration=duration
        self._pending_indicators=dict(indicators)
        self._pending_timestamp=time.time()
        self._contract_symbol=self.symbol
        self._saved_account_key=self.risk_manager._account_key
        self._proposal_req_id=self._next_request_id()
        self._phase='proposal'
        self._in_trade=True
        self._set_active_contract_state()
        proposal=dict(proposal=1,amount=self._pending_stake,basis=BASIS,contract_type='CALL' if direction=='BUY' else 'PUT',
                      currency=CURRENCY,duration=duration,duration_unit=DURATION_UNIT,req_id=self._proposal_req_id)
        proposal['underlying_symbol' if self._uses_bearer_auth() else 'symbol']=self.symbol
        ws.send(json.dumps(proposal))
        self._last_contract_attempt_epoch = self._last_tick_epoch
        self._record_decision(reason="proposal_sent", signal=direction, indicators=indicators)
        return True

    def _handle_proposal(self,ws,proposal,req_id=None):
        if self._phase!='proposal' or req_id is None or req_id!=self._proposal_req_id:
            return
        if self._market_switch_pending() or time.time()-self._pending_timestamp>PROPOSAL_TIMEOUT_SEC:
            self._clear_pending()
            return
        probability=self._pending_indicators.get('win_probability')
        price,payout=proposal.get('ask_price'),proposal.get('payout')
        model_selected = (
            'trained_model' in self._pending_indicators.get('decision_modules', [])
            or ('decision_modules' not in self._pending_indicators
                and bool(self._pending_indicators.get('model_id')))
        )
        proposal_valid = (
            profitable_proposal(price, payout, probability)
            if model_selected
            else valid_payout_proposal(price, payout)
        )
        if (not proposal_valid or float(price)>self._pending_stake
                or not proposal.get('id') or not self._risk_allows_trade()):
            self._clear_pending()
            return
        self._pending_buy_price=float(price)
        self._pending_payout=float(payout)
        self._proposal_id=proposal['id']
        self._buy_req_id=self._next_request_id()
        self._phase='buy'
        self._pending_timestamp=time.time()
        # Persiste a intenção ANTES de enviar; crash não autoriza segunda compra.
        self._set_active_contract_state()
        ws.send(json.dumps({'buy':self._proposal_id,'price':self._pending_buy_price,'req_id':self._buy_req_id}))

    def _handle_buy(self,ws,buy,req_id=None):
        if self._phase not in ('buy','reconciling') or req_id is None or req_id!=self._buy_req_id:
            return
        contract_id=str(buy.get('contract_id',''))
        if not contract_id.isdigit():
            return
        self._open_contract_id=contract_id
        self._pending_stake=self._pending_buy_price or self._pending_stake
        self._phase='open'
        self._in_trade=True
        self._proposal_api_error_streak = 0
        self._buy_timestamp=time.time()
        self._pending_timestamp=0.
        self._entry_price=0.
        self._entry_epoch=0  # somente a atualização oficial fornece o spot
        self._set_active_contract_state()
        ws.send(json.dumps({'proposal_open_contract':1,'contract_id':int(contract_id),'subscribe':1}))
        self._schedule_poll(contract_id,duration_seconds(self._pending_duration,DURATION_UNIT)+2)

    def _schedule_poll(self,contract_id,delay=5):
        if self._contract_poll_timer is not None:
            self._contract_poll_timer.cancel()
        self._contract_poll_timer=threading.Timer(delay,self._poll_contract_result,args=[contract_id])
        self._contract_poll_timer.daemon=True
        self._contract_poll_timer.start()

    def _handle_contract_update(self,contract):
        with self._state_lock:
            if str(contract.get('contract_id',''))!=self._open_contract_id:
                return
            for key,attr,cast in [
                ('entry_spot','_entry_price',float),
                ('entry_tick','_entry_price',float),
                ('date_start','_entry_epoch',int),
                ('entry_tick_time','_entry_epoch',int),
            ]:
                if contract.get(key) is not None:
                    setattr(self,attr,cast(contract[key]))
            if contract.get('is_sold') not in (1, True, '1'):
                self._set_active_contract_state()
                return
            try:
                profit=float(contract['profit'])
                if not math.isfinite(profit):
                    return
            except (ValueError,KeyError,TypeError):
                return
            details=dict(duration_unit=DURATION_UNIT,entry_price=contract.get('entry_spot',contract.get('entry_tick','')),
                         exit_price=contract.get('exit_spot',contract.get('exit_tick','')),
                         entry_epoch=contract.get('entry_tick_time',contract.get('date_start','')),exit_epoch=contract.get('exit_spot_time',contract.get('exit_tick_time',contract.get('date_expiry',''))),
                         payout=contract.get('payout',self._pending_payout),buy_price=contract.get('buy_price',self._pending_buy_price))
            self.risk_manager.record_result(symbol=getattr(self,'_contract_symbol',self.symbol),direction=self._pending_direction,
                stake=self._pending_stake,duration=self._pending_duration,profit=profit,indicators=self._pending_indicators,
                contract_id=self._open_contract_id,details=details)
            if self._contract_poll_timer is not None:
                self._contract_poll_timer.cancel()
                self._contract_poll_timer=None
            self._clear_pending()

    def _poll_contract_result(self,contract_id,attempt=0):
        if self._open_contract_id!=contract_id:
            return
        self._contract_poll_timer=None
        try:
            if self._ws is not None:
                self._ws.send(json.dumps({'proposal_open_contract':1,'contract_id':int(contract_id)}))
        finally:
            if self._open_contract_id==contract_id:
                self._schedule_poll(contract_id,15)

    def _reset_watchdog(self) -> None:
        """Reinicia o timer de heartbeat a cada tick."""

        if self._watchdog is not None:
            self._watchdog.cancel()

        self._watchdog = threading.Timer(
            HEARTBEAT_TIMEOUT_SEC,
            self._on_heartbeat_fail,
        )

        self._watchdog.daemon = True
        self._watchdog.start()

    def _on_heartbeat_fail(self) -> None:
        print(
            f"\n[SAÚDE] Nenhum tick recebido por "
            f"{HEARTBEAT_TIMEOUT_SEC}s "
            "— WebSocket pode estar morto "
            "ou mercado fechado; forçando reconexão"
        )
        self._write_runtime_status(
            tick_epoch=self._last_tick_epoch,
            force=True,
            event="heartbeat_timeout",
        )
        ws = self._ws
        if ws is not None:
            try:
                ws.close()
            except Exception as exc:
                print(f"[SAÚDE] Falha ao fechar WebSocket morto: {exc}")

    # ─────────────────────────────────────────────────────────
    # Detecção de padrões de vela
    # ─────────────────────────────────────────────────────────

    def _check_candle_patterns(
        self,
        current_price: float,
    ) -> None:
        """
        Verifica padrões de vela ao fechar cada vela.

        Sempre roda, independentemente de CANDLE_NOTIFY,
        para alimentar _candle_alerts.
        """

        candles=list(self._candles)[-FEATURE_WINDOW:]
        if len(candles)<4:
            return
        patterns = ind.detect_candle_patterns(
            candles
        )

        pa = ind.price_action_features(
            candles,
            PA_SR_TOLERANCE,
        )

        for pattern in patterns:
            if pattern["strength"] < 0.5:
                continue

            context_parts = []

            if pa is not None:
                if pa["pa_demand_zone"] > 0.3:
                    context_parts.append(
                        "Demand Zone"
                    )

                if pa["pa_supply_zone"] > 0.3:
                    context_parts.append(
                        "Supply Zone"
                    )

                if pa["pa_sr_distance"] < 0.3:
                    if pa["pa_sr_position"] < -0.3:
                        context_parts.append(
                            "Suporte próximo"
                        )
                    elif pa["pa_sr_position"] > 0.3:
                        context_parts.append(
                            "Resistência próxima"
                        )

                if pa["pa_fvg_bullish"] > 0.3:
                    context_parts.append(
                        "FVG Bullish"
                    )

                if pa["pa_fvg_bearish"] > 0.3:
                    context_parts.append(
                        "FVG Bearish"
                    )

            context = (
                " + ".join(context_parts)
                if context_parts
                else "Sem contexto especial"
            )

            icon = {
                "bullish": "🟢",
                "bearish": "🔴",
                "neutral": "🟡",
            }.get(
                pattern["direction"],
                "⚪",
            )

            action_map = {
                "bullish": "BUY favorecido",
                "bearish": "SELL favorecido",
                "neutral": "Aguardar",
            }

            action = action_map.get(
                pattern["direction"],
                "",
            )

            alert = {
                "name": pattern["name"],
                "direction": pattern["direction"],
                "strength": pattern["strength"],
                "price": current_price,
                "timestamp": self._last_tick_epoch,
                "context": context,
            }

            self._candle_alerts.append(alert)

            if not CANDLE_NOTIFY:
                continue

            print(
                "\n[VELA] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            )

            print(
                f"  {icon} {pattern['name']} "
                f"(força: {pattern['strength']:.2f})"
            )

            print(
                f"  Preço: {current_price:.4f} | "
                f"Contexto: {context}"
            )

            print(
                f"  Ação sugerida: {action}"
            )

            print(
                "[VELA] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            )

    # ─────────────────────────────────────────────────────────
    # Display
    # ─────────────────────────────────────────────────────────

    def _print_status(
        self,
        price: float,
        indicators: dict,
        signal: Optional[str],
    ) -> None:
        if not indicators:
            return

        adx_v = indicators.get(
            "adx",
            0.0,
        )

        rsi_v = indicators.get(
            "rsi",
            0.0,
        )

        ema9_v = indicators.get(
            "ema9",
            0.0,
        )

        ema21_v = indicators.get(
            "ema21",
            0.0,
        )

        macd_v = indicators.get(
            "macd_hist",
            0.0,
        )

        tag = ""

        if adx_v < 20:
            tag = "[LATERAL]"
        elif signal:
            tag = f"[→ {signal}]"

        print(
            f"\r[TICK] {price:.4f} | "
            f"EMA9:{ema9_v:.4f} "
            f"EMA21:{ema21_v:.4f} | "
            f"RSI:{rsi_v:.1f} "
            f"ADX:{adx_v:.1f} "
            f"MACD_H:{macd_v:+.5f} "
            f"{tag}          ",
            end="",
            flush=True,
        )
