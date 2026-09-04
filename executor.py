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
    MIN_TICKS,
    MIN_CANDLES,
    ENTRY_TICK_INTERVAL,
    ENTRY_CANDLE_INTERVAL,
    PRICE_BUFFER_SIZE,
    PROPOSAL_TIMEOUT_SEC,
    HEARTBEAT_TIMEOUT_SEC,
    CANDLE_SIZE,
    CANDLE_NOTIFY,
    PA_SR_TOLERANCE,
    CANDLE_TIMEFRAME_SEC,
    TICKS_CSV,
)

from risk_manager import RiskManager
from strategy import get_signal, get_adaptive_adx_min
import ai_predictor
import indicators as ind


_STATE_JSON = Path(__file__).resolve().parent / "state.json"


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
        self.risk_manager = risk_manager
        self.demo = demo
        self.auth_mode = auth_mode
        self.account_id = account_id
        self._next_ws_url = initial_ws_url

        self._ws_url = (
            f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}"
        )

        # P12: Buffer de ticks brutos.
        self._prices: deque = deque(maxlen=PRICE_BUFFER_SIZE)

        # Buffer de candles de tempo.
        self._candle_closes: deque = deque(maxlen=500)
        self._candle_epochs: deque = deque(maxlen=500)

        # Vela em construção.
        self._cur_candle_open: Optional[float] = None
        self._cur_candle_high: float = 0.0
        self._cur_candle_low: float = 0.0
        self._cur_candle_close: float = 0.0
        self._cur_candle_epoch: int = 0

        # Sinaliza que uma nova vela acabou de ser fechada neste tick.
        self._new_candle_closed: bool = False

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

    def _set_active_contract_state(self, clear: bool = False) -> None:
        """Sincroniza contrato ativo no state.json para o dashboard."""
        try:
            state = {}

            if (
                _STATE_JSON.exists()
                and _STATE_JSON.stat().st_size > 0
            ):
                with _STATE_JSON.open("r", encoding="utf-8") as f:
                    state = json.load(f)

            if clear:
                state["active_contract"] = None
            else:
                state["active_contract"] = {
                    "has_active": True,
                    "contract_id": self._open_contract_id,
                    "symbol": SYMBOL,
                    "direction": self._pending_direction,
                    "duration": self._pending_duration,
                    "buy_timestamp": self._buy_timestamp,
                    "entry_epoch": self._entry_epoch,
                    "entry_price": self._entry_price,
                }

            with _STATE_JSON.open("w", encoding="utf-8") as f:
                json.dump(state, f)

        except Exception:
            pass

    # ─────────────────────────────────────────────────────────
    # API pública
    # ─────────────────────────────────────────────────────────

    def run(self) -> None:
        """Inicia o bot. Bloqueia até encerramento."""
        self._set_active_contract_state(clear=True)
        self._preload_candles_from_csv()

        mode = "DEMO" if self.demo else "REAL 💸"

        print(f"\n{'=' * 55}")
        print(f"  Deriv Trading Bot — Modo: {mode}")
        print(f"  Símbolo : {SYMBOL}")
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

        while True:
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

    def _preload_candles_from_csv(self) -> None:
        """
        Pré-carrega closes de candles históricos do ticks.csv
        para o símbolo ativo, eliminando o período de aquecimento.
        """
        try:
            import csv as _csv

            ticks_path = Path(TICKS_CSV)

            if (
                not ticks_path.exists()
                or ticks_path.stat().st_size == 0
            ):
                return

            ticks_list = []

            with ticks_path.open("r", encoding="utf-8") as f:
                reader = _csv.DictReader(f)

                for row in reader:
                    try:
                        sym = row.get("symbol", "")

                        if sym != SYMBOL:
                            continue

                        ticks_list.append(
                            {
                                "epoch": int(row["epoch"]),
                                "price": float(row["price"]),
                            }
                        )

                    except (KeyError, ValueError):
                        continue

            if len(ticks_list) < 60:
                return

            candles = ind.ticks_to_candles_by_time(
                ticks_list,
                CANDLE_TIMEFRAME_SEC,
            )

            if not candles:
                return

            # Carregar apenas os últimos 490 candles.
            for candle in candles[-490:]:
                self._candle_closes.append(candle["close"])
                self._candle_epochs.append(candle["epoch"])

            # Pré-aquecer histórico de ADX.
            closes = list(self._candle_closes)

            for i in range(14, len(closes)):
                adx_value = ind.adx(
                    closes[max(0, i - 50):i],
                    14,
                )

                if adx_value:
                    self._adx_history.append(adx_value)

            n = len(self._candle_closes)

            print(
                f"[BOT] Pré-carregados {n} candles históricos "
                f"de {SYMBOL} — aquecimento instantâneo."
            )

        except Exception as exc:
            print(
                f"[BOT] Aviso: pré-carga de candles falhou "
                f"({exc}) — aguardando ao vivo."
            )

    # ─────────────────────────────────────────────────────────
    # Handlers WebSocket
    # ─────────────────────────────────────────────────────────

    def _on_open(self, ws) -> None:
        print("[BOT] Conectado à Deriv WebSocket API")

        if self._uses_bearer_auth():
            self._start_authenticated_stream(ws)
        else:
            ws.send(json.dumps({"authorize": TOKEN}))

    def _on_message(self, ws, message: str) -> None:
        try:
            data = json.loads(message)
        except (TypeError, json.JSONDecodeError) as exc:
            print(f"[BOT] Mensagem WebSocket inválida: {exc}")
            return

        if "error" in data:
            error = data.get("error") or {}
            print(
                f"[BOT] Erro API: "
                f"{error.get('message', 'Erro desconhecido')}"
            )
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
            )

        elif msg_type == "buy":
            self._handle_buy(
                ws,
                data["buy"],
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

    def _on_error(self, ws, error) -> None:
        print(f"[BOT] Erro WebSocket: {error}")

    def _on_close(self, ws, code, msg) -> None:
        print(
            f"[BOT] Conexão encerrada "
            f"(código: {code}) — tentando reconectar..."
        )

    # ─────────────────────────────────────────────────────────
    # Lógica de negócio
    # ─────────────────────────────────────────────────────────

    def _handle_authorize(self, ws, auth: dict) -> None:
        balance = float(
            auth.get(
                "balance",
                self.risk_manager.balance,
            )
        )

        self.risk_manager.balance = balance

        # Bug #6: sincroniza saldo inicial diário
        # com o saldo real da API.
        self.risk_manager._daily_start_balance = balance

        print(
            f"[BOT] Autorizado | "
            f"Saldo real: {balance:.2f} USD"
        )

        self._start_authenticated_stream(ws)

    def _start_authenticated_stream(self, ws) -> None:
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

        # Reconexão: candles já estão em memória.
        if self._api_history_loaded:
            ws.send(
                json.dumps(
                    {
                        "ticks": SYMBOL,
                        "subscribe": 1,
                    }
                )
            )
            return

        # Primeira conexão: solicita candles históricos.
        print(
            f"[BOT] Baixando 100 candles históricos "
            f"de {SYMBOL} ({CANDLE_TIMEFRAME_SEC}s)..."
        )

        ws.send(
            json.dumps(
                {
                    "ticks_history": SYMBOL,
                    "adjust_start_time": 1,
                    "count": 100,
                    "end": "latest",
                    "granularity": CANDLE_TIMEFRAME_SEC,
                    "style": "candles",
                }
            )
        )

    def _handle_api_candles(
        self,
        ws,
        candles: list,
    ) -> None:
        """Processa candles OHLC recebidos da API Deriv."""

        if not candles:
            print(
                "[BOT] Histórico de candles vazio "
                "— subscrevendo ticks direto."
            )

            self._api_history_loaded = True

            ws.send(
                json.dumps(
                    {
                        "ticks": SYMBOL,
                        "subscribe": 1,
                    }
                )
            )
            return

        # Popula closes e epochs.
        # O último candle pode ainda estar aberto.
        loaded = 0

        for candle in candles[:-1]:
            try:
                self._candle_closes.append(
                    float(candle["close"])
                )

                self._candle_epochs.append(
                    int(candle["epoch"])
                )

                loaded += 1

            except (KeyError, ValueError, TypeError):
                continue

        # Pré-aquecer histórico de ADX.
        closes = list(self._candle_closes)

        for i in range(14, len(closes)):
            adx_value = ind.adx(
                closes[max(0, i - 50):i],
                14,
            )

            if adx_value:
                self._adx_history.append(adx_value)

        self._api_history_loaded = True

        print(
            f"[BOT] {loaded} candles históricos "
            f"carregados via API — aquecimento instantâneo."
        )

        ws.send(
            json.dumps(
                {
                    "ticks": SYMBOL,
                    "subscribe": 1,
                }
            )
        )

    def _handle_tick(self, ws, tick: dict) -> None:
        price = float(tick["quote"])

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

        # ── Montar vela de tempo ──────────────────────────────

        self._new_candle_closed = False

        if (
            self._cur_candle_open is None
            and epoch > 0
        ):
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
                # Fechar vela corrente.
                self._candle_closes.append(
                    self._cur_candle_close
                )

                self._candle_epochs.append(
                    self._cur_candle_epoch
                )

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

        # ── Aquecimento ───────────────────────────────────────

        n_candles = len(self._candle_closes)

        if n_candles < MIN_CANDLES:
            remain_sec = (
                (MIN_CANDLES - n_candles)
                * CANDLE_TIMEFRAME_SEC
                // 60
            )

            print(
                f"\r[BOT] Aquecendo... "
                f"{n_candles}/{MIN_CANDLES} velas"
                f" (~{remain_sec} min restantes)",
                end="",
                flush=True,
            )

            return

        # P1: timeout de proposal sem resposta.
        if (
            self._in_trade
            and self._pending_timestamp > 0
            and self._open_contract_id is None
        ):
            if (
                time.time()
                - self._pending_timestamp
                > PROPOSAL_TIMEOUT_SEC
            ):
                print(
                    f"\n[BOT] Proposal sem resposta por "
                    f"{PROPOSAL_TIMEOUT_SEC}s — cancelando"
                )

                self._in_trade = False
                self._pending_timestamp = 0.0
                self._pending_direction = ""
                self._pending_stake = 0.0

        # P7: timeout de contrato aberto.
        if (
            self._in_trade
            and self._open_contract_id
            and self._buy_timestamp > 0
        ):
            contract_timeout = max(
                self._pending_duration * 60 * 3,
                180,
            )

            if (
                time.time()
                - self._buy_timestamp
                > contract_timeout
            ):
                print(
                    f"\n[BOT] Contrato "
                    f"{self._open_contract_id} sem resultado "
                    f"após {contract_timeout}s — resetando estado"
                )

                if self._contract_poll_timer is not None:
                    self._contract_poll_timer.cancel()
                    self._contract_poll_timer = None

                self._in_trade = False
                self._open_contract_id = None
                self._buy_timestamp = 0.0
                self._entry_price = 0.0
                self._entry_epoch = 0

                self._set_active_contract_state(
                    clear=True
                )

        # Não abrir nova ordem se há contrato ativo.
        if self._in_trade:
            return

        if not self.risk_manager.can_trade():
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
            self._candles_since_last_entry
            < adaptive_interval
            or not self._new_candle_closed
        ):
            remain = max(
                0,
                adaptive_interval
                - self._candles_since_last_entry,
            )

            if not self._new_candle_closed:
                print(
                    f"\r[BOT] {price:.4f} | "
                    f"Aguardando fechamento da vela... "
                    f"(candles ok: {n_candles})   ",
                    end="",
                    flush=True,
                )
            else:
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
            list(self._adx_history)
        )

        signal, indicators = get_signal(
            candle_closes,
            adx_min=adx_min,
            adx_history=list(self._adx_history),
            candle_alerts=list(self._candle_alerts),
        )

        # Atualizar histórico de ADX.
        if indicators.get("adx"):
            self._adx_history.append(
                indicators["adx"]
            )

        self._print_status(
            price,
            indicators,
            signal,
        )

        if signal in ("BUY", "SELL"):
            self._send_proposal(
                ws,
                signal,
                indicators,
            )

            self._candles_since_last_entry = 0

    def _send_proposal(
        self,
        ws,
        direction: str,
        indicators: dict,
    ) -> None:
        contract_type = (
            "CALL"
            if direction == "BUY"
            else "PUT"
        )

        stake = self.risk_manager.get_stake()

        # Duração escolhida pela IA.
        duration = ai_predictor.predict_duration(
            list(self._candle_closes)
        )

        # Guardar estado da operação pendente.
        self._pending_direction = direction
        self._pending_stake = stake
        self._pending_duration = duration
        self._pending_indicators = indicators
        self._pending_timestamp = time.time()

        proposal = {
            "proposal": 1,
            "amount": stake,
            "basis": BASIS,
            "contract_type": contract_type,
            "currency": CURRENCY,
            "duration": duration,
            "duration_unit": DURATION_UNIT,
        }

        if self._uses_bearer_auth():
            proposal["underlying_symbol"] = SYMBOL
        else:
            proposal["symbol"] = SYMBOL

        ws.send(json.dumps(proposal))

        duration_unit_label = (
            DURATION_UNIT
            if DURATION_UNIT in ("t", "s", "m")
            else DURATION_UNIT
        )

        print(
            f"\n[BOT] → {direction} | "
            f"Stake: {stake:.2f} USD | "
            f"Duração: {duration}"
            f"{duration_unit_label} | "
            f"ADX:{indicators.get('adx', 0):.1f} "
            f"RSI:{indicators.get('rsi', 0):.1f}"
        )

    def _handle_proposal(
        self,
        ws,
        proposal: dict,
    ) -> None:
        proposal_id = proposal["id"]

        self._in_trade = True

        ws.send(
            json.dumps(
                {
                    "buy": proposal_id,
                    "price": self._pending_stake,
                }
            )
        )

    def _handle_buy(
        self,
        ws,
        buy: dict,
    ) -> None:
        contract_id = str(
            buy.get("contract_id", "")
        )

        self._open_contract_id = contract_id

        # Proposal aceita.
        self._pending_timestamp = 0.0

        # P7: marcar abertura.
        self._buy_timestamp = time.time()

        self._entry_price = self._last_tick_price
        self._entry_epoch = self._last_tick_epoch

        self._set_active_contract_state(
            clear=False
        )

        print(
            f"[BOT] Contrato aberto | ID: {contract_id}"
        )

        # Subscrever atualizações do contrato.
        if contract_id:
            ws.send(
                json.dumps(
                    {
                        "proposal_open_contract": 1,
                        "contract_id": int(contract_id),
                        "subscribe": 1,
                    }
                )
            )

        # P13: agendar poll ativo.
        if self._contract_poll_timer is not None:
            self._contract_poll_timer.cancel()

        # Duração em minutos → segundos.
        poll_delay = max(
            self._pending_duration * 60 + 2,
            5,
        )

        self._contract_poll_timer = threading.Timer(
            poll_delay,
            self._poll_contract_result,
            args=[contract_id],
        )

        self._contract_poll_timer.daemon = True
        self._contract_poll_timer.start()

    def _handle_contract_update(
        self,
        contract: dict,
    ) -> None:
        # Ignorar contratos anteriores.
        if (
            str(contract.get("contract_id", ""))
            != self._open_contract_id
        ):
            return

        if not contract.get("is_sold"):
            return

        profit = float(
            contract.get("profit", 0.0)
        )

        self.risk_manager.record_result(
            symbol=SYMBOL,
            direction=self._pending_direction,
            stake=self._pending_stake,
            duration=self._pending_duration,
            profit=profit,
            indicators=self._pending_indicators,
        )

        self._in_trade = False
        self._open_contract_id = None
        self._buy_timestamp = 0.0
        self._entry_price = 0.0
        self._entry_epoch = 0

        self._set_active_contract_state(
            clear=True
        )

        # P13: cancelar poll.
        if self._contract_poll_timer is not None:
            self._contract_poll_timer.cancel()
            self._contract_poll_timer = None

    # ─────────────────────────────────────────────────────────
    # P13 — Poll ativo de resultado
    # ─────────────────────────────────────────────────────────

    def _poll_contract_result(
        self,
        contract_id: str,
        attempt: int = 0,
    ) -> None:
        """
        Consulta pontual do estado do contrato.

        Disparada após a duração esperada. Caso a subscrição
        passiva tenha perdido o evento final, a consulta tenta
        recuperar o resultado.

        Tenta até 6 vezes, uma a cada 5 segundos.
        """

        if self._open_contract_id != contract_id:
            return

        ws = self._ws

        if ws is None:
            return

        print(
            f"\n[BOT] Poll de resultado para contrato "
            f"{contract_id} "
            f"(tentativa {attempt + 1})..."
        )

        ws.send(
            json.dumps(
                {
                    "proposal_open_contract": 1,
                    "contract_id": int(contract_id),
                }
            )
        )

        if attempt < 5:
            self._contract_poll_timer = threading.Timer(
                5,
                self._poll_contract_result,
                args=[
                    contract_id,
                    attempt + 1,
                ],
            )

            self._contract_poll_timer.daemon = True
            self._contract_poll_timer.start()

    # ─────────────────────────────────────────────────────────
    # P9 — Watchdog de heartbeat
    # ─────────────────────────────────────────────────────────

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
            "ou mercado fechado"
        )

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

        closes = list(
            self._candle_closes
        )

        if len(closes) < 4:
            return

        # Constrói candles a partir da série de closes.
        candles = (
            ind.ticks_to_candles(
                closes,
                max(4, len(closes) // 20),
            )
            if len(closes) >= 4
            else []
        )

        if len(candles) < 4:
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
                "timestamp": time.time(),
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