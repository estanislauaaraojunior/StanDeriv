"""
executor.py — cliente WebSocket da Deriv e execução de ordens.

Responsabilidades:
  - Gerenciar ciclo de vida da conexão WebSocket
  - Autorizar token e subscrever fluxo de ticks
  - Chamar strategy.get_signal() a cada tick
  - Enviar proposals de compra/venda via API Deriv
  - Rastrear contratos abertos e registrar resultado via RiskManager

Fluxo de uma operação:
  tick → get_signal() → send_proposal → handle_proposal (buy) → handle_contract (result)
"""

import json
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

import websocket

from config import (
    APP_ID, TOKEN, SYMBOL,
    DURATION, DURATION_UNIT, BASIS, CURRENCY,
    MIN_TICKS, ENTRY_TICK_INTERVAL,
    PRICE_BUFFER_SIZE,
    PROPOSAL_TIMEOUT_SEC,
    HEARTBEAT_TIMEOUT_SEC,
    CANDLE_SIZE, CANDLE_NOTIFY, PA_SR_TOLERANCE,
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
        demo:         True  → exibe avisos de modo demo
                      False → opera em conta real (requer TOKEN de conta real)
    """

    def __init__(self, risk_manager: RiskManager, demo: bool = True) -> None:
        self.risk_manager = risk_manager
        self.demo = demo
        self._ws_url = f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}"

        # P12: Buffer maior para indicadores mais estáveis (500 ticks)
        self._prices: deque = deque(maxlen=PRICE_BUFFER_SIZE)

        # P10: Histórico de ADX para cálculo adaptativo do limiar
        self._adx_history: deque = deque(maxlen=500)

        # Alertas de padrões de vela (últimos 20)
        self._candle_alerts: deque = deque(maxlen=20)
        self._candle_tick_counter: int = 0

        # Estado da operação em andamento
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

        # P1: Timestamp da última proposal enviada (detecta proposals sem resposta)
        self._pending_timestamp: float = 0.0

        # P7: Timestamp da abertura do contrato (detecta contratos travados)
        self._buy_timestamp: float = 0.0

        # P13: Timer de poll ativo — consulta o contrato após duração esperada
        self._contract_poll_timer: Optional[threading.Timer] = None

        # P9: Watchdog de heartbeat (alerta se nenhum tick chegar em N segundos)
        self._watchdog: Optional[threading.Timer] = None

        # Cadência de entradas: começa já no limite para disparar na 1ª oportunidade
        # após o aquecimento (aprendizado dos 500 candles históricos já aplicado).
        self._ticks_since_last_entry: int = ENTRY_TICK_INTERVAL

        self._ws: Optional[websocket.WebSocketApp] = None

    def _set_active_contract_state(self, clear: bool = False) -> None:
        """Sincroniza contrato ativo no state.json para consumo do dashboard."""
        try:
            state = {}
            if _STATE_JSON.exists() and _STATE_JSON.stat().st_size > 0:
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
    #  API pública
    # ─────────────────────────────────────────────────────────

    def run(self) -> None:
        """Inicia o bot. Bloqueia até encerramento."""
        self._set_active_contract_state(clear=True)
        mode = "DEMO" if self.demo else "REAL 💸"
        print(f"\n{'=' * 55}")
        print(f"  Deriv Trading Bot — Modo: {mode}")
        print(f"  Símbolo : {SYMBOL}")
        print(f"  Duração : {DURATION} {DURATION_UNIT}")
        print(f"  Saldo   : {self.risk_manager.balance:.2f} USD")
        print(f"  Stake   : {self.risk_manager.get_stake():.2f} USD (1% do saldo)")
        print(f"{'=' * 55}\n")

        ws = websocket.WebSocketApp(
            self._ws_url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self._ws = ws
        ws.run_forever(reconnect=5)

    # ─────────────────────────────────────────────────────────
    #  Handlers WebSocket
    # ─────────────────────────────────────────────────────────

    def _on_open(self, ws) -> None:
        print("[BOT] Conectado à Deriv WebSocket API")
        ws.send(json.dumps({"authorize": TOKEN}))

    def _on_message(self, ws, message: str) -> None:
        data = json.loads(message)

        if "error" in data:
            print(f"[BOT] Erro API: {data['error']['message']}")
            return

        msg_type = data.get("msg_type", "")

        if msg_type == "authorize":
            self._handle_authorize(ws, data["authorize"])
        elif msg_type == "tick":
            self._handle_tick(ws, data["tick"])
        elif msg_type == "proposal":
            self._handle_proposal(ws, data["proposal"])
        elif msg_type == "buy":
            self._handle_buy(ws, data["buy"])
        elif msg_type == "proposal_open_contract":
            self._handle_contract_update(data["proposal_open_contract"])

    def _on_error(self, ws, error) -> None:
        print(f"[BOT] Erro WebSocket: {error}")

    def _on_close(self, ws, code, msg) -> None:
        print(f"[BOT] Conexão encerrada (código: {code}) — tentando reconectar...")

    # ─────────────────────────────────────────────────────────
    #  Lógica de negócio
    # ─────────────────────────────────────────────────────────

    def _handle_authorize(self, ws, auth: dict) -> None:
        balance = float(auth.get("balance", self.risk_manager.balance))
        self.risk_manager.balance = balance
        print(f"[BOT] Autorizado | Saldo real: {balance:.2f} USD")
        ws.send(json.dumps({"ticks": SYMBOL, "subscribe": 1}))

        # Re-subscrever ao contrato ativo em caso de reconexão
        if self._in_trade and self._open_contract_id:
            ws.send(json.dumps({
                "proposal_open_contract": 1,
                "contract_id":           int(self._open_contract_id),
                "subscribe":             1,
            }))
            print(f"[BOT] Re-subscrevendo ao contrato ativo: {self._open_contract_id}")

    def _handle_tick(self, ws, tick: dict) -> None:
        price = float(tick["quote"])
        self._last_tick_price = price
        try:
            self._last_tick_epoch = int(tick.get("epoch", 0) or 0)
        except Exception:
            self._last_tick_epoch = 0
        self._prices.append(price)

        # P9: Reiniciar watchdog a cada tick recebido
        self._reset_watchdog()

        # Aguardar acúmulo de ticks para indicadores estáveis
        n = len(self._prices)
        if n < MIN_TICKS:
            print(f"\r[BOT] Aquecendo... {n}/{MIN_TICKS} ticks", end="", flush=True)
            return

        # Verificar padrões de vela a cada CANDLE_SIZE ticks
        self._candle_tick_counter += 1
        if self._candle_tick_counter >= CANDLE_SIZE:
            self._candle_tick_counter = 0
            self._check_candle_patterns(price)

        # Incrementa o contador de ticks desde a última entrada
        self._ticks_since_last_entry += 1

        # P1: Verificar timeout de proposal sem resposta da API
        if self._in_trade and self._pending_timestamp > 0 and self._open_contract_id is None:
            if time.time() - self._pending_timestamp > PROPOSAL_TIMEOUT_SEC:
                print(f"\n[BOT] Proposal sem resposta por {PROPOSAL_TIMEOUT_SEC}s — cancelando")
                self._in_trade = False
                self._pending_timestamp = 0.0

        # P7: Verificar timeout de contrato aberto sem resultado
        if self._in_trade and self._open_contract_id and self._buy_timestamp > 0:
            contract_timeout = max(self._pending_duration * 3, 60)
            if time.time() - self._buy_timestamp > contract_timeout:
                print(
                    f"\n[BOT] Contrato {self._open_contract_id} sem resultado "
                    f"após {contract_timeout}s — resetando estado"
                )
                # P13: cancelar poll ativo junto com o reset
                if self._contract_poll_timer is not None:
                    self._contract_poll_timer.cancel()
                    self._contract_poll_timer = None
                self._in_trade = False
                self._open_contract_id = None
                self._buy_timestamp = 0.0
                self._entry_price = 0.0
                self._entry_epoch = 0
                self._set_active_contract_state(clear=True)

        # Não abrir nova ordem se já há contrato ativo
        if self._in_trade:
            return

        if not self.risk_manager.can_trade():
            return

        # Cadência adaptativa: ajustar intervalo com base no desempenho
        adaptive_interval = ENTRY_TICK_INTERVAL
        if hasattr(self.risk_manager, "consecutive_losses"):
            if self.risk_manager.consecutive_losses >= 2:
                adaptive_interval = int(ENTRY_TICK_INTERVAL * 1.5)

        if self._ticks_since_last_entry < adaptive_interval:
            remaining = adaptive_interval - self._ticks_since_last_entry
            print(
                f"\r[BOT] {price:.4f} | Próxima entrada em {remaining} ticks   ",
                end="", flush=True,
            )
            return

        # P10: Calcular ADX mínimo adaptativo e armazenar histórico
        prices_list = list(self._prices)
        adx_min = get_adaptive_adx_min(list(self._adx_history))
        signal, indicators = get_signal(prices_list, adx_min=adx_min, adx_history=list(self._adx_history))

        # Atualizar histórico de ADX para próximas chamadas
        if indicators.get("adx"):
            self._adx_history.append(indicators["adx"])

        self._print_status(price, indicators, signal)

        if signal in ("BUY", "SELL"):
            self._send_proposal(ws, signal, indicators)
            self._ticks_since_last_entry = 0  # reinicia contagem após entrada

    def _send_proposal(self, ws, direction: str, indicators: dict) -> None:
        contract_type = "CALL" if direction == "BUY" else "PUT"
        stake = self.risk_manager.get_stake()

        # Duração escolhida pela IA (fallback para DURATION se modelo não existir)
        duration = ai_predictor.predict_duration(list(self._prices))

        # Salva estado da operação pendente
        self._pending_direction   = direction
        self._pending_stake       = stake
        self._pending_duration    = duration
        self._pending_indicators  = indicators
        self._pending_timestamp   = time.time()  # P1: marca início do timeout

        proposal = {
            "proposal":       1,
            "amount":         stake,
            "basis":          BASIS,
            "contract_type":  contract_type,
            "currency":       CURRENCY,
            "duration":       duration,
            "duration_unit":  DURATION_UNIT,
            "symbol":         SYMBOL,
        }
        ws.send(json.dumps(proposal))
        print(
            f"\n[BOT] → {direction} | Stake: {stake:.2f} USD | Duração: {duration}t | "
            f"ADX:{indicators.get('adx', '?'):.1f} RSI:{indicators.get('rsi', '?'):.1f}"
        )

    def _handle_proposal(self, ws, proposal: dict) -> None:
        proposal_id = proposal["id"]
        self._in_trade = True
        ws.send(json.dumps({"buy": proposal_id, "price": self._pending_stake}))

    def _handle_buy(self, ws, buy: dict) -> None:
        contract_id = str(buy.get("contract_id", ""))
        self._open_contract_id = contract_id
        self._pending_timestamp = 0.0          # P1: proposal foi aceita, limpar timeout
        self._buy_timestamp = time.time()      # P7: marcar abertura do contrato
        self._entry_price = self._last_tick_price
        self._entry_epoch = self._last_tick_epoch
        self._set_active_contract_state(clear=False)
        print(f"[BOT] Contrato aberto | ID: {contract_id}")

        # Subscrever atualizações do contrato para capturar o resultado
        if contract_id:
            ws.send(json.dumps({
                "proposal_open_contract": 1,
                "contract_id":           int(contract_id),
                "subscribe":             1,
            }))

        # P13: Agendar poll ativo caso a subscrição perca o resultado final
        # (contratos de poucos ticks expiram antes da subscrição ser confirmada)
        if self._contract_poll_timer is not None:
            self._contract_poll_timer.cancel()
        poll_delay = max(self._pending_duration + 2, 5)  # mínimo de 5s
        self._contract_poll_timer = threading.Timer(
            poll_delay, self._poll_contract_result, args=[contract_id]
        )
        self._contract_poll_timer.daemon = True
        self._contract_poll_timer.start()

    def _handle_contract_update(self, contract: dict) -> None:
        # Ignorar atualizações de contratos anteriores ou de outras subscrições
        if str(contract.get("contract_id", "")) != self._open_contract_id:
            return
        if not contract.get("is_sold"):
            return  # contrato ainda aberto

        profit = float(contract.get("profit", 0.0))

        self.risk_manager.record_result(
            symbol     = SYMBOL,
            direction  = self._pending_direction,
            stake      = self._pending_stake,
            duration   = self._pending_duration,
            profit     = profit,
            indicators = self._pending_indicators,
        )

        self._in_trade         = False
        self._open_contract_id = None
        self._buy_timestamp    = 0.0  # P7: limpar após contrato encerrado
        self._entry_price      = 0.0
        self._entry_epoch      = 0
        self._set_active_contract_state(clear=True)

        # P13: cancelar poll ativo pois resultado já foi recebido
        if self._contract_poll_timer is not None:
            self._contract_poll_timer.cancel()
            self._contract_poll_timer = None

    # ─────────────────────────────────────────────────────────
    #  P13 — Poll ativo de resultado de contrato
    # ─────────────────────────────────────────────────────────

    def _poll_contract_result(self, contract_id: str, attempt: int = 0) -> None:
        """Consulta pontual (sem subscribe) do estado do contrato.

        Disparada após a duração esperada do contrato.  Caso a subscrição
        passiva tenha perdido o evento `is_sold`, esta consulta garante que
        `_handle_contract_update` receberá o resultado e desbloqueará o bot.

        Tenta até 6 vezes (a cada 5 s) antes de desistir e deixar o timeout
        de 60 s resetar o estado.
        """
        if self._open_contract_id != contract_id:
            return  # contrato já encerrado pelo caminho normal
        ws = self._ws
        if ws is None:
            return
        print(f"\n[BOT] Poll de resultado para contrato {contract_id} (tentativa {attempt + 1})...")
        ws.send(json.dumps({
            "proposal_open_contract": 1,
            "contract_id":           int(contract_id),
        }))
        # Reagendar enquanto o contrato ainda não foi resolvido (máx 6 tentativas)
        if attempt < 5:
            self._contract_poll_timer = threading.Timer(
                5, self._poll_contract_result, args=[contract_id, attempt + 1]
            )
            self._contract_poll_timer.daemon = True
            self._contract_poll_timer.start()

    # ─────────────────────────────────────────────────────────
    #  P9 — Watchdog de heartbeat
    # ─────────────────────────────────────────────────────────

    def _reset_watchdog(self) -> None:
        """Cancela e reinicia o timer de heartbeat a cada tick recebido."""
        if self._watchdog is not None:
            self._watchdog.cancel()
        self._watchdog = threading.Timer(HEARTBEAT_TIMEOUT_SEC, self._on_heartbeat_fail)
        self._watchdog.daemon = True
        self._watchdog.start()

    def _on_heartbeat_fail(self) -> None:
        print(
            f"\n[SAÚDE] Nenhum tick recebido por {HEARTBEAT_TIMEOUT_SEC}s "
            "— WebSocket pode estar morto ou mercado fechado"
        )

    # ─────────────────────────────────────────────────────────
    #  Detecção de padrões de vela
    # ─────────────────────────────────────────────────────────

    def _check_candle_patterns(self, current_price: float) -> None:
        """Verifica padrões de vela e exibe notificação no terminal."""
        if not CANDLE_NOTIFY:
            return

        prices_list = list(self._prices)
        candles = ind.ticks_to_candles(prices_list, CANDLE_SIZE)
        if len(candles) < 4:
            return

        patterns = ind.detect_candle_patterns(candles)
        pa = ind.price_action_features(candles, PA_SR_TOLERANCE)

        for p in patterns:
            if p["strength"] < 0.5:
                continue

            # Contexto PA
            context_parts = []
            if pa is not None:
                if pa["pa_demand_zone"] > 0.3:
                    context_parts.append("Demand Zone")
                if pa["pa_supply_zone"] > 0.3:
                    context_parts.append("Supply Zone")
                if pa["pa_sr_distance"] < 0.3:
                    if pa["pa_sr_position"] < -0.3:
                        context_parts.append("Suporte próximo")
                    elif pa["pa_sr_position"] > 0.3:
                        context_parts.append("Resistência próxima")
                if pa["pa_fvg_bullish"] > 0.3:
                    context_parts.append("FVG Bullish")
                if pa["pa_fvg_bearish"] > 0.3:
                    context_parts.append("FVG Bearish")

            context = " + ".join(context_parts) if context_parts else "Sem contexto especial"

            icon = {"bullish": "🟢", "bearish": "🔴", "neutral": "🟡"}.get(p["direction"], "⚪")
            action_map = {"bullish": "BUY favorecido", "bearish": "SELL favorecido", "neutral": "Aguardar"}
            action = action_map.get(p["direction"], "")

            alert = {
                "name": p["name"],
                "direction": p["direction"],
                "strength": p["strength"],
                "price": current_price,
                "timestamp": time.time(),
                "context": context,
            }
            self._candle_alerts.append(alert)

            print(f"\n[VELA] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print(f"  {icon} {p['name']} (força: {p['strength']:.2f})")
            print(f"  Preço: {current_price:.4f} | Contexto: {context}")
            print(f"  Ação sugerida: {action}")
            print(f"[VELA] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # ─────────────────────────────────────────────────────────
    #  Display
    # ─────────────────────────────────────────────────────────

    def _print_status(self, price: float, indicators: dict, signal: Optional[str]) -> None:
        if not indicators:
            return

        adx_v   = indicators.get("adx", 0.0)
        rsi_v   = indicators.get("rsi", 0.0)
        ema9_v  = indicators.get("ema9", 0.0)
        ema21_v = indicators.get("ema21", 0.0)
        mach_v  = indicators.get("macd_hist", 0.0)

        tag = ""
        if adx_v < 20:
            tag = "[LATERAL]"
        elif signal:
            tag = f"[→ {signal}]"

        print(
            f"\r[TICK] {price:.4f} | "
            f"EMA9:{ema9_v:.4f} EMA21:{ema21_v:.4f} | "
            f"RSI:{rsi_v:.1f} ADX:{adx_v:.1f} MACD_H:{mach_v:+.5f} "
            f"{tag}          ",
            end="",
            flush=True,
        )
