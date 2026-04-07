#!/usr/bin/env python3
"""
pipeline.py — orquestrador automático do pipeline completo.

Executa e mantém rodando em paralelo:
  1. COLETOR    — coleta ticks em background (ticks.csv)
  2. AGUARDA    — espera MIN_TICKS ticks antes do primeiro treino
  3. DATASET    — constrói dataset.csv via dataset_builder.py
  4. TREINO     — treina e salva model.pkl via train_model.py
  5. BOT        — inicia o bot de trading (executor.py)
  6. RE-TREINO  — a cada N minutos, reconstrói dataset e retreina
                  em background (bot continua operando ininterrupto;
                  o novo model.pkl entra na próxima predição)

Uso:
    python pipeline.py                           # demo, padrões de config.py
    python pipeline.py --real                    # modo real (pede confirmação)
    python pipeline.py --history-count 1000      # baixa 1000 ticks históricos
    python pipeline.py --min-ticks 2000          # mín. no CSV antes de treinar
    python pipeline.py --retrain-interval 10     # re-treina a cada 10 min
    python pipeline.py --balance 500             # saldo inicial p/ RiskManager
    python pipeline.py --skip-collect            # usa ticks.csv já existente
    python pipeline.py --force-retrain           # retreina mesmo com model.pkl
    python pipeline.py --no-scan                 # usa o SYMBOL de config.py sem escanear tendência
"""

import csv
import json
import os
import signal
import sys
import threading
import time
from datetime import datetime
from typing import Optional

import websocket

import dataset_builder
import train_model
from config import (
    APP_ID, TOKEN, SYMBOL,
    TICKS_CSV, DATASET_CSV, AI_MODEL_PATH,
    TARGET_LOOKFORWARD,
    MODEL_PROMOTION_MIN_AUC_DELTA,
    MODEL_PROMOTION_MAX_ACC_DROP,
    MODEL_PROMOTION_MAX_F1_DROP,
    DEMO_MODE,
    CANDLE_TIMEFRAME_SEC, MIN_CANDLES,
)
from executor import DerivBot
from risk_manager import RiskManager

# ─────────────────────────────────────────────────────────────────
#  Defaults do pipeline
# ─────────────────────────────────────────────────────────────────

_DEFAULT_HISTORY_COUNT = 500  # candles históricos baixados da API ao iniciar
_DEFAULT_MIN_TICKS     = 100  # mín. no CSV (candles) antes do 1º treino
_DEFAULT_RETRAIN_MIN   = 10   # intervalo de re-treino em minutos
_WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}"

# Nomes descritivos dos índices sintéticos da Deriv
_SYMBOL_NAMES: dict = {
    "R_10":      "Volatility 10 Index",
    "R_25":      "Volatility 25 Index",
    "R_50":      "Volatility 50 Index",
    "R_75":      "Volatility 75 Index",
    "R_100":     "Volatility 100 Index",
    "1HZ10V":    "Volatility 10 (1s) Index",
    "1HZ25V":    "Volatility 25 (1s) Index",
    "1HZ50V":    "Volatility 50 (1s) Index",
    "1HZ75V":    "Volatility 75 (1s) Index",
    "1HZ100V":   "Volatility 100 (1s) Index",
    "BOOM300N":  "Boom 300 Index",
    "BOOM500":   "Boom 500 Index",
    "BOOM1000":  "Boom 1000 Index",
    "CRASH300N": "Crash 300 Index",
    "CRASH500":  "Crash 500 Index",
    "CRASH1000": "Crash 1000 Index",
    "stpRNG":    "Step Index",
    "JD10":      "Jump 10 Index",
    "JD25":      "Jump 25 Index",
    "JD50":      "Jump 50 Index",
    "JD75":      "Jump 75 Index",
    "JD100":     "Jump 100 Index",
}


def _symbol_display(symbol: str) -> str:
    """Retorna 'SYMBOL — Nome Descritivo' ou apenas 'SYMBOL' se não mapeado."""
    name = _SYMBOL_NAMES.get(symbol)
    return f"{symbol} — {name}" if name else symbol


# Evento global: sinaliza encerramento a todas as threads
_shutdown = threading.Event()


# ─────────────────────────────────────────────────────────────────
#  Utilitários
# ─────────────────────────────────────────────────────────────────

def _count_ticks() -> int:
    """Conta linhas de dados em ticks.csv (desconta o cabeçalho)."""
    if not os.path.exists(TICKS_CSV) or os.path.getsize(TICKS_CSV) == 0:
        return 0
    with open(TICKS_CSV, "r") as f:
        return max(0, sum(1 for _ in f) - 1)


def _ensure_ticks_header() -> None:
    """Cria ticks.csv com cabeçalho caso não exista ou esteja vazio."""
    if not os.path.exists(TICKS_CSV) or os.path.getsize(TICKS_CSV) == 0:
        with open(TICKS_CSV, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "datetime", "symbol", "price"])


# ─────────────────────────────────────────────────────────────────
#  PRÉ-FASE — Scan de tendência: elege o melhor índice antes de coletar
# ─────────────────────────────────────────────────────────────────

# Grupo 1: índices de volatilidade (mais líquidos, avaliados primeiro)
_SCAN_SYMBOLS_PRIMARY = [
    "R_10", "R_25", "R_50", "R_75", "R_100",
    "1HZ10V", "1HZ25V", "1HZ50V", "1HZ75V", "1HZ100V",
]
# Grupo 2: índices de boom/crash/jump (consultados só se primários estiverem laterais)
_SCAN_SYMBOLS_SECONDARY = [
    "BOOM500", "BOOM1000", "CRASH500", "CRASH1000",
    "BOOM300N", "CRASH300N",
    "JD10", "JD25", "JD50", "JD75", "JD100",
    "stpRNG",
]
_SCAN_TICKS    = 200   # ticks históricos por símbolo (suficiente p/ ADX/EMA/MACD)
_SCAN_TIMEOUT  = 15    # segundos máximos de espera por símbolo
_ADX_TREND_MIN = 15.0  # score mínimo para considerar o mercado em tendência


def _fetch_prices_for_symbol(symbol: str, count: int) -> list:
    """
    Busca `count` ticks históricos de `symbol` via ticks_history.
    Retorna lista de preços (float) em ordem cronológica, ou [] em caso de erro.
    """
    received: list = []
    done = threading.Event()

    def _on_open(ws):
        ws.send(json.dumps({"authorize": TOKEN}))

    def _on_message(ws, message):
        data = json.loads(message)
        if "error" in data:
            ws.close()
            return
        if data.get("msg_type") == "authorize":
            ws.send(json.dumps({
                "ticks_history":     symbol,
                "end":               "latest",
                "count":             count,
                "style":             "ticks",
                "adjust_start_time": 1,
            }))
            return
        if data.get("msg_type") == "history":
            received.extend(float(p) for p in data.get("history", {}).get("prices", []))
            ws.close()

    def _on_error(ws, err):
        ws.close()

    def _on_close(ws, *_):
        done.set()

    ws = websocket.WebSocketApp(
        _WS_URL,
        on_open=_on_open, on_message=_on_message,
        on_error=_on_error, on_close=_on_close,
    )
    threading.Thread(target=ws.run_forever, daemon=True).start()
    done.wait(timeout=_SCAN_TIMEOUT)
    return received


def _score_trend(prices: list) -> float:
    """
    Pontuação de tendência combinando ADX (força), alinhamento EMA e MACD histogram.
    Quanto maior o valor, mais forte e direcionada a tendência.
    Retorna 0.0 se dados insuficientes.
    """
    import indicators as ind
    from config import EMA_FAST, EMA_SLOW, ADX_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL

    adx_val  = ind.adx(prices, ADX_PERIOD)
    ema9     = ind.ema(prices, EMA_FAST)
    ema21    = ind.ema(prices, EMA_SLOW)
    macd_res = ind.macd(prices, MACD_FAST, MACD_SLOW, MACD_SIGNAL)

    if any(v is None for v in [adx_val, ema9, ema21, macd_res]):
        return 0.0

    _, _, macd_hist = macd_res

    # EMA desalinhada (cruza) → sem tendência clara
    ema_factor  = 1.2 if (ema9 != ema21) else 1.0
    # Histograma MACD reforça o score proporcionalmente à sua magnitude
    last_price  = prices[-1] if prices else 1.0
    macd_factor = 1.0 + min(abs(macd_hist) / (last_price * 0.001 + 1e-10), 0.5)

    return adx_val * ema_factor * macd_factor


def _scan_group(candidates: list) -> dict:
    """Escaneia um grupo de símbolos em paralelo e retorna {symbol: score}."""
    results: dict = {}
    lock = threading.Lock()

    def _scan_one(symbol: str) -> None:
        prices = _fetch_prices_for_symbol(symbol, _SCAN_TICKS)
        score  = _score_trend(prices) if len(prices) >= 50 else 0.0
        direction = ""
        if len(prices) >= 21:
            import indicators as ind
            e9, e21 = ind.ema(prices, 9), ind.ema(prices, 21)
            if e9 is not None and e21 is not None:
                direction = "↑" if e9 > e21 else "↓"
        with lock:
            results[symbol] = score
        print(f"  {symbol:<12} score={score:6.2f} {direction}")

    threads = [
        threading.Thread(target=_scan_one, args=(s,), daemon=True)
        for s in candidates
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=_SCAN_TIMEOUT + 5)
    return results


def _best_from(results: dict) -> tuple:
    """Retorna (melhor_símbolo, score) de um dict de resultados."""
    if not results:
        return "", 0.0
    best = max(results, key=results.__getitem__)
    return best, results[best]


def _detect_trending_symbol(no_scan: bool = False) -> str:
    """
    Etapas:
      1. Escaneia os símbolos primários em paralelo.
      2. Se o melhor tiver score >= _ADX_TREND_MIN → retorna ele.
      3. Se todos estiverem laterais → escaneia o grupo secundário.
      4. Se nenhum grupo tiver tendência → retorna SYMBOL de config.py.
    """
    import config as _cfg

    if no_scan:
        return _cfg.SYMBOL

    print("\n[TENDÊNCIA] ── Scan primário ──────────────────────────────")
    print(f"[TENDÊNCIA] Candidatos: {', '.join(_SCAN_SYMBOLS_PRIMARY)}")
    primary = _scan_group(_SCAN_SYMBOLS_PRIMARY)
    best, score = _best_from(primary)

    if score >= _ADX_TREND_MIN:
        print(f"\n[TENDÊNCIA] ✔ Melhor índice: {_symbol_display(best)} (score={score:.2f})")
        return best

    # Mercado primário lateral → tenta grupo secundário
    print(
        f"\n[TENDÊNCIA] ⚠ Todos os índices primários estão laterais "
        f"(melhor score={score:.2f} < {_ADX_TREND_MIN})."
    )
    print("[TENDÊNCIA] ── Scan secundário ─────────────────────────────")
    print(f"[TENDÊNCIA] Candidatos: {', '.join(_SCAN_SYMBOLS_SECONDARY)}")
    secondary = _scan_group(_SCAN_SYMBOLS_SECONDARY)
    best2, score2 = _best_from(secondary)

    if score2 >= _ADX_TREND_MIN:
        print(f"\n[TENDÊNCIA] ✔ Melhor índice (secundário): {_symbol_display(best2)} (score={score2:.2f})")
        return best2

    # Nenhum grupo com tendência → mantém padrão
    print(
        f"\n[TENDÊNCIA] ⚠ Nenhum índice com tendência clara "
        f"(melhor score secundário={score2:.2f}). "
        f"Usando padrão: {_cfg.SYMBOL}."
    )
    return _cfg.SYMBOL


# ─────────────────────────────────────────────────────────────────
#  FASE 0 — Busca de ticks históricos via API Deriv
# ─────────────────────────────────────────────────────────────────

def _fetch_historical_ticks(count: int) -> int:
    """
    Baixa os últimos `count` candles de CANDLE_TIMEFRAME_SEC segundos via API Deriv
    e grava os closes em ticks.csv em ordem cronológica.

    Epochs já presentes no arquivo são ignorados (deduplicação) para não
    conflitar com o coletor ao vivo que inicia logo em seguida.

    Retorna o número de candles (closes) efetivamente gravados.
    """
    tf_min = CANDLE_TIMEFRAME_SEC // 60
    print(
        f"\n[HISTÓRICO] Buscando os últimos {count:,} candles de {tf_min} min "
        f"de '{_symbol_display(SYMBOL)}'..."
    )

    _ensure_ticks_header()

    # Carrega (epoch, symbol) já existentes para deduplicação por símbolo
    existing_epochs: set = set()
    if os.path.exists(TICKS_CSV) and os.path.getsize(TICKS_CSV) > 0:
        with open(TICKS_CSV, "r") as f:
            for row in csv.DictReader(f):
                try:
                    existing_epochs.add((int(row["epoch"]), row.get("symbol", "")))
                except (KeyError, ValueError):
                    pass

    received: list = []
    done      = threading.Event()
    error_msg: list = []

    def _on_open(ws):
        ws.send(json.dumps({"authorize": TOKEN}))

    def _on_message(ws, message):
        data = json.loads(message)

        if "error" in data:
            error_msg.append(data["error"]["message"])
            ws.close()
            return

        if data.get("msg_type") == "authorize":
            ws.send(json.dumps({
                "ticks_history": SYMBOL,
                "end":           "latest",
                "count":         count,
                "style":         "candles",
                "granularity":   CANDLE_TIMEFRAME_SEC,
                "adjust_start_time": 1,
            }))
            return

        if data.get("msg_type") == "candles":
            for candle in data.get("candles", []):
                epoch = int(candle.get("epoch", 0))
                close = float(candle.get("close", candle.get("open", 0)))
                received.append((epoch, close))
            ws.close()

    def _on_error(ws, err):
        error_msg.append(str(err))
        ws.close()

    def _on_close(ws, code, msg):
        done.set()

    ws = websocket.WebSocketApp(
        _WS_URL,
        on_open=_on_open,
        on_message=_on_message,
        on_error=_on_error,
        on_close=_on_close,
    )
    t = threading.Thread(target=ws.run_forever, daemon=True)
    t.start()
    done.wait(timeout=30)

    if error_msg:
        print(f"[HISTÓRICO] Erro da API: {error_msg[0]}")
        return 0

    if not received:
        print("[HISTÓRICO] Nenhum candle histórico recebido.")
        return 0

    # Grava em ordem cronológica, pulando epochs já existentes
    written = 0
    with open(TICKS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        for epoch, close in received:
            if (epoch, SYMBOL) not in existing_epochs:
                dt_str = datetime.fromtimestamp(epoch).isoformat()
                writer.writerow([epoch, dt_str, SYMBOL, close])
                existing_epochs.add((epoch, SYMBOL))
                written += 1

    print(
        f"[HISTÓRICO] {written:,} candles de {tf_min}min gravados — "
        f"total no CSV: {_count_ticks():,}"
    )
    return written


def _banner(history_count: int, min_ticks: int, retrain_min: int, demo: bool) -> None:
    mode = "DEMO" if demo else "REAL 💸"
    tf_min = CANDLE_TIMEFRAME_SEC // 60
    print("=" * 60)
    print("  Deriv Pipeline — Orquestrador Automático")
    print(f"  Modo              : {mode}")
    print(f"  Símbolo           : {_symbol_display(SYMBOL)}")
    print(f"  Candles históricos : {history_count:,} × {tf_min} min")
    print(f"  Candles p/ treino  : {min_ticks:,}")
    print(f"  Re-treino a cada  : {retrain_min} min")
    print("  Pressione Ctrl+C para encerrar tudo")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────
#  FASE 1 — Coletor em thread daemon
# ─────────────────────────────────────────────────────────────────

class _CollectorThread(threading.Thread):
    """Thread de coleta de ticks. Roda em background com reconexão automática."""

    def __init__(self) -> None:
        super().__init__(daemon=True, name="Collector")
        self._ws: Optional[websocket.WebSocketApp] = None
        self._local_count = 0
        # Maior epoch já gravado (histórico incluído): descarta duplicatas ao vivo
        self._last_epoch: int = self._load_last_epoch()

    def _load_last_epoch(self) -> int:
        """Lê o maior epoch já gravado em ticks.csv para evitar duplicatas."""
        last = 0
        if not os.path.exists(TICKS_CSV) or os.path.getsize(TICKS_CSV) == 0:
            return last
        try:
            with open(TICKS_CSV, "r") as f:
                for row in csv.DictReader(f):
                    try:
                        e = int(row["epoch"])
                        if e > last:
                            last = e
                    except (KeyError, ValueError):
                        pass
        except Exception:
            pass
        return last

    def run(self) -> None:
        _ensure_ticks_header()
        while not _shutdown.is_set():
            try:
                ws = websocket.WebSocketApp(
                    _WS_URL,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                self._ws = ws
                ws.run_forever(reconnect=5)
            except Exception as exc:
                if not _shutdown.is_set():
                    print(f"\n[COLETOR] Exceção inesperada: {exc} — reconectando em 5s")
                    time.sleep(5)

    def stop(self) -> None:
        if self._ws:
            self._ws.close()

    def _on_open(self, ws) -> None:
        print(f"\n[COLETOR] Conectado | {_symbol_display(SYMBOL)}")
        ws.send(json.dumps({"authorize": TOKEN}))
        ws.send(json.dumps({"ticks": SYMBOL, "subscribe": 1}))

    def _on_message(self, ws, message: str) -> None:
        data = json.loads(message)

        if "error" in data:
            print(f"\n[COLETOR] Erro API: {data['error']['message']}")
            return

        if data.get("msg_type") == "authorize":
            print("[COLETOR] Autorizado. Coleta iniciada...")
            return

        if "tick" in data:
            tick   = data["tick"]
            epoch  = tick["epoch"]
            price  = tick["quote"]

            # Descarta ticks já gravados pelo histórico
            if epoch <= self._last_epoch:
                return
            self._last_epoch = epoch

            dt_str = datetime.fromtimestamp(epoch).isoformat()
            self._local_count += 1

            with open(TICKS_CSV, "a", newline="") as f:
                csv.writer(f).writerow([epoch, dt_str, SYMBOL, price])

            print(
                f"\r[COLETOR] Ticks: {self._local_count:>7,} | "
                f"{price} @ {dt_str}",
                end="", flush=True,
            )

    def _on_error(self, ws, error) -> None:
        if not _shutdown.is_set():
            print(f"\n[COLETOR] Erro WebSocket: {error}")

    def _on_close(self, ws, code, msg) -> None:
        if not _shutdown.is_set():
            print(f"\n[COLETOR] Conexão encerrada (código: {code}) — reconectando...")


# ─────────────────────────────────────────────────────────────────
#  FASE 2 — Aguardar ticks mínimos
# ─────────────────────────────────────────────────────────────────

def _wait_for_ticks(min_ticks: int, poll_sec: int = 10) -> None:
    """Bloqueia até ticks.csv conter pelo menos min_ticks linhas de dados (candles)."""
    current = _count_ticks()
    if current >= min_ticks:
        print(f"[PIPELINE] {current:,} candles já disponíveis — pronto para treino.")
        return

    print(
        f"\n[PIPELINE] Aguardando {min_ticks:,} candles para o primeiro treino "
        f"(atual: {current:,})..."
    )
    while not _shutdown.is_set():
        time.sleep(poll_sec)
        current = _count_ticks()
        print(
            f"\r[PIPELINE] Candles: {current:,} / {min_ticks:,}   ",
            end="", flush=True,
        )
        if current >= min_ticks:
            print()
            return

    sys.exit(0)  # shutdown acionado antes de atingir mínimo


# ─────────────────────────────────────────────────────────────────
#  FASES 3 e 4 — Construir dataset e treinar modelo
# ─────────────────────────────────────────────────────────────────

def _trim_ticks(ticks_path: str, max_ticks: int = 50000) -> None:
    """Limita o arquivo em ticks_path aos últimos max_ticks mais recentes."""
    import pandas as pd
    if not os.path.exists(ticks_path) or os.path.getsize(ticks_path) == 0:
        return
    try:
        df = pd.read_csv(ticks_path)
        if len(df) > max_ticks:
            trimmed = len(df) - max_ticks
            df = df.tail(max_ticks)
            df.to_csv(ticks_path, index=False)
            print(f"[PIPELINE] Ticks trimados: {trimmed:,} removidos, {max_ticks:,} mantidos.")
    except Exception as e:
        print(f"[PIPELINE] Aviso: não foi possível trimar ticks: {e}")


def _load_latest_final_metrics() -> Optional[dict]:
    """Lê a última entrada de métricas com stage='final' em model_metrics.json."""
    metrics_path = os.path.join(os.path.dirname(os.path.abspath(AI_MODEL_PATH)), "model_metrics.json")
    if not os.path.exists(metrics_path):
        return None
    try:
        with open(metrics_path, "r") as f:
            hist = json.load(f)
        if not isinstance(hist, list):
            return None
        for item in reversed(hist):
            if isinstance(item, dict) and item.get("stage") == "final":
                return item
    except Exception:
        return None
    return None


def _should_promote_model(current: Optional[dict], challenger: Optional[dict]) -> tuple[bool, str]:
    """Decide promoção de challenger com regras conservadoras de não-regressão."""
    if challenger is None:
        return False, "métricas do challenger indisponíveis"
    if current is None:
        return True, "sem champion anterior com métricas finais"

    cur_auc = float(current.get("auc", 0.0))
    cur_acc = float(current.get("accuracy", 0.0))
    cur_f1 = float(current.get("f1", 0.0))

    ch_auc = float(challenger.get("auc", 0.0))
    ch_acc = float(challenger.get("accuracy", 0.0))
    ch_f1 = float(challenger.get("f1", 0.0))

    auc_ok = ch_auc >= (cur_auc + MODEL_PROMOTION_MIN_AUC_DELTA)
    acc_ok = ch_acc >= (cur_acc - MODEL_PROMOTION_MAX_ACC_DROP)
    f1_ok = ch_f1 >= (cur_f1 - MODEL_PROMOTION_MAX_F1_DROP)

    if auc_ok and acc_ok and f1_ok:
        return True, (
            f"AUC {ch_auc:.4f} >= {cur_auc:.4f}+{MODEL_PROMOTION_MIN_AUC_DELTA:.4f}, "
            f"ACC {ch_acc:.4f} sem queda excessiva, F1 {ch_f1:.4f} sem queda excessiva"
        )

    return False, (
        f"gate reprovado | champion(AUC={cur_auc:.4f}, ACC={cur_acc:.4f}, F1={cur_f1:.4f}) "
        f"vs challenger(AUC={ch_auc:.4f}, ACC={ch_acc:.4f}, F1={ch_f1:.4f})"
    )


def _run_training(
    dataset_path: str = DATASET_CSV,
    model_path: str = AI_MODEL_PATH,
    test_ratio: float = 0.2,
) -> tuple[bool, Optional[dict]]:
    """
    Constrói dataset.csv e treina model.pkl.

    Copia ticks.csv para um snapshot temporário antes de trimar, de modo que
    o coletor ao vivo nunca leia um arquivo parcialmente truncado.

    Retorna (sucesso, métricas_finais).
    """
    import shutil

    print("\n[PIPELINE] ── Fase 3: Construindo dataset ──")
    tmp_ticks = f"{TICKS_CSV}.train_snapshot"
    try:
        # Snapshot do CSV — o original não é alterado enquanto o coletor grava
        shutil.copy2(TICKS_CSV, tmp_ticks)

        # Filtrar snapshot para o símbolo ativo antes do trim:
        # Quando o símbolo muda (ex: R_100 → R_10), o CSV tem a maioria dos
        # ticks do símbolo anterior; filtrar garante que o treino usa só os
        # ticks relevantes, sem ser engolido pelos dados do símbolo antigo.
        try:
            import pandas as _pd
            import config as _cfg
            _active_sym = _cfg.get_active_symbol()
            _snap_df = _pd.read_csv(tmp_ticks)
            if "symbol" in _snap_df.columns and _active_sym:
                _before = len(_snap_df)
                _snap_df = _snap_df[_snap_df["symbol"] == _active_sym]
                if len(_snap_df) < _before:
                    print(
                        f"[PIPELINE] Snapshot filtrado para '{_active_sym}': "
                        f"{len(_snap_df):,} ticks ({_before - len(_snap_df):,} de outros símbolos removidos)"
                    )
                _snap_df.to_csv(tmp_ticks, index=False)
        except Exception:
            pass  # falha silenciosa — dataset_builder fará sua própria filtragem

        _trim_ticks(tmp_ticks, max_ticks=200000)
        n = dataset_builder.build_dataset(
            tmp_ticks,
            dataset_path,
            window_size=50,
            lookforward=TARGET_LOOKFORWARD,
        )
        if n < 30:
            print(f"[PIPELINE] Dataset muito pequeno ({n} linhas). Colete mais ticks.")
            return False, None
    except SystemExit:
        return False, None
    except Exception as exc:
        print(f"[PIPELINE] Erro ao construir dataset: {exc}")
        return False, None
    finally:
        if os.path.exists(tmp_ticks):
            try:
                os.remove(tmp_ticks)
            except OSError:
                pass

    print("\n[PIPELINE] ── Fase 4: Treinando modelo ──")
    metrics: Optional[dict] = None
    try:
        metrics = train_model.train(dataset_path, model_path, test_ratio)
    except SystemExit:
        return False, None
    except Exception as exc:
        print(f"[PIPELINE] Erro ao treinar modelo: {exc}")
        return False, None

    return True, metrics


# ─────────────────────────────────────────────────────────────────
#  FASE 6 — Re-treino periódico em background
# ─────────────────────────────────────────────────────────────────

def _reset_ai_predictor() -> None:
    """
    Reseta o singleton do ai_predictor para que o novo model.pkl
    seja carregado na próxima inferência (sem reiniciar o bot).
    """
    try:
        import ai_predictor
        ai_predictor._model          = None
        ai_predictor._model_loaded   = False
        ai_predictor._model_features = []
        ai_predictor._dur_model          = None
        ai_predictor._dur_model_loaded   = False
        ai_predictor._dur_model_features = []
        ai_predictor._tft_model         = None
        ai_predictor._tft_model_loaded  = False
        print("[RE-TREINO] Singleton da IA resetado — novo modelo ativo na próxima predição.")
    except Exception:
        pass


def _retrain_loop(interval_min: int) -> None:
    """
    Thread de re-treino periódico.

    Aguarda `interval_min` minutos, reconstrói dataset em arquivo
    temporário e substitui model.pkl atomicamente (os.replace).
    O bot nunca para durante o processo.
    """
    interval_sec = interval_min * 60

    while not _shutdown.is_set():
        # Dorme em fatias de 5s para responder rápido ao shutdown
        elapsed = 0
        while elapsed < interval_sec and not _shutdown.is_set():
            time.sleep(5)
            elapsed += 5

        if _shutdown.is_set():
            break

        n_ticks = _count_ticks()
        print(
            f"\n[RE-TREINO] Iniciando re-treino automático "
            f"({n_ticks:,} ticks disponíveis)..."
        )

        # Usa arquivos temporários para não corromper os arquivos em uso
        tmp_dataset = f"{DATASET_CSV}.tmp"
        tmp_model   = f"{AI_MODEL_PATH}.new"

        champion_metrics = _load_latest_final_metrics()
        success, challenger_metrics = _run_training(dataset_path=tmp_dataset, model_path=tmp_model)

        if success and os.path.exists(tmp_model):
            promote, reason = _should_promote_model(champion_metrics, challenger_metrics)
            if promote:
                # Substituição atômica: o bot nunca lê um arquivo pela metade
                os.replace(tmp_model, AI_MODEL_PATH)
                _reset_ai_predictor()
                print(
                    f"[RE-TREINO] model.pkl atualizado com sucesso ({reason}). "
                    f"Próximo re-treino em {interval_min} min."
                )
            else:
                print(f"[RE-TREINO] Challenger não promovido: {reason}")
        else:
            print("[RE-TREINO] Falhou — bot continua com o modelo anterior.")

        # Limpeza de temporários
        for tmp in (tmp_dataset, tmp_model):
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass


# ─────────────────────────────────────────────────────────────────
#  Confirmação modo real
# ─────────────────────────────────────────────────────────────────

def _confirm_real_mode() -> bool:
    print("\n" + "!" * 55)
    print("  ATENÇÃO: Modo REAL ativado.")
    print("  Este pipeline usará DINHEIRO REAL na sua conta Deriv.")
    print("  Certifique-se de que o TOKEN em config.py é de")
    print("  conta REAL e tem permissões de 'trade'.")
    print("!" * 55)
    answer = input("\nDigite 'sim' para confirmar: ").strip().lower()
    return answer == "sim"


# ─────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Pipeline automático: coleta → dataset → treino → bot → re-treino.\n"
            "Tudo em um único comando, sem intervenção manual."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--real", action="store_true",
        help="Força modo real (dinheiro real). Requer confirmação.",
    )
    mode_group.add_argument(
        "--demo", action="store_true",
        help="Força modo demo (ignora config.DEMO_MODE).",
    )

    parser.add_argument(
        "--balance", type=float, default=1000.0, metavar="USD",
        help="Saldo inicial para o RiskManager (padrão: 1000.0).",
    )
    parser.add_argument(
        "--min-ticks", type=int, default=_DEFAULT_MIN_TICKS, metavar="N",
        help=f"Ticks mínimos antes do 1º treino (padrão: {_DEFAULT_MIN_TICKS}).",
    )
    parser.add_argument(
        "--retrain-interval", type=int, default=_DEFAULT_RETRAIN_MIN, metavar="MIN",
        help=f"Intervalo de re-treino em minutos (padrão: {_DEFAULT_RETRAIN_MIN}).",
    )
    parser.add_argument(
        "--skip-collect", action="store_true",
        help="Não inicia o coletor (usa ticks.csv já existente).",
    )
    parser.add_argument(
        "--force-retrain", action="store_true",
        help="Reconstrói dataset e retreina mesmo que model.pkl já exista.",
    )
    parser.add_argument(
        "--history-count", type=int, default=_DEFAULT_HISTORY_COUNT, metavar="N",
        help=f"Ticks históricos a baixar da API antes de iniciar (padrão: {_DEFAULT_HISTORY_COUNT}).",
    )
    parser.add_argument(
        "--no-scan", action="store_true",
        help="Pula o scan de tendência — usa o SYMBOL definido em config.py diretamente.",
    )

    args = parser.parse_args()

    # ── Determina modo demo/real ───────────────────────────────
    if args.real:
        is_demo = False
    elif args.demo:
        is_demo = True
    else:
        is_demo = DEMO_MODE

    if not is_demo:
        if not _confirm_real_mode():
            print("Operação cancelada.")
            sys.exit(0)

    _banner(args.history_count, args.min_ticks, args.retrain_interval, is_demo)

    # ── Handler global de Ctrl+C ───────────────────────────────
    def _handle_interrupt(sig, frame) -> None:
        print("\n\n[PIPELINE] Encerrando tudo... aguarde.")
        _shutdown.set()
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_interrupt)

    # ── PRÉ-FASE: Detectar índice com maior tendência ──────────
    import config as _cfg
    detected = _detect_trending_symbol(no_scan=args.no_scan)
    if detected != _cfg.SYMBOL:
        print(f"[TENDÊNCIA] Símbolo alterado: {_cfg.SYMBOL} → {detected}")
        # Propaga o novo símbolo para todos os módulos que o importaram
        import executor, strategy, dataset_builder, train_model
        for _mod in (_cfg, executor, strategy, dataset_builder, train_model):
            if hasattr(_mod, "SYMBOL"):
                setattr(_mod, "SYMBOL", detected)
        # Atualiza o getter de símbolo ativo em config
        _cfg.set_active_symbol(detected)
        # Atualiza a variável deste módulo (importada com 'from config import SYMBOL')
        globals()["SYMBOL"] = detected

    # Salva símbolo ativo em state.json para o dashboard
    try:
        import json as _json
        _state_path = os.path.join(os.path.dirname(TICKS_CSV), "state.json")
        _state = {}
        if os.path.exists(_state_path):
            try:
                with open(_state_path) as _sf:
                    _state = _json.load(_sf)
            except Exception:
                _state = {}
        _state["symbol"] = detected
        with open(_state_path, "w") as _sf:
            _json.dump(_state, _sf)
    except Exception:
        pass

    # ── FASE 0: Histórico inicial ──────────────────────────────
    if not args.skip_collect:
        n_hist = _fetch_historical_ticks(args.history_count)
        if n_hist == 0:
            print("[PIPELINE] Aviso: histórico não obtido — coletor ao vivo cobrirá os dados.")
    else:
        print("[PIPELINE] Fase 0: --skip-collect ativo — histórico ignorado.")

    # ── FASE 1: Coletor ────────────────────────────────────────
    collector: Optional[_CollectorThread] = None
    if not args.skip_collect:
        collector = _CollectorThread()   # já carrega _last_epoch do histórico gravado
        collector.start()
        print("[PIPELINE] Fase 1: Coletor iniciado em background (sem duplicatas).")
    else:
        n_existing = _count_ticks()
        print(
            f"[PIPELINE] Fase 1: --skip-collect ativo. "
            f"Usando '{TICKS_CSV}' existente ({n_existing:,} ticks)."
        )

    # ── FASES 2/3/4: Treino inicial ────────────────────────────
    model_exists = os.path.exists(AI_MODEL_PATH) and not args.force_retrain
    if model_exists:
        print(
            f"\n[PIPELINE] Fases 2-4: model.pkl encontrado — treino inicial ignorado.\n"
            f"           Use --force-retrain para forçar novo treino."
        )
    else:
        # Aguarda ticks suficientes (fase 2)
        _wait_for_ticks(args.min_ticks)

        # Constrói dataset e treina (fases 3 e 4)
        success, _metrics = _run_training()
        if not success:
            print("\n[PIPELINE] Treino inicial falhou. Encerrando.")
            _shutdown.set()
            sys.exit(1)

    # ── FASE 5: Re-treino periódico ────────────────────────────
    retrain_thread = threading.Thread(
        target=_retrain_loop,
        args=(args.retrain_interval,),
        daemon=True,
        name="Retrain",
    )
    retrain_thread.start()
    print(
        f"\n[PIPELINE] Fase 5: Re-treino automático agendado "
        f"a cada {args.retrain_interval} min."
    )

    # ── FASE 6: Bot ────────────────────────────────────────────
    print("\n[PIPELINE] Fase 6: Iniciando bot de trading...\n")
    risk_manager = RiskManager(initial_balance=args.balance)
    bot = DerivBot(risk_manager=risk_manager, demo=is_demo)
    bot.run()  # bloqueia até Ctrl+C


if __name__ == "__main__":
    main()
