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
    python pipeline.py
    python pipeline.py --real
    python pipeline.py --history-count 1000
    python pipeline.py --min-ticks 2000
    python pipeline.py --balance 500
    python pipeline.py --skip-collect
    python pipeline.py --force-retrain
    python pipeline.py --no-scan
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


def _reexec_with_project_venv() -> None:
    """Relança o pipeline com a .venv local quando chamado via Python global."""
    project_dir = os.path.dirname(os.path.abspath(__file__))
    venv_python = os.path.join(project_dir, ".venv", "bin", "python")

    if not os.path.exists(venv_python):
        return

    if os.path.abspath(sys.executable) == os.path.abspath(venv_python):
        return

    os.execv(
        venv_python,
        [venv_python, *sys.argv],
    )


_reexec_with_project_venv()

import websocket

import dataset_builder
import deriv_session
import train_model

from config import (
    APP_ID,
    TOKEN,
    SYMBOL,
    DERIV_AUTH_MODE,
    TICKS_CSV,
    DATASET_CSV,
    AI_MODEL_PATH,
    TRANSFORMER_MODEL_PATH,
    DURATION_MODEL_PATH,
    TARGET_LOOKFORWARD,
    MODEL_PROMOTION_MIN_AUC_DELTA,
    MODEL_PROMOTION_MAX_ACC_DROP,
    MODEL_PROMOTION_MAX_F1_DROP,
    DEMO_MODE,
    CANDLE_TIMEFRAME_SEC,
    MIN_CANDLES,
    TICK_SPIKE_THRESHOLD,
    RETRAIN_EVAL_WINDOW,
    RETRAIN_WIN_RATE_TRIGGER,
    RETRAIN_CONFIDENCE_TRIGGER,
    RETRAIN_NEW_TICKS_TRIGGER,
    RETRAIN_MIN_INTERVAL_SEC,
    RETRAIN_MAX_INTERVAL_SEC,
    OPERATIONS_LOG,
)

from executor import DerivBot
from risk_manager import RiskManager


# ─────────────────────────────────────────────────────────────────
# Defaults do pipeline
# ─────────────────────────────────────────────────────────────────

_DEFAULT_HISTORY_COUNT = 500
_DEFAULT_MIN_TICKS = 100

_WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}"

_AUTH_MODE = "legacy"
_PAT_ACCOUNT_ID: Optional[str] = None
_PAT_ACCOUNT: dict = {}
_PAT_WS_URL: Optional[str] = None


# ─────────────────────────────────────────────────────────────────
# Nomes descritivos dos instrumentos da Deriv
# ─────────────────────────────────────────────────────────────────

_SYMBOL_NAMES: dict = {
    # Índices sintéticos
    "R_10": "Volatility 10 Index",
    "R_25": "Volatility 25 Index",
    "R_50": "Volatility 50 Index",
    "R_75": "Volatility 75 Index",
    "R_100": "Volatility 100 Index",
    "1HZ10V": "Volatility 10 (1s) Index",
    "1HZ25V": "Volatility 25 (1s) Index",
    "1HZ50V": "Volatility 50 (1s) Index",
    "1HZ75V": "Volatility 75 (1s) Index",
    "1HZ100V": "Volatility 100 (1s) Index",
    "BOOM300N": "Boom 300 Index",
    "BOOM500": "Boom 500 Index",
    "BOOM1000": "Boom 1000 Index",
    "CRASH300N": "Crash 300 Index",
    "CRASH500": "Crash 500 Index",
    "CRASH1000": "Crash 1000 Index",
    "stpRNG": "Step Index",
    "JD10": "Jump 10 Index",
    "JD25": "Jump 25 Index",
    "JD50": "Jump 50 Index",
    "JD75": "Jump 75 Index",
    "JD100": "Jump 100 Index",

    # Forex — Majors
    "frxEURUSD": "Euro / US Dollar",
    "frxGBPUSD": "British Pound / US Dollar",
    "frxUSDJPY": "US Dollar / Japanese Yen",
    "frxAUDUSD": "Australian Dollar / US Dollar",
    "frxUSDCAD": "US Dollar / Canadian Dollar",
    "frxUSDCHF": "US Dollar / Swiss Franc",
    "frxNZDUSD": "New Zealand Dollar / US Dollar",

    # Forex — Crosses
    "frxEURGBP": "Euro / British Pound",
    "frxEURJPY": "Euro / Japanese Yen",
    "frxEURCHF": "Euro / Swiss Franc",
    "frxEURCAD": "Euro / Canadian Dollar",
    "frxEURNZD": "Euro / New Zealand Dollar",
    "frxEURAUD": "Euro / Australian Dollar",
    "frxGBPJPY": "British Pound / Japanese Yen",
    "frxGBPCHF": "British Pound / Swiss Franc",
    "frxGBPCAD": "British Pound / Canadian Dollar",
    "frxGBPAUD": "British Pound / Australian Dollar",
    "frxGBPNZD": "British Pound / New Zealand Dollar",
    "frxAUDJPY": "Australian Dollar / Japanese Yen",
    "frxAUDCAD": "Australian Dollar / Canadian Dollar",
    "frxAUDNZD": "Australian Dollar / New Zealand Dollar",
    "frxAUDCHF": "Australian Dollar / Swiss Franc",
    "frxCADJPY": "Canadian Dollar / Japanese Yen",
    "frxCADCHF": "Canadian Dollar / Swiss Franc",
    "frxCHFJPY": "Swiss Franc / Japanese Yen",
    "frxNZDJPY": "New Zealand Dollar / Japanese Yen",
    "frxNZDCAD": "New Zealand Dollar / Canadian Dollar",
}


def _symbol_display(symbol: str) -> str:
    """Retorna 'SYMBOL — Nome Descritivo' ou apenas 'SYMBOL'."""
    name = _SYMBOL_NAMES.get(symbol)
    return f"{symbol} — {name}" if name else symbol


# Evento global: sinaliza encerramento a todas as threads
_shutdown = threading.Event()


# ─────────────────────────────────────────────────────────────────
# Utilitários
# ─────────────────────────────────────────────────────────────────

def _count_ticks() -> int:
    """Conta linhas de dados em ticks.csv, descontando o cabeçalho."""
    if (
        not os.path.exists(TICKS_CSV)
        or os.path.getsize(TICKS_CSV) == 0
    ):
        return 0

    try:
        with open(TICKS_CSV, "r", encoding="utf-8") as f:
            return max(0, sum(1 for _ in f) - 1)
    except Exception:
        return 0


def _ensure_ticks_header() -> None:
    """Cria ticks.csv com cabeçalho caso não exista ou esteja vazio."""
    if (
        not os.path.exists(TICKS_CSV)
        or os.path.getsize(TICKS_CSV) == 0
    ):
        with open(
            TICKS_CSV,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            csv.writer(f).writerow(
                [
                    "epoch",
                    "datetime",
                    "symbol",
                    "price",
                ]
            )


def _uses_bearer_auth() -> bool:
    return (
        _AUTH_MODE == "bearer"
        and bool(_PAT_ACCOUNT_ID)
    )


def _get_ws_url() -> str:
    global _PAT_WS_URL

    if not _uses_bearer_auth():
        return _WS_URL

    if _PAT_WS_URL:
        ws_url = _PAT_WS_URL
        _PAT_WS_URL = None
        return ws_url

    return deriv_session.get_options_ws_url(
        str(_PAT_ACCOUNT_ID),
        app_id=APP_ID,
        token=TOKEN,
        attempts=3,
    )


# ─────────────────────────────────────────────────────────────────
# PRÉ-FASE — Scan de tendência
# ─────────────────────────────────────────────────────────────────

_SCAN_SYMBOLS_FOREX = [
    "frxEURUSD",
    "frxGBPUSD",
    "frxUSDJPY",
    "frxAUDUSD",
    "frxUSDCAD",
    "frxUSDCHF",
    "frxNZDUSD",
    "frxEURGBP",
    "frxEURJPY",
    "frxEURCHF",
    "frxEURCAD",
    "frxEURNZD",
    "frxEURAUD",
    "frxGBPJPY",
    "frxGBPCHF",
    "frxGBPCAD",
    "frxGBPAUD",
    "frxGBPNZD",
    "frxAUDJPY",
    "frxAUDCAD",
    "frxAUDNZD",
    "frxAUDCHF",
    "frxCADJPY",
    "frxCADCHF",
    "frxCHFJPY",
    "frxNZDJPY",
    "frxNZDCAD",
]

_SCAN_SYMBOLS_PRIMARY = [
    "R_10",
    "R_25",
    "R_50",
    "R_75",
    "R_100",
    "1HZ10V",
    "1HZ25V",
    "1HZ50V",
    "1HZ75V",
    "1HZ100V",
]

_SCAN_SYMBOLS_SECONDARY = [
    "BOOM500",
    "BOOM1000",
    "CRASH500",
    "CRASH1000",
    "BOOM300N",
    "CRASH300N",
    "JD10",
    "JD25",
    "JD50",
    "JD75",
    "JD100",
    "stpRNG",
]

_SCAN_TICKS = 200
_SCAN_TIMEOUT = 15
_ADX_TREND_MIN = 15.0


def _fetch_prices_for_symbol(
    symbol: str,
    count: int,
) -> list:
    """
    Busca `count` ticks históricos de `symbol`.

    Retorna lista de preços em ordem cronológica.
    """
    received: list = []
    done = threading.Event()

    def _send_history_request(ws) -> None:
        ws.send(
            json.dumps(
                {
                    "ticks_history": symbol,
                    "end": "latest",
                    "count": count,
                    "style": "ticks",
                    "adjust_start_time": 1,
                }
            )
        )

    def _on_open(ws) -> None:
        if _uses_bearer_auth():
            _send_history_request(ws)
        else:
            ws.send(
                json.dumps(
                    {
                        "authorize": TOKEN
                    }
                )
            )

    def _on_message(ws, message) -> None:
        try:
            data = json.loads(message)
        except (TypeError, json.JSONDecodeError):
            ws.close()
            return

        if "error" in data:
            ws.close()
            return

        if data.get("msg_type") == "authorize":
            _send_history_request(ws)
            return

        if data.get("msg_type") == "history":
            try:
                prices = (
                    data.get("history", {})
                    .get("prices", [])
                )

                received.extend(
                    float(p)
                    for p in prices
                )
            except (TypeError, ValueError):
                pass

            ws.close()

    def _on_error(ws, err) -> None:
        ws.close()

    def _on_close(ws, *_args) -> None:
        done.set()

    ws = websocket.WebSocketApp(
        _get_ws_url(),
        on_open=_on_open,
        on_message=_on_message,
        on_error=_on_error,
        on_close=_on_close,
    )

    threading.Thread(
        target=ws.run_forever,
        daemon=True,
    ).start()

    done.wait(timeout=_SCAN_TIMEOUT)

    return received


def _score_trend(prices: list) -> float:
    """
    Pontuação de tendência combinando:
      - ADX
      - alinhamento EMA
      - MACD histogram

    Quanto maior o valor, mais forte a tendência.
    """
    import indicators as ind

    from config import (
        EMA_FAST,
        EMA_SLOW,
        ADX_PERIOD,
        MACD_FAST,
        MACD_SLOW,
        MACD_SIGNAL,
    )

    if not prices:
        return 0.0

    adx_val = ind.adx(
        prices,
        ADX_PERIOD,
    )

    ema9 = ind.ema(
        prices,
        EMA_FAST,
    )

    ema21 = ind.ema(
        prices,
        EMA_SLOW,
    )

    macd_res = ind.macd(
        prices,
        MACD_FAST,
        MACD_SLOW,
        MACD_SIGNAL,
    )

    if any(
        v is None
        for v in (
            adx_val,
            ema9,
            ema21,
            macd_res,
        )
    ):
        return 0.0

    try:
        _, _, macd_hist = macd_res
    except (TypeError, ValueError):
        return 0.0

    ema_factor = (
        1.2
        if ema9 != ema21
        else 1.0
    )

    last_price = (
        prices[-1]
        if prices
        else 1.0
    )

    macd_factor = (
        1.0
        + min(
            abs(macd_hist)
            / (
                last_price * 0.001
                + 1e-10
            ),
            0.5,
        )
    )

    return (
        adx_val
        * ema_factor
        * macd_factor
    )


def _scan_group(candidates: list) -> dict:
    """Escaneia um grupo de símbolos em paralelo."""
    results: dict = {}
    lock = threading.Lock()

    def _scan_one(symbol: str) -> None:
        prices = _fetch_prices_for_symbol(
            symbol,
            _SCAN_TICKS,
        )

        score = (
            _score_trend(prices)
            if len(prices) >= 50
            else 0.0
        )

        direction = ""

        if len(prices) >= 21:
            import indicators as ind

            e9 = ind.ema(
                prices,
                9,
            )

            e21 = ind.ema(
                prices,
                21,
            )

            if (
                e9 is not None
                and e21 is not None
            ):
                direction = (
                    "↑"
                    if e9 > e21
                    else "↓"
                )

        with lock:
            results[symbol] = score

        print(
            f"  {symbol:<12} "
            f"score={score:6.2f} "
            f"{direction}"
        )

    threads = [
        threading.Thread(
            target=_scan_one,
            args=(symbol,),
            daemon=True,
        )
        for symbol in candidates
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join(
            timeout=_SCAN_TIMEOUT + 5
        )

    return results


def _best_from(results: dict) -> tuple:
    """Retorna (melhor_símbolo, score)."""
    if not results:
        return "", 0.0

    best = max(
        results,
        key=results.__getitem__,
    )

    return (
        best,
        results[best],
    )


def _detect_trending_symbol(
    no_scan: bool = False,
) -> str:
    """
    Etapas:

      1. Escaneia pares Forex.
      2. Se o melhor Forex tiver score suficiente, retorna-o.
      3. Caso contrário, escaneia índices de volatilidade.
      4. Caso contrário, escaneia Boom/Crash/Jump.
      5. Se nenhum tiver tendência suficiente,
         mantém o SYMBOL configurado.
    """
    import config as _cfg

    if no_scan:
        return _cfg.SYMBOL

    # ── Etapa 1: Scan Forex ───────────────────────────────────

    print(
        "\n[TENDÊNCIA] "
        "── Scan forex (pares de moedas) ────────────────"
    )

    print(
        "[TENDÊNCIA] Candidatos: "
        + ", ".join(_SCAN_SYMBOLS_FOREX)
    )

    forex = _scan_group(
        _SCAN_SYMBOLS_FOREX
    )

    best_fx, score_fx = _best_from(
        forex
    )

    if score_fx >= _ADX_TREND_MIN:
        print(
            "\n[TENDÊNCIA] ✔ Melhor par forex: "
            f"{_symbol_display(best_fx)} "
            f"(score={score_fx:.2f})"
        )

        return best_fx

    print(
        "\n[TENDÊNCIA] ⚠ Nenhum par forex "
        "com tendência clara "
        f"(melhor score={score_fx:.2f} "
        f"< {_ADX_TREND_MIN}) "
        "— mercado fechado ou lateral. "
        "Tentando índices sintéticos..."
    )

    # ── Etapa 2: Scan primários ───────────────────────────────

    print(
        "\n[TENDÊNCIA] "
        "── Scan primário (volatilidade) ────────────────"
    )

    print(
        "[TENDÊNCIA] Candidatos: "
        + ", ".join(_SCAN_SYMBOLS_PRIMARY)
    )

    primary = _scan_group(
        _SCAN_SYMBOLS_PRIMARY
    )

    best, score = _best_from(
        primary
    )

    if score >= _ADX_TREND_MIN:
        print(
            "\n[TENDÊNCIA] ✔ Melhor índice: "
            f"{_symbol_display(best)} "
            f"(score={score:.2f})"
        )

        return best

    print(
        "\n[TENDÊNCIA] ⚠ Todos os índices "
        "primários estão laterais "
        f"(melhor score={score:.2f} "
        f"< {_ADX_TREND_MIN})."
    )

    # ── Etapa 3: Scan secundários ─────────────────────────────

    print(
        "[TENDÊNCIA] "
        "── Scan secundário (boom/crash/jump) ───────────"
    )

    print(
        "[TENDÊNCIA] Candidatos: "
        + ", ".join(_SCAN_SYMBOLS_SECONDARY)
    )

    secondary = _scan_group(
        _SCAN_SYMBOLS_SECONDARY
    )

    best2, score2 = _best_from(
        secondary
    )

    if score2 >= _ADX_TREND_MIN:
        print(
            "\n[TENDÊNCIA] ✔ Melhor índice "
            "(secundário): "
            f"{_symbol_display(best2)} "
            f"(score={score2:.2f})"
        )

        return best2

    # ── Etapa 4: fallback ─────────────────────────────────────

    print(
        "\n[TENDÊNCIA] ⚠ Nenhum instrumento "
        "com tendência clara "
        f"(melhor score secundário={score2:.2f}). "
        f"Usando padrão: {_cfg.SYMBOL}."
    )

    return _cfg.SYMBOL


# ─────────────────────────────────────────────────────────────────
# FASE 0 — Busca de ticks históricos
# ─────────────────────────────────────────────────────────────────

def _fetch_historical_ticks(
    count: int,
) -> int:
    """
    Baixa candles históricos da API Deriv e grava em ticks.csv.

    Epochs já existentes são ignorados.
    """
    tf_min = CANDLE_TIMEFRAME_SEC // 60

    print(
        f"\n[HISTÓRICO] Buscando os últimos "
        f"{count:,} candles de {tf_min} min "
        f"de '{_symbol_display(SYMBOL)}'..."
    )

    _ensure_ticks_header()

    existing_epochs: set = set()

    if (
        os.path.exists(TICKS_CSV)
        and os.path.getsize(TICKS_CSV) > 0
    ):
        try:
            with open(
                TICKS_CSV,
                "r",
                encoding="utf-8",
            ) as f:
                for row in csv.DictReader(f):
                    try:
                        existing_epochs.add(
                            (
                                int(row["epoch"]),
                                row.get(
                                    "symbol",
                                    "",
                                ),
                            )
                        )
                    except (
                        KeyError,
                        ValueError,
                    ):
                        pass
        except Exception:
            pass

    received: list = []
    done = threading.Event()
    error_msg: list = []

    def _send_history_request(ws) -> None:
        ws.send(
            json.dumps(
                {
                    "ticks_history": SYMBOL,
                    "end": "latest",
                    "count": count,
                    "style": "candles",
                    "granularity": CANDLE_TIMEFRAME_SEC,
                    "adjust_start_time": 1,
                }
            )
        )

    def _on_open(ws) -> None:
        if _uses_bearer_auth():
            _send_history_request(ws)
        else:
            ws.send(
                json.dumps(
                    {
                        "authorize": TOKEN
                    }
                )
            )

    def _on_message(ws, message) -> None:
        try:
            data = json.loads(message)
        except (
            TypeError,
            json.JSONDecodeError,
        ):
            return

        if "error" in data:
            error_msg.append(
                data["error"].get(
                    "message",
                    "Erro desconhecido",
                )
            )
            ws.close()
            return

        if data.get("msg_type") == "authorize":
            _send_history_request(ws)
            return

        if data.get("msg_type") == "candles":
            for candle in data.get(
                "candles",
                [],
            ):
                try:
                    epoch = int(
                        candle.get(
                            "epoch",
                            0,
                        )
                    )

                    close = float(
                        candle.get(
                            "close",
                            candle.get(
                                "open",
                                0,
                            ),
                        )
                    )

                    if epoch > 0:
                        received.append(
                            (
                                epoch,
                                close,
                            )
                        )

                except (
                    TypeError,
                    ValueError,
                ):
                    continue

            ws.close()

    def _on_error(ws, err) -> None:
        error_msg.append(str(err))
        ws.close()

    def _on_close(
        ws,
        code,
        msg,
    ) -> None:
        done.set()

    ws = websocket.WebSocketApp(
        _get_ws_url(),
        on_open=_on_open,
        on_message=_on_message,
        on_error=_on_error,
        on_close=_on_close,
    )

    thread = threading.Thread(
        target=ws.run_forever,
        daemon=True,
    )

    thread.start()

    done.wait(timeout=30)

    if error_msg:
        print(
            f"[HISTÓRICO] Erro da API: "
            f"{error_msg[0]}"
        )
        return 0

    if not received:
        print(
            "[HISTÓRICO] Nenhum candle "
            "histórico recebido."
        )
        return 0

    received.sort(
        key=lambda item: item[0]
    )

    written = 0

    with open(
        TICKS_CSV,
        "a",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)

        for epoch, close in received:
            key = (
                epoch,
                SYMBOL,
            )

            if key in existing_epochs:
                continue

            dt_str = datetime.fromtimestamp(
                epoch
            ).isoformat()

            writer.writerow(
                [
                    epoch,
                    dt_str,
                    SYMBOL,
                    close,
                ]
            )

            existing_epochs.add(key)
            written += 1

    print(
        f"[HISTÓRICO] {written:,} candles "
        f"de {tf_min}min gravados — "
        f"total no CSV: {_count_ticks():,}"
    )

    return written


def _banner(
    history_count: int,
    min_ticks: int,
    demo: bool,
) -> None:
    mode = (
        "DEMO"
        if demo
        else "REAL"
    )

    tf_min = CANDLE_TIMEFRAME_SEC // 60

    print("=" * 60)
    print(
        "  Deriv Pipeline — "
        "Orquestrador Automático"
    )
    print(
        f"  Modo              : {mode}"
    )
    print(
        f"  Símbolo           : "
        f"{_symbol_display(SYMBOL)}"
    )
    print(
        f"  Candles históricos : "
        f"{history_count:,} × {tf_min} min"
    )
    print(
        f"  Candles p/ treino  : "
        f"{min_ticks:,}"
    )
    print(
        "  Re-treino          : "
        "adaptativo (IA decide automaticamente)"
    )
    print(
        f"    › WinRate < "
        f"{RETRAIN_WIN_RATE_TRIGGER:.0%} | "
        f"Conf < "
        f"{RETRAIN_CONFIDENCE_TRIGGER:.0%} | "
        f"+{RETRAIN_NEW_TICKS_TRIGGER} ticks | "
        f"backstop "
        f"{RETRAIN_MAX_INTERVAL_SEC // 60}min"
    )
    print(
        "  Pressione Ctrl+C "
        "para encerrar tudo"
    )
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────
# Pré-checagem — autenticação Deriv
# ─────────────────────────────────────────────────────────────────

def _validate_deriv_token(
    demo: bool,
    timeout_sec: int = 15,
) -> bool:
    """
    Valida DERIV_TOKEN antes de iniciar scan/coletor/treino/bot.
    """
    if not str(TOKEN).strip():
        print(
            "\n[AUTH] DERIV_TOKEN vazio. "
            "Atualize o .env e tente novamente."
        )
        return False

    if deriv_session.is_bearer_token(
        TOKEN,
        DERIV_AUTH_MODE,
    ):
        return _validate_deriv_bearer_token(
            demo=demo,
            timeout_sec=timeout_sec,
        )

    done = threading.Event()

    result = {
        "ok": False,
        "message": "",
        "code": "",
        "loginid": "",
        "currency": "",
    }

    def _on_open(ws) -> None:
        ws.send(
            json.dumps(
                {
                    "authorize": TOKEN
                }
            )
        )

    def _on_message(
        ws,
        message,
    ) -> None:
        try:
            data = json.loads(message)
        except (
            TypeError,
            json.JSONDecodeError,
        ):
            return

        if "error" in data:
            error = data.get(
                "error",
                {},
            )

            result["message"] = error.get(
                "message",
                "Erro de autenticação desconhecido.",
            )

            result["code"] = error.get(
                "code",
                "",
            )

            ws.close()
            return

        if data.get(
            "msg_type"
        ) == "authorize":
            auth = data.get(
                "authorize",
                {},
            )

            result["ok"] = True
            result["loginid"] = str(
                auth.get(
                    "loginid",
                    "",
                )
            )
            result["currency"] = str(
                auth.get(
                    "currency",
                    "",
                )
            )

            ws.close()

    def _on_error(ws, err) -> None:
        result["message"] = str(err)
        ws.close()

    def _on_close(
        ws,
        *_args,
    ) -> None:
        done.set()

    print(
        "\n[AUTH] Validando token Deriv..."
    )

    ws = websocket.WebSocketApp(
        _WS_URL,
        on_open=_on_open,
        on_message=_on_message,
        on_error=_on_error,
        on_close=_on_close,
    )

    thread = threading.Thread(
        target=ws.run_forever,
        daemon=True,
    )

    thread.start()

    done.wait(
        timeout=timeout_sec
    )

    if not done.is_set():
        try:
            ws.close()
        except Exception:
            pass

        print(
            f"[AUTH] Timeout após "
            f"{timeout_sec}s ao validar o token."
        )

        return False

    if result["ok"]:
        account = (
            result["loginid"]
            or "desconhecida"
        )

        currency = (
            result["currency"]
            or "sem moeda"
        )

        print(
            f"[AUTH] Token validado | "
            f"Conta: {account} | "
            f"Moeda: {currency}"
        )

        return True

    message = (
        result["message"]
        or "A Deriv recusou a autenticação."
    )

    code = (
        f" ({result['code']})"
        if result["code"]
        else ""
    )

    print(
        f"[AUTH] Falha na autenticação: "
        f"{message}{code}"
    )

    if "disabled" in message.lower():
        print(
            "[AUTH] A conta vinculada "
            "ao token está desativada."
        )

    return False


def _validate_deriv_bearer_token(
    demo: bool,
    timeout_sec: int = 15,
) -> bool:
    """
    Valida tokens Bearer/PAT através do fluxo novo da Deriv.
    """
    global _AUTH_MODE
    global _PAT_ACCOUNT_ID
    global _PAT_ACCOUNT
    global _PAT_WS_URL

    token_kind = (
        "PAT"
        if deriv_session.is_pat_token(TOKEN)
        else "Bearer/OAuth"
    )

    print(
        f"[AUTH] Token {token_kind} detectado "
        "(fluxo novo da Deriv)."
    )

    try:
        session = (
            deriv_session.setup_bearer_options_session(
                app_id=APP_ID,
                token=TOKEN,
                demo=demo,
                timeout_sec=timeout_sec,
            )
        )

    except deriv_session.DerivAPIError as exc:
        code = (
            f" ({exc.code})"
            if exc.code
            else ""
        )

        print(
            f"[AUTH] Falha no token Bearer "
            f"via REST: {exc}{code}"
        )

        if exc.status == 401:
            print(
                "[AUTH] O token está inválido, "
                "expirado, revogado ou não pertence "
                "a este App ID."
            )

        elif (
            "1015" in str(exc)
            or exc.status == 429
        ):
            print(
                "[AUTH] Cloudflare bloqueou "
                "a requisição."
            )

        _print_bearer_migration_hint()

        return False

    except Exception as exc:
        print(
            f"[AUTH] Erro inesperado ao validar "
            f"token Bearer: {exc}"
        )
        return False

    _AUTH_MODE = "bearer"

    _PAT_ACCOUNT_ID = session[
        "account_id"
    ]

    _PAT_ACCOUNT = session[
        "account"
    ]

    _PAT_WS_URL = session.get(
        "ws_url"
    )

    balance = _PAT_ACCOUNT.get(
        "balance",
        "?",
    )

    currency = _PAT_ACCOUNT.get(
        "currency",
        "USD",
    )

    account_type = _PAT_ACCOUNT.get(
        "account_type",
        "demo",
    )

    print(
        f"[AUTH] Token Bearer validado | "
        f"Conta Options: {_PAT_ACCOUNT_ID} "
        f"({account_type}) | "
        f"Saldo: {balance} {currency}"
    )

    return True


def _print_bearer_migration_hint() -> None:
    print(
        "[AUTH] ──────────────────────────────────────────"
    )

    print(
        "[AUTH] Verifique o App ID e o token "
        "Bearer/PAT configurados."
    )

    if str(APP_ID) == "1089":
        print(
            "[AUTH] ATENÇÃO: App ID legado 1089 "
            "pode não funcionar com o fluxo PAT/REST."
        )

    print(
        "[AUTH] ──────────────────────────────────────────"
    )


# ─────────────────────────────────────────────────────────────────
# FASE 1 — Coletor em thread daemon
# ─────────────────────────────────────────────────────────────────

class _CollectorThread(threading.Thread):
    """Thread de coleta de ticks com reconexão automática."""

    def __init__(self) -> None:
        super().__init__(
            daemon=True,
            name="Collector",
        )

        self._ws: Optional[
            websocket.WebSocketApp
        ] = None

        self._local_count = 0

        self._last_epoch: int = (
            self._load_last_epoch()
        )

        self._last_price: float = 0.0

    def _load_last_epoch(self) -> int:
        """Lê o maior epoch gravado em ticks.csv."""
        last = 0

        if (
            not os.path.exists(TICKS_CSV)
            or os.path.getsize(TICKS_CSV) == 0
        ):
            return last

        try:
            with open(
                TICKS_CSV,
                "r",
                encoding="utf-8",
            ) as f:
                for row in csv.DictReader(f):
                    try:
                        epoch = int(
                            row["epoch"]
                        )

                        if epoch > last:
                            last = epoch

                    except (
                        KeyError,
                        ValueError,
                    ):
                        pass

        except Exception:
            pass

        return last

    def run(self) -> None:
        _ensure_ticks_header()

        while not _shutdown.is_set():
            try:
                ws = websocket.WebSocketApp(
                    _get_ws_url(),
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )

                self._ws = ws

                ws.run_forever(
                    reconnect=5
                )

            except Exception as exc:
                if not _shutdown.is_set():
                    print(
                        f"\n[COLETOR] Exceção inesperada: "
                        f"{exc} — reconectando em 5s"
                    )

                    time.sleep(5)

    def stop(self) -> None:
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass

    def _on_open(self, ws) -> None:
        print(
            f"\n[COLETOR] Conectado | "
            f"{_symbol_display(SYMBOL)}"
        )

        if not _uses_bearer_auth():
            ws.send(
                json.dumps(
                    {
                        "authorize": TOKEN
                    }
                )
            )

        ws.send(
            json.dumps(
                {
                    "ticks": SYMBOL,
                    "subscribe": 1,
                }
            )
        )

    def _on_message(
        self,
        ws,
        message: str,
    ) -> None:
        try:
            data = json.loads(message)
        except (
            TypeError,
            json.JSONDecodeError,
        ):
            return

        if "error" in data:
            print(
                "\n[COLETOR] Erro API: "
                f"{data['error'].get('message', 'Erro desconhecido')}"
            )
            return

        if data.get(
            "msg_type"
        ) == "authorize":
            print(
                "[COLETOR] Autorizado. "
                "Coleta iniciada..."
            )
            return

        if "tick" not in data:
            return

        tick = data["tick"]

        try:
            epoch = int(
                tick["epoch"]
            )

            price = float(
                tick["quote"]
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            return

        if epoch <= self._last_epoch:
            return

        self._last_epoch = epoch

        # Filtro anti-spike.
        if self._last_price != 0.0:
            variation = abs(
                (
                    price
                    - self._last_price
                )
                / self._last_price
            )

            if (
                variation
                >= TICK_SPIKE_THRESHOLD
            ):
                print(
                    f"\n[COLETOR] Spike rejeitado: "
                    f"{self._last_price:.4f} → "
                    f"{price:.4f} "
                    f"({variation * 100:.2f}%)"
                )

                return

        self._last_price = price

        dt_str = datetime.fromtimestamp(
            epoch
        ).isoformat()

        self._local_count += 1

        with open(
            TICKS_CSV,
            "a",
            newline="",
            encoding="utf-8",
        ) as f:
            csv.writer(f).writerow(
                [
                    epoch,
                    dt_str,
                    SYMBOL,
                    price,
                ]
            )

        print(
            f"\r[COLETOR] Ticks: "
            f"{self._local_count:>7,} | "
            f"{price} @ {dt_str}",
            end="",
            flush=True,
        )

    def _on_error(
        self,
        ws,
        error,
    ) -> None:
        if not _shutdown.is_set():
            print(
                f"\n[COLETOR] Erro WebSocket: "
                f"{error}"
            )

    def _on_close(
        self,
        ws,
        code,
        msg,
    ) -> None:
        if not _shutdown.is_set():
            print(
                f"\n[COLETOR] Conexão encerrada "
                f"(código: {code}) — reconectando..."
            )


# ─────────────────────────────────────────────────────────────────
# FASE 2 — Aguardar ticks mínimos
# ─────────────────────────────────────────────────────────────────

def _wait_for_ticks(
    min_ticks: int,
    poll_sec: int = 10,
) -> None:
    """Bloqueia até ticks.csv conter min_ticks linhas."""
    current = _count_ticks()

    if current >= min_ticks:
        print(
            f"[PIPELINE] {current:,} candles "
            "já disponíveis — pronto para treino."
        )
        return

    print(
        f"\n[PIPELINE] Aguardando "
        f"{min_ticks:,} candles para o primeiro treino "
        f"(atual: {current:,})..."
    )

    while not _shutdown.is_set():
        time.sleep(poll_sec)

        current = _count_ticks()

        print(
            f"\r[PIPELINE] Candles: "
            f"{current:,} / {min_ticks:,}   ",
            end="",
            flush=True,
        )

        if current >= min_ticks:
            print()
            return

    sys.exit(0)


# ─────────────────────────────────────────────────────────────────
# FASES 3 e 4 — Dataset + treino
# ─────────────────────────────────────────────────────────────────

def _trim_ticks(
    ticks_path: str,
    max_ticks: int = 50000,
) -> None:
    """Limita o arquivo aos últimos max_ticks."""
    import pandas as pd

    if (
        not os.path.exists(ticks_path)
        or os.path.getsize(ticks_path) == 0
    ):
        return

    try:
        df = pd.read_csv(
            ticks_path
        )

        if len(df) > max_ticks:
            trimmed = (
                len(df)
                - max_ticks
            )

            df = df.tail(
                max_ticks
            )

            df.to_csv(
                ticks_path,
                index=False,
            )

            print(
                f"[PIPELINE] Ticks trimados: "
                f"{trimmed:,} removidos, "
                f"{max_ticks:,} mantidos."
            )

    except Exception as exc:
        print(
            "[PIPELINE] Aviso: não foi possível "
            f"trimar ticks: {exc}"
        )


def _load_latest_final_metrics() -> Optional[dict]:
    """Lê a última métrica final registrada."""
    metrics_path = os.path.join(
        os.path.dirname(
            os.path.abspath(
                AI_MODEL_PATH
            )
        ),
        "model_metrics.json",
    )

    if not os.path.exists(
        metrics_path
    ):
        return None

    try:
        with open(
            metrics_path,
            "r",
            encoding="utf-8",
        ) as f:
            hist = json.load(f)

        if not isinstance(
            hist,
            list,
        ):
            return None

        for item in reversed(hist):
            if (
                isinstance(
                    item,
                    dict,
                )
                and item.get(
                    "stage"
                )
                == "final"
            ):
                return item

    except Exception:
        return None

    return None


def _should_promote_model(
    current: Optional[dict],
    challenger: Optional[dict],
) -> tuple[bool, str]:
    """Decide se o challenger pode substituir o champion."""
    if challenger is None:
        return (
            False,
            "métricas do challenger indisponíveis",
        )

    if current is None:
        return (
            True,
            "sem champion anterior com métricas finais",
        )

    cur_auc = float(
        current.get(
            "auc",
            0.0,
        )
    )

    cur_acc = float(
        current.get(
            "accuracy",
            0.0,
        )
    )

    cur_f1 = float(
        current.get(
            "f1",
            0.0,
        )
    )

    ch_auc = float(
        challenger.get(
            "auc",
            0.0,
        )
    )

    ch_acc = float(
        challenger.get(
            "accuracy",
            0.0,
        )
    )

    ch_f1 = float(
        challenger.get(
            "f1",
            0.0,
        )
    )

    auc_ok = (
        ch_auc
        >= cur_auc
        + MODEL_PROMOTION_MIN_AUC_DELTA
    )

    acc_ok = (
        ch_acc
        >= cur_acc
        - MODEL_PROMOTION_MAX_ACC_DROP
    )

    f1_ok = (
        ch_f1
        >= cur_f1
        - MODEL_PROMOTION_MAX_F1_DROP
    )

    if (
        auc_ok
        and acc_ok
        and f1_ok
    ):
        return (
            True,
            (
                f"AUC {ch_auc:.4f} >= "
                f"{cur_auc:.4f}+"
                f"{MODEL_PROMOTION_MIN_AUC_DELTA:.4f}, "
                f"ACC {ch_acc:.4f} sem queda excessiva, "
                f"F1 {ch_f1:.4f} sem queda excessiva"
            ),
        )

    return (
        False,
        (
            "gate reprovado | "
            f"champion(AUC={cur_auc:.4f}, "
            f"ACC={cur_acc:.4f}, "
            f"F1={cur_f1:.4f}) "
            "vs "
            f"challenger(AUC={ch_auc:.4f}, "
            f"ACC={ch_acc:.4f}, "
            f"F1={ch_f1:.4f})"
        ),
    )


def _run_training(
    dataset_path: str = DATASET_CSV,
    model_path: str = AI_MODEL_PATH,
    test_ratio: float = 0.2,
    tft_output_path: str | None = None,
    duration_output_path: str | None = None,
) -> tuple[bool, Optional[dict]]:
    """
    Constrói dataset e treina o modelo.

    O ticks.csv original nunca é trimado diretamente durante o treino:
    primeiro é criado um snapshot temporário.
    """
    import shutil

    print(
        "\n[PIPELINE] "
        "── Fase 3: Construindo dataset ──"
    )

    tmp_ticks = (
        f"{TICKS_CSV}.train_snapshot"
    )

    try:
        shutil.copy2(
            TICKS_CSV,
            tmp_ticks,
        )

        # Filtra o snapshot para o símbolo ativo.
        try:
            import pandas as _pd
            import config as _cfg

            _active_sym = (
                _cfg.get_active_symbol()
            )

            _snap_df = _pd.read_csv(
                tmp_ticks
            )

            if (
                "symbol"
                in _snap_df.columns
                and _active_sym
            ):
                before = len(
                    _snap_df
                )

                _snap_df = _snap_df[
                    _snap_df["symbol"]
                    == _active_sym
                ]

                if len(_snap_df) < before:
                    print(
                        f"[PIPELINE] Snapshot filtrado "
                        f"para '{_active_sym}': "
                        f"{len(_snap_df):,} ticks "
                        f"({before - len(_snap_df):,} "
                        "de outros símbolos removidos)"
                    )

                _snap_df.to_csv(
                    tmp_ticks,
                    index=False,
                )

        except Exception:
            pass

        _trim_ticks(
            tmp_ticks,
            max_ticks=200000,
        )

        n = dataset_builder.build_dataset(
            tmp_ticks,
            dataset_path,
            window_size=50,
            lookforward=TARGET_LOOKFORWARD,
        )

        if n < 30:
            print(
                f"[PIPELINE] Dataset muito pequeno "
                f"({n} linhas). Colete mais ticks."
            )

            return False, None

    except SystemExit:
        return False, None

    except Exception as exc:
        print(
            f"[PIPELINE] Erro ao construir "
            f"dataset: {exc}"
        )

        return False, None

    finally:
        if os.path.exists(
            tmp_ticks
        ):
            try:
                os.remove(
                    tmp_ticks
                )
            except OSError:
                pass

    print(
        "\n[PIPELINE] "
        "── Fase 4: Treinando modelo ──"
    )

    metrics: Optional[dict] = None

    try:
        metrics = train_model.train(
            dataset_path,
            model_path,
            test_ratio,
            tft_output_path=tft_output_path,
            duration_output_path=duration_output_path,
        )

    except SystemExit:
        return False, None

    except Exception as exc:
        print(
            f"[PIPELINE] Erro ao treinar "
            f"modelo: {exc}"
        )

        return False, None

    return True, metrics


# ─────────────────────────────────────────────────────────────────
# FASE 6 — Re-treino adaptativo
# ─────────────────────────────────────────────────────────────────

def _reset_ai_predictor() -> None:
    """
    Reseta os modelos singleton para que os novos modelos
    sejam carregados na próxima inferência.
    """
    try:
        import ai_predictor

        ai_predictor._model = None
        ai_predictor._model_loaded = False
        ai_predictor._model_features = []

        ai_predictor._dur_model = None
        ai_predictor._dur_model_loaded = False
        ai_predictor._dur_model_features = []

        ai_predictor._tft_model = None
        ai_predictor._tft_model_loaded = False

        print(
            "[RE-TREINO] Singleton da IA resetado — "
            "novo modelo ativo na próxima predição."
        )

    except Exception:
        pass


_retrain_state: dict = {
    "last_epoch": 0.0,
    "last_tick_count": 0,
}


def _should_retrain_now() -> tuple[bool, str]:
    """
    Avalia os gatilhos de re-treino.

    Gatilhos:
      1. Backstop temporal
      2. Win rate recente
      3. Confiança média da IA
      4. Novos ticks acumulados
    """
    elapsed = (
        time.time()
        - _retrain_state["last_epoch"]
    )

    if (
        elapsed
        < RETRAIN_MIN_INTERVAL_SEC
    ):
        return (
            False,
            (
                "intervalo mínimo não atingido "
                f"({elapsed:.0f}s < "
                f"{RETRAIN_MIN_INTERVAL_SEC}s)"
            ),
        )

    if (
        elapsed
        >= RETRAIN_MAX_INTERVAL_SEC
    ):
        return (
            True,
            (
                "backstop temporal "
                f"({elapsed / 60:.0f} min "
                "sem retreino)"
            ),
        )

    ops_path = os.path.join(
        os.path.dirname(
            os.path.abspath(
                TICKS_CSV
            )
        ),
        OPERATIONS_LOG,
    )

    try:
        import pandas as _pd

        if (
            os.path.exists(ops_path)
            and os.path.getsize(ops_path) > 0
        ):
            df = _pd.read_csv(
                ops_path
            )

            if (
                len(df)
                >= RETRAIN_EVAL_WINDOW
            ):
                recent = df.tail(
                    RETRAIN_EVAL_WINDOW
                )

                if (
                    "result"
                    in recent.columns
                ):
                    wr = (
                        float(
                            (
                                recent["result"]
                                == "WIN"
                            ).sum()
                        )
                        / len(recent)
                    )

                    if (
                        wr
                        < RETRAIN_WIN_RATE_TRIGGER
                    ):
                        return (
                            True,
                            (
                                "win rate recente "
                                f"{wr:.1%} < limiar "
                                f"{RETRAIN_WIN_RATE_TRIGGER:.1%}"
                            ),
                        )

                if (
                    "ai_confidence"
                    in recent.columns
                ):
                    avg_conf = float(
                        recent[
                            "ai_confidence"
                        ].mean()
                    )

                    if (
                        avg_conf
                        < RETRAIN_CONFIDENCE_TRIGGER
                    ):
                        return (
                            True,
                            (
                                "confiança média "
                                f"{avg_conf:.2f} < limiar "
                                f"{RETRAIN_CONFIDENCE_TRIGGER:.2f}"
                            ),
                        )

    except Exception:
        pass

    ticks_now = _count_ticks()

    new_ticks = (
        ticks_now
        - _retrain_state[
            "last_tick_count"
        ]
    )

    if (
        new_ticks
        >= RETRAIN_NEW_TICKS_TRIGGER
    ):
        return (
            True,
            (
                f"{new_ticks:,} novos ticks "
                "acumulados "
                f"(limiar: "
                f"{RETRAIN_NEW_TICKS_TRIGGER:,})"
            ),
        )

    return (
        False,
        "nenhum gatilho acionado",
    )


def _retrain_loop() -> None:
    """
    Thread de re-treino adaptativo.

    O bot continua funcionando enquanto o treino acontece.
    """
    _retrain_state[
        "last_epoch"
    ] = time.time()

    _retrain_state[
        "last_tick_count"
    ] = _count_ticks()

    while not _shutdown.is_set():

        for _ in range(12):
            if _shutdown.is_set():
                return

            time.sleep(5)

        should, reason = (
            _should_retrain_now()
        )

        if not should:
            continue

        n_ticks = _count_ticks()

        print(
            f"\n[RE-TREINO] Gatilho: "
            f"{reason} "
            f"({n_ticks:,} ticks disponíveis). "
            "Iniciando re-treino..."
        )

        tmp_dataset = (
            f"{DATASET_CSV}.tmp"
        )

        tmp_model = (
            f"{AI_MODEL_PATH}.new"
        )

        tmp_tft = (
            f"{TRANSFORMER_MODEL_PATH}.new"
        )

        tmp_duration = (
            f"{DURATION_MODEL_PATH}.new"
        )

        champion_metrics = (
            _load_latest_final_metrics()
        )

        success, challenger_metrics = (
            _run_training(
                dataset_path=tmp_dataset,
                model_path=tmp_model,
                tft_output_path=tmp_tft,
                duration_output_path=tmp_duration,
            )
        )

        if (
            success
            and os.path.exists(tmp_model)
        ):
            promote, promo_reason = (
                _should_promote_model(
                    champion_metrics,
                    challenger_metrics,
                )
            )

            if promote:
                os.replace(
                    tmp_model,
                    AI_MODEL_PATH,
                )

                if os.path.exists(
                    tmp_tft
                ):
                    os.replace(
                        tmp_tft,
                        TRANSFORMER_MODEL_PATH,
                    )

                if os.path.exists(
                    tmp_duration
                ):
                    os.replace(
                        tmp_duration,
                        DURATION_MODEL_PATH,
                    )

                _reset_ai_predictor()

                print(
                    "[RE-TREINO] model.pkl "
                    "atualizado com sucesso "
                    f"({promo_reason})."
                )

            else:
                print(
                    "[RE-TREINO] Challenger "
                    f"não promovido: {promo_reason}"
                )

        else:
            print(
                "[RE-TREINO] Falhou — bot continua "
                "com o modelo anterior."
            )

        _retrain_state[
            "last_epoch"
        ] = time.time()

        _retrain_state[
            "last_tick_count"
        ] = _count_ticks()

        # Limpeza dos temporários.
        for tmp in (
            tmp_dataset,
            tmp_model,
            tmp_tft,
            tmp_duration,
        ):
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass


# ─────────────────────────────────────────────────────────────────
# Confirmação modo real
# ─────────────────────────────────────────────────────────────────

def _confirm_real_mode() -> bool:
    print(
        "\n" + "!" * 55
    )

    print(
        "  ATENÇÃO: Modo REAL ativado."
    )

    print(
        "  Este pipeline usará dinheiro real "
        "na conta configurada."
    )

    print(
        "!" * 55
    )

    answer = input(
        "\nDigite 'sim' para confirmar: "
    ).strip().lower()

    return answer == "sim"


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Pipeline automático: "
            "coleta → dataset → treino → "
            "bot → re-treino."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    mode_group = (
        parser.add_mutually_exclusive_group()
    )

    mode_group.add_argument(
        "--real",
        action="store_true",
        help=(
            "Força modo real. "
            "Requer confirmação."
        ),
    )

    mode_group.add_argument(
        "--demo",
        action="store_true",
        help=(
            "Força modo demo "
            "(ignora config.DEMO_MODE)."
        ),
    )

    parser.add_argument(
        "--balance",
        type=float,
        default=1000.0,
        metavar="USD",
        help=(
            "Saldo inicial para o RiskManager "
            "(padrão: 1000.0)."
        ),
    )

    parser.add_argument(
        "--min-ticks",
        type=int,
        default=_DEFAULT_MIN_TICKS,
        metavar="N",
        help=(
            "Ticks mínimos antes do primeiro treino."
        ),
    )

    parser.add_argument(
        "--skip-collect",
        action="store_true",
        help=(
            "Não inicia o coletor."
        ),
    )

    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help=(
            "Reconstrói dataset e retreina "
            "mesmo com model.pkl existente."
        ),
    )

    parser.add_argument(
        "--history-count",
        type=int,
        default=_DEFAULT_HISTORY_COUNT,
        metavar="N",
        help=(
            "Candles históricos a baixar."
        ),
    )

    parser.add_argument(
        "--no-scan",
        action="store_true",
        help=(
            "Pula o scan de tendência e "
            "usa SYMBOL de config.py."
        ),
    )

    args = parser.parse_args()

    # ── Determina modo ────────────────────────────────────────

    if args.real:
        is_demo = False

    elif args.demo:
        is_demo = True

    else:
        is_demo = DEMO_MODE

    if not is_demo:
        if not _confirm_real_mode():
            print(
                "Operação cancelada."
            )
            sys.exit(0)

    _banner(
        args.history_count,
        args.min_ticks,
        is_demo,
    )

    # ── Ctrl+C ────────────────────────────────────────────────

    def _handle_interrupt(
        sig,
        frame,
    ) -> None:
        print(
            "\n\n[PIPELINE] "
            "Encerrando tudo... aguarde."
        )

        _shutdown.set()

        sys.exit(0)

    signal.signal(
        signal.SIGINT,
        _handle_interrupt,
    )

    # ── Autenticação ──────────────────────────────────────────

    if not _validate_deriv_token(
        demo=is_demo
    ):
        sys.exit(1)

    # ── Scan de tendência ────────────────────────────────────

    import config as _cfg

    detected = (
        _detect_trending_symbol(
            no_scan=args.no_scan
        )
    )

    if detected != _cfg.SYMBOL:
        print(
            f"[TENDÊNCIA] Símbolo alterado: "
            f"{_cfg.SYMBOL} → {detected}"
        )

        import executor
        import strategy

        for module in (
            _cfg,
            executor,
            strategy,
            dataset_builder,
            train_model,
        ):
            if hasattr(
                module,
                "SYMBOL",
            ):
                setattr(
                    module,
                    "SYMBOL",
                    detected,
                )

        _cfg.set_active_symbol(
            detected
        )

        globals()["SYMBOL"] = detected

    # ── Guarda símbolo no state.json ─────────────────────────

    try:
        state_path = os.path.join(
            os.path.dirname(TICKS_CSV),
            "state.json",
        )

        state = {}

        if os.path.exists(
            state_path
        ):
            try:
                with open(
                    state_path,
                    "r",
                    encoding="utf-8",
                ) as f:
                    state = json.load(f)
            except Exception:
                state = {}

        state["symbol"] = detected

        with open(
            state_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                state,
                f,
            )

    except Exception:
        pass

    # ── FASE 0: Histórico ─────────────────────────────────────

    if not args.skip_collect:
        n_hist = (
            _fetch_historical_ticks(
                args.history_count
            )
        )

        if n_hist == 0:
            print(
                "[PIPELINE] Aviso: histórico "
                "não obtido — coletor ao vivo "
                "cobrirá os dados."
            )

    else:
        print(
            "[PIPELINE] Fase 0: "
            "--skip-collect ativo — "
            "histórico ignorado."
        )

    # ── FASE 1: Coletor ───────────────────────────────────────

    collector: Optional[
        _CollectorThread
    ] = None

    if not args.skip_collect:
        collector = _CollectorThread()
        collector.start()

        print(
            "[PIPELINE] Fase 1: Coletor "
            "iniciado em background "
            "(sem duplicatas)."
        )

    else:
        n_existing = _count_ticks()

        print(
            f"[PIPELINE] Fase 1: "
            f"--skip-collect ativo. "
            f"Usando '{TICKS_CSV}' existente "
            f"({n_existing:,} ticks)."
        )

    # ── FASES 2/3/4: Treino inicial ──────────────────────────

    model_exists = (
        os.path.exists(
            AI_MODEL_PATH
        )
        and not args.force_retrain
    )

    if model_exists:
        print(
            "\n[PIPELINE] Fases 2-4: "
            "model.pkl encontrado — "
            "treino inicial ignorado."
        )

        print(
            "           Use --force-retrain "
            "para forçar novo treino."
        )

    else:
        _wait_for_ticks(
            args.min_ticks
        )

        success, _metrics = (
            _run_training()
        )

        if not success:
            print(
                "\n[PIPELINE] Treino inicial "
                "falhou. Encerrando."
            )

            _shutdown.set()

            sys.exit(1)

    # ── FASE 5: Re-treino ─────────────────────────────────────

    retrain_thread = threading.Thread(
        target=_retrain_loop,
        daemon=True,
        name="Retrain",
    )

    retrain_thread.start()

    print(
        "\n[PIPELINE] Fase 5: "
        "Re-treino adaptativo iniciado "
        f"(backstop: "
        f"{RETRAIN_MAX_INTERVAL_SEC // 60} min)."
    )

    # ── FASE 6: Bot ───────────────────────────────────────────

    print(
        "\n[PIPELINE] Fase 6: "
        "Iniciando bot de trading...\n"
    )

    initial_balance = args.balance

    if _uses_bearer_auth():
        try:
            initial_balance = float(
                _PAT_ACCOUNT.get(
                    "balance",
                    initial_balance,
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            pass

    risk_manager = RiskManager(
        initial_balance=initial_balance
    )

    bot = DerivBot(
        risk_manager=risk_manager,
        demo=is_demo,
        auth_mode=_AUTH_MODE,
        account_id=str(
            _PAT_ACCOUNT_ID or ""
        ),
        initial_ws_url=(
            _get_ws_url()
            if _uses_bearer_auth()
            else ""
        ),
    )

    bot.run()


if __name__ == "__main__":
    main()