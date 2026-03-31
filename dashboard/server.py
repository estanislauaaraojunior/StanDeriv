#!/home/stanis/Repositorios/Binary/.venv/bin/python3
"""
dashboard/server.py — Backend Flask do Dashboard do Bot Deriv.

Uso:
    cd /home/stanis/Repositorios/Binary
    .venv/bin/python3 dashboard/server.py
    # ou: python3 dashboard/server.py (com venv ativo)

Acesso: http://localhost:5055
"""

import collections
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

# Garante que imports do bot funcionam mesmo rodando de dentro de dashboard/
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False

# ─── Paths ───────────────────────────────────────────────────────────────────

TICKS_CSV      = _ROOT / "ticks.csv"
OPERATIONS_CSV = _ROOT / "operacoes_log.csv"
DATASET_CSV    = _ROOT / "dataset.csv"
MODEL_PKL      = _ROOT / "model.pkl"
TFT_PKL        = _ROOT / "transformer_model.pkl"
DURATION_PKL   = _ROOT / "duration_model.pkl"
PIPELINE_PY    = _ROOT / "pipeline.py"
PID_FILE       = _ROOT / "dashboard" / "bot.pid"
STATIC_DIR     = Path(__file__).resolve().parent

# ─── Flask App ────────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")

# ─── Log buffer do pipeline ──────────────────────────────────────────────────

_bot_log_buffer: collections.deque = collections.deque(maxlen=200)
_bot_process: subprocess.Popen | None = None
_bot_log_lock = threading.Lock()


def _read_bot_output(proc: subprocess.Popen) -> None:
    """Lê stdout/stderr do pipeline em thread e preenche o buffer circular."""
    try:
        for line in iter(proc.stdout.readline, b""):
            decoded = line.decode("utf-8", errors="replace").rstrip()
            if decoded:
                with _bot_log_lock:
                    _bot_log_buffer.append(decoded)
        proc.stdout.close()
    except Exception:
        pass


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _read_csv_safe(path, **kwargs):
    """Lê CSV com tratamento seguro; retorna DataFrame vazio em caso de falha."""
    try:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return pd.DataFrame()
        return pd.read_csv(path, **kwargs)
    except Exception:
        return pd.DataFrame()


def _bot_pid() -> int | None:
    """Retorna PID salvo em bot.pid ou None."""
    try:
        if PID_FILE.exists():
            pid = int(PID_FILE.read_text().strip())
            return pid
    except (ValueError, OSError):
        pass
    return None


def _bot_running() -> tuple[bool, int | None]:
    """(is_running, pid) — verifica se o processo pipeline.py existe."""
    pid = _bot_pid()
    if pid is None:
        return False, None
    if _PSUTIL:
        try:
            proc = psutil.Process(pid)
            return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE, pid
        except psutil.NoSuchProcess:
            return False, None
    else:
        # Fallback sem psutil: verifica via kill -0
        try:
            os.kill(pid, 0)
            return True, pid
        except (ProcessLookupError, PermissionError):
            return False, None


# ─── Rotas estáticas ─────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(str(STATIC_DIR), "index.html")


# ─── API: Resumo do bot ───────────────────────────────────────────────────────

@app.route("/api/summary")
def api_summary():
    df = _read_csv_safe(OPERATIONS_CSV)

    if df.empty:
        return jsonify({
            "balance": 0.0,
            "balance_initial": 0.0,
            "pnl_today": 0.0,
            "pnl_pct": 0.0,
            "win_rate": 0.0,
            "win_rate_recent": 0.0,
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "consec_losses": 0,
            "is_paused": False,
            "drawdown_pct": 0.0,
        })

    # Última linha disponível
    last = df.iloc[-1]

    total = len(df)
    wins  = int((df["result"] == "WIN").sum()) if "result" in df.columns else 0
    losses = total - wins
    win_rate = round(wins / total * 100, 1) if total > 0 else 0.0

    balance       = float(last.get("balance_after", 0))
    balance_init  = float(df.iloc[0].get("balance_before", balance))
    pnl_today     = float(last.get("profit", 0)) if "profit" in df.columns else 0.0

    # Soma os profits do dia atual
    if "timestamp" in df.columns:
        try:
            df["_ts"] = pd.to_datetime(df["timestamp"], errors="coerce")
            today = pd.Timestamp.now().normalize()
            today_df = df[df["_ts"] >= today]
            pnl_today = float(today_df["profit"].sum()) if "profit" in today_df.columns else 0.0
        except Exception:
            pass

    pnl_pct      = round((pnl_today / balance_init * 100), 2) if balance_init != 0 else 0.0
    consec_losses = int(last.get("consec_losses", 0))
    drawdown_pct  = float(last.get("drawdown_pct", 0.0))
    win_rate_rec  = float(last.get("win_rate_recent", 0.0))

    return jsonify({
        "balance":          round(balance, 2),
        "balance_initial":  round(balance_init, 2),
        "pnl_today":        round(pnl_today, 2),
        "pnl_pct":          pnl_pct,
        "win_rate":         win_rate,
        "win_rate_recent":  round(win_rate_rec, 1),
        "total_trades":     total,
        "wins":             wins,
        "losses":           losses,
        "consec_losses":    consec_losses,
        "is_paused":        consec_losses >= 3,
        "drawdown_pct":     round(drawdown_pct, 2),
    })


# ─── API: Ticks ───────────────────────────────────────────────────────────────

@app.route("/api/ticks")
def api_ticks():
    n = min(int(request.args.get("n", 300)), 2000)
    df = _read_csv_safe(TICKS_CSV)

    if df.empty:
        return jsonify([])

    # Suporta formato antigo (sem cabeçalho) e novo (com cabeçalho)
    if "price" not in df.columns:
        df.columns = ["epoch", "price"] if len(df.columns) == 2 else list(df.columns)

    tail = df.tail(n)
    result = []
    for _, row in tail.iterrows():
        result.append({
            "epoch": int(row.get("epoch", 0)),
            "price": float(row.get("price", 0)),
        })
    return jsonify(result)


# ─── API: Trades ──────────────────────────────────────────────────────────────

@app.route("/api/trades")
def api_trades():
    n = min(int(request.args.get("n", 50)), 500)
    df = _read_csv_safe(OPERATIONS_CSV)

    if df.empty:
        return jsonify([])

    tail = df.tail(n).iloc[::-1]  # mais recente primeiro
    records = []
    for _, row in tail.iterrows():
        records.append({
            "timestamp":     str(row.get("timestamp", "")),
            "symbol":        str(row.get("symbol", "")),
            "direction":     str(row.get("direction", "")),
            "stake":         float(row.get("stake", 0)),
            "duration":      int(row.get("duration", 0)),
            "result":        str(row.get("result", "")),
            "profit":        float(row.get("profit", 0)),
            "balance_after": float(row.get("balance_after", 0)),
            "ema9":          float(row.get("ema9", 0)),
            "ema21":         float(row.get("ema21", 0)),
            "rsi":           float(row.get("rsi", 0)),
            "adx":           float(row.get("adx", 0)),
            "macd_hist":     float(row.get("macd_hist", 0)),
            "ai_confidence": float(row.get("ai_confidence", 0)),
            "ai_score":      float(row.get("ai_score", 0)),
            "market_condition": str(row.get("market_condition", "")),
        })
    return jsonify(records)


# ─── API: Indicadores (última operação) ───────────────────────────────────────

@app.route("/api/indicators")
def api_indicators():
    df = _read_csv_safe(OPERATIONS_CSV)

    if df.empty:
        return jsonify({})

    last = df.iloc[-1]
    return jsonify({
        "ema9":          float(last.get("ema9", 0)),
        "ema21":         float(last.get("ema21", 0)),
        "rsi":           float(last.get("rsi", 0)),
        "adx":           float(last.get("adx", 0)),
        "macd_hist":     float(last.get("macd_hist", 0)),
        "ai_confidence": float(last.get("ai_confidence", 0)),
        "ai_score":      float(last.get("ai_score", 0)),
        "market_condition": str(last.get("market_condition", "unknown")),
    })


# ─── API: Informações dos modelos ─────────────────────────────────────────────

@app.route("/api/model")
def api_model():
    ticks_count  = 0
    dataset_rows = 0

    df_ticks = _read_csv_safe(TICKS_CSV)
    if not df_ticks.empty:
        ticks_count = len(df_ticks)

    df_ds = _read_csv_safe(DATASET_CSV)
    if not df_ds.empty:
        dataset_rows = len(df_ds)

    model_mtime = None
    if MODEL_PKL.exists():
        model_mtime = int(MODEL_PKL.stat().st_mtime)

    return jsonify({
        "model_exists":   MODEL_PKL.exists(),
        "tft_exists":     TFT_PKL.exists(),
        "duration_exists": DURATION_PKL.exists(),
        "model_mtime":    model_mtime,
        "ticks_count":    ticks_count,
        "dataset_rows":   dataset_rows,
    })


# ─── API: Status do bot ───────────────────────────────────────────────────────

@app.route("/api/bot/status")
def api_bot_status():
    running, pid = _bot_running()

    uptime_sec = None
    mode = "unknown"

    if running and pid and _PSUTIL:
        try:
            proc = psutil.Process(pid)
            uptime_sec = int(time.time() - proc.create_time())
        except Exception:
            pass

    return jsonify({
        "running":    running,
        "pid":        pid,
        "uptime_sec": uptime_sec,
        "mode":       "demo",  # lido em tempo real do config.py se disponível
    })


# ─── API: Iniciar bot ─────────────────────────────────────────────────────────

@app.route("/api/bot/start", methods=["POST"])
def api_bot_start():
    running, pid = _bot_running()
    if running:
        return jsonify({"ok": False, "msg": f"Bot já está rodando (PID {pid})."}), 409

    data = request.get_json(silent=True) or {}
    mode = data.get("mode", "demo")
    balance = float(data.get("balance", 1000.0))
    skip_collect = bool(data.get("skip_collect", False))

    cmd = [sys.executable, str(PIPELINE_PY), "--demo" if mode == "demo" else "--real"]
    cmd += ["--balance", str(balance)]
    if skip_collect:
        cmd += ["--skip-collect"]

    try:
        global _bot_process
        proc = subprocess.Popen(
            cmd,
            cwd=str(_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        _bot_process = proc
        with _bot_log_lock:
            _bot_log_buffer.clear()
        threading.Thread(target=_read_bot_output, args=(proc,), daemon=True).start()
        PID_FILE.write_text(str(proc.pid))
        return jsonify({"ok": True, "pid": proc.pid, "mode": mode})
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)}), 500


# ─── API: Parar bot ───────────────────────────────────────────────────────────

@app.route("/api/bot/stop", methods=["POST"])
def api_bot_stop():
    running, pid = _bot_running()
    if not running or pid is None:
        return jsonify({"ok": False, "msg": "Bot não está rodando."}), 409

    try:
        if _PSUTIL:
            proc = psutil.Process(pid)
            children = proc.children(recursive=True)
            for child in children:
                child.terminate()
            proc.terminate()
        else:
            os.kill(pid, signal.SIGTERM)

        # Remove PID file
        if PID_FILE.exists():
            PID_FILE.unlink()

        return jsonify({"ok": True, "pid": pid})
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)}), 500


# ─── API: Logs do pipeline ───────────────────────────────────────────────────

@app.route("/api/bot/logs")
def api_bot_logs():
    with _bot_log_lock:
        lines = list(_bot_log_buffer)
    return jsonify({"lines": lines})


# ─── API: Equity curve ────────────────────────────────────────────────────────

@app.route("/api/equity")
def api_equity():
    df = _read_csv_safe(OPERATIONS_CSV)
    if df.empty:
        return jsonify([])

    result = []
    for _, row in df.iterrows():
        result.append({
            "timestamp":     str(row.get("timestamp", "")),
            "balance_after": float(row.get("balance_after", 0)),
            "result":        str(row.get("result", "")),
        })
    return jsonify(result)


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Deriv Bot Dashboard")
    print("  http://localhost:5055")
    print("  Ctrl+C para encerrar")
    print("=" * 55)
    app.run(host="0.0.0.0", port=5055, debug=False, use_reloader=False)
