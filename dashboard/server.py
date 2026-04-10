#!/home/stanis/.venv/bin/python3
"""
dashboard/server.py — Backend Flask do Dashboard do Bot Deriv.

Uso:
    cd /home/stanis/Repositorios/Binary
    .venv/bin/python3 dashboard/server.py
    # ou: python3 dashboard/server.py (com venv ativo)

Acesso: http://localhost:5055
"""

import collections
import json
import os
import signal
import subprocess
import sys
import threading
import time
from functools import wraps
from pathlib import Path

# Garante que imports do bot funcionam mesmo rodando de dentro de dashboard/
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False

# ─── Autenticação ─────────────────────────────────────────────────────────────────

# Defina DASHBOARD_TOKEN no .env ou como variável de ambiente.
# Se não definido, apenas acesso local sem token é permitido.
_DASHBOARD_TOKEN: str = os.environ.get("DASHBOARD_TOKEN", "")


def _require_token(f):
    """Decorator: exige 'X-Auth-Token' correto para rotas de controle do bot."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        # Permite requisições sem token apenas de localhost (sem token configurado)
        is_local = request.remote_addr in ("127.0.0.1", "::1")
        if _DASHBOARD_TOKEN:
            token = request.headers.get("X-Auth-Token", "")
            if token != _DASHBOARD_TOKEN:
                return jsonify({"ok": False, "msg": "Unauthorized"}), 401
        elif not is_local:
            # Sem token configurado: bloqueia qualquer origem não-local
            return jsonify({"ok": False, "msg": "Unauthorized — configure DASHBOARD_TOKEN"}), 401
        return f(*args, **kwargs)
    return wrapper

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
RISK_STATE_JSON = _ROOT / "risk_state.json"
STATE_JSON      = _ROOT / "state.json"
MODEL_METRICS_JSON = _ROOT / "model_metrics.json"
PIPELINE_LOG_TXT = _ROOT / "pipeline.log"

# ─── Flask App ────────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="")
CORS(app, origins=["http://localhost:*", "http://127.0.0.1:*"])

# Serializa NaN/Inf como null para produzir JSON válido no browser
import math as _math

class _SafeJSONProvider(app.json_provider_class):  # type: ignore[name-defined]
    def dumps(self, obj, **kw):
        import json as _json
        class _Enc(_json.JSONEncoder):
            def iterencode(self, o, _one_shot=False):
                return super().iterencode(self._clean(o), _one_shot)
            def _clean(self, o):
                if isinstance(o, float):
                    if _math.isnan(o) or _math.isinf(o):
                        return None
                    return o
                if isinstance(o, dict):
                    return {k: self._clean(v) for k, v in o.items()}
                if isinstance(o, (list, tuple)):
                    return [self._clean(v) for v in o]
                return o
            def default(self, o):
                return super().default(o)
        return _json.dumps(obj, cls=_Enc, **kw)

app.json_provider_class = _SafeJSONProvider
app.json = _SafeJSONProvider(app)

# ─── Log buffer do pipeline ──────────────────────────────────────────────────

_bot_log_buffer: collections.deque = collections.deque(maxlen=200)
_bot_process: subprocess.Popen | None = None
_bot_log_lock = threading.Lock()


def _read_bot_output(proc: subprocess.Popen) -> None:
    """Lê stdout/stderr do pipeline em thread e preenche o buffer circular."""
    stdout = proc.stdout
    if stdout is None:
        return
    try:
        with PIPELINE_LOG_TXT.open("a", encoding="utf-8") as log_file:
            for line in stdout:
                decoded = line.rstrip()
                if not decoded:
                    continue
                log_file.write(decoded + "\n")
                log_file.flush()
                with _bot_log_lock:
                    _bot_log_buffer.append(decoded)
        stdout.close()
    except Exception:
        pass


def _read_recent_pipeline_logs(max_lines: int = 500) -> list[str]:
    """Retorna as linhas mais recentes do log persistido do pipeline."""
    try:
        if not PIPELINE_LOG_TXT.exists() or PIPELINE_LOG_TXT.stat().st_size == 0:
            return []
        with PIPELINE_LOG_TXT.open("r", encoding="utf-8", errors="replace") as log_file:
            return [line.rstrip("\n") for line in collections.deque(log_file, maxlen=max_lines) if line.strip()]
    except Exception:
        return []


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


@app.route("/favicon.ico")
def favicon():
    """Retorna 204 No Content para suprimir o erro 404 do favicon."""
    from flask import Response
    return Response(status=204)


@app.route("/style.css")
def style_css():
    """Serve style.css do diretório estático raiz."""
    return send_from_directory(str(STATIC_DIR), "style.css")


@app.route("/app.js")
def app_js():
    """Serve app.js do diretório estático raiz."""
    return send_from_directory(str(STATIC_DIR), "app.js")


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
            "risk_state": _read_json_safe(RISK_STATE_JSON),
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
    risk_state    = _read_json_safe(RISK_STATE_JSON)

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
        "risk_state":       risk_state,
    })


# ─── API: Ticks ───────────────────────────────────────────────────────────────

@app.route("/api/ticks")
def api_ticks():
    n = min(int(request.args.get("n", 300)), 2000)
    symbol = str(request.args.get("symbol", "")).strip()
    from_epoch_raw = request.args.get("from_epoch")
    from_epoch = None
    if from_epoch_raw not in (None, ""):
        try:
            from_epoch = int(from_epoch_raw)
        except (TypeError, ValueError):
            from_epoch = None
    df = _read_csv_safe(TICKS_CSV)

    if df.empty:
        return jsonify([])

    # Suporta formato antigo (sem cabeçalho) e novo (com cabeçalho)
    if "price" not in df.columns:
        df.columns = ["epoch", "price"] if len(df.columns) == 2 else list(df.columns)

    if symbol and "symbol" in df.columns:
        df = df[df["symbol"].astype(str) == symbol]

    if from_epoch is not None and "epoch" in df.columns:
        try:
            df = df[df["epoch"].astype(int) >= from_epoch]
        except Exception:
            pass

    tail = df.tail(n)
    result = []
    for _, row in tail.iterrows():
        result.append({
            "epoch": int(row.get("epoch", 0)),
            "price": float(row.get("price", 0)),
            "symbol": str(row.get("symbol", "")),
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
@_require_token
def api_bot_start():
    running, pid = _bot_running()
    if running:
        return jsonify({"ok": False, "msg": f"Bot já está rodando (PID {pid})."}), 409

    data = request.get_json(silent=True) or {}
    mode = data.get("mode", "demo")
    if mode not in ("demo", "real"):
        return jsonify({"ok": False, "msg": "mode deve ser 'demo' ou 'real'"}), 400
    try:
        balance = float(data.get("balance", 1000.0))
        if balance <= 0:
            raise ValueError()
    except (TypeError, ValueError):
        return jsonify({"ok": False, "msg": "balance deve ser um número positivo"}), 400
    skip_collect = bool(data.get("skip_collect", False))
    no_scan = bool(data.get("no_scan", False))
    min_ticks = int(data.get("min_ticks", 200))
    history_count = int(data.get("history_count", 2000))
    force_retrain = bool(data.get("force_retrain", False))

    cmd = [sys.executable, "-u", str(PIPELINE_PY), "--demo" if mode == "demo" else "--real"]
    cmd += ["--balance", str(balance)]
    min_ticks_target = max(50, min_ticks)
    cmd += ["--min-ticks", str(min_ticks_target)]
    cmd += ["--history-count", str(max(0, history_count))]
    if skip_collect:
        cmd += ["--skip-collect"]
    if no_scan:
        cmd += ["--no-scan"]
    if force_retrain:
        cmd += ["--force-retrain"]

    # Reseta risk_state.json para que a nova sessão comece do zero
    # (sem herdar consec_losses, pause ou daily_pnl de execuções anteriores)
    try:
        import datetime as _dt
        _fresh_state = {
            "is_paused": False,
            "pause_remaining_sec": 0,
            "daily_pnl": 0.0,
            "daily_pnl_pct": 0.0,
            "consec_losses": 0,
            "drift_detected": False,
            "win_rate_recent": 0.0,
            "date": _dt.date.today().isoformat(),
            "balance": balance,
            "pause_until_epoch": 0.0,
            "recent_results": [],
        }
        RISK_STATE_JSON.write_text(json.dumps(_fresh_state))
    except Exception:
        pass

    # Registra timestamp de início da sessão em state.json
    try:
        import datetime as _dt2
        _cur_state = _read_json_safe(STATE_JSON)
        _cur_state["session_start"] = _dt2.datetime.now().isoformat()
        _cur_state["active_contract"] = None
        _cur_state["min_ticks_target"] = int(min_ticks_target)
        STATE_JSON.write_text(json.dumps(_cur_state))
    except Exception:
        pass

    try:
        global _bot_process
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(
            cmd,
            cwd=str(_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
            env=env,
        )
        _bot_process = proc
        with _bot_log_lock:
            _bot_log_buffer.clear()
        try:
            PIPELINE_LOG_TXT.write_text("", encoding="utf-8")
        except Exception:
            pass
        threading.Thread(target=_read_bot_output, args=(proc,), daemon=True).start()
        PID_FILE.write_text(str(proc.pid))
        return jsonify({"ok": True, "pid": proc.pid, "mode": mode})
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)}), 500


# ─── API: Parar bot ───────────────────────────────────────────────────────────

@app.route("/api/bot/stop", methods=["POST"])
@_require_token
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

        try:
            _state = _read_json_safe(STATE_JSON)
            _state["active_contract"] = None
            STATE_JSON.write_text(json.dumps(_state))
        except Exception:
            pass

        return jsonify({"ok": True, "pid": pid})
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)}), 500


# ─── API: Logs do pipeline ───────────────────────────────────────────────────

@app.route("/api/bot/logs")
def api_bot_logs():
    with _bot_log_lock:
        lines = list(_bot_log_buffer)
    if not lines:
        lines = _read_recent_pipeline_logs()
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


# ─── API: Candle Patterns (alertas de padrões de vela) ────────────────────────

@app.route("/api/candle-patterns")
def api_candle_patterns():
    """Retorna padrões de vela detectados a partir dos ticks mais recentes."""
    try:
        import indicators as ind
        from config import CANDLE_SIZE, PA_SR_TOLERANCE

        df = _read_csv_safe(TICKS_CSV)
        if df.empty:
            return jsonify([])

        if "price" in df.columns:
            prices = df["price"].astype(float).tolist()
        elif df.shape[1] >= 4:
            prices = df.iloc[:, 3].astype(float).tolist()
        else:
            prices = df.iloc[:, -1].astype(float).tolist()

        # Usar últimos 500 ticks
        prices = prices[-500:]
        candles = ind.ticks_to_candles(prices, CANDLE_SIZE)

        if len(candles) < 4:
            return jsonify([])

        patterns = ind.detect_candle_patterns(candles)
        pa = ind.price_action_features(candles, PA_SR_TOLERANCE)

        result = []
        for p in patterns:
            context_parts = []
            if pa is not None:
                if pa["pa_demand_zone"] > 0.3:
                    context_parts.append("Demand Zone")
                if pa["pa_supply_zone"] > 0.3:
                    context_parts.append("Supply Zone")
                if pa["pa_sr_distance"] < 0.3:
                    pos = pa["pa_sr_position"]
                    context_parts.append("Suporte" if pos < -0.3 else "Resistência" if pos > 0.3 else "S/R")

            result.append({
                "name": p["name"],
                "direction": p["direction"],
                "strength": p["strength"],
                "price": round(prices[-1], 4),
                "context": " + ".join(context_parts) if context_parts else "",
            })

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── Fase 3: helpers de leitura eficiente ────────────────────────────────────

def _tail_csv(path, n: int) -> pd.DataFrame:
    """Lê apenas as últimas `n` linhas de um CSV sem carregar o arquivo inteiro."""
    try:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return pd.DataFrame()
        with open(path, "rb") as f:
            # Lê cabeçalho
            header = f.readline().decode("utf-8", errors="replace").rstrip()
            cols = header.split(",")
            # Busca final do arquivo e lê últimas n linhas
            f.seek(0, 2)
            file_size = f.tell()
            chunk = min(file_size, n * 80)  # estimativa de ~80 bytes por linha
            f.seek(max(0, file_size - chunk))
            raw = f.read().decode("utf-8", errors="replace")
        lines = [l for l in raw.splitlines() if l.strip()]
        # Remove cabeçalho se apareceu no meio
        data_lines = [l for l in lines if not l.startswith(cols[0])]
        tail_lines = data_lines[-n:]
        import io
        csv_text = header + "\n" + "\n".join(tail_lines)
        return pd.read_csv(io.StringIO(csv_text))
    except Exception:
        return _read_csv_safe(path)


def _read_json_safe(path) -> dict:
    try:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


_candle_patterns_cache: dict = {"ts": 0.0, "data": []}
_CANDLE_CACHE_TTL = 5.0  # segundos


# ─── API: Estado consolidado (/api/state) ─────────────────────────────────────

@app.route("/api/state")
def api_state():
    """Endpoint único que retorna todos os dados do dashboard em um só JSON."""
    running, pid = _bot_running()

    # — Resumo de operações (filtrado pela sessão atual) —
    df_ops_all = _read_csv_safe(OPERATIONS_CSV)
    active_state_pre = _read_json_safe(STATE_JSON)
    session_start_str = active_state_pre.get("session_start", "")

    df_ops = df_ops_all.copy()
    if session_start_str and not df_ops.empty and "timestamp" in df_ops.columns:
        try:
            _ss = pd.to_datetime(session_start_str, errors="coerce")
            df_ops["_ts"] = pd.to_datetime(df_ops["timestamp"], errors="coerce")
            df_ops = df_ops[df_ops["_ts"] >= _ss].reset_index(drop=True)
        except Exception:
            pass
    summary = {
        "balance": 0.0, "balance_initial": 0.0, "pnl_today": 0.0, "pnl_pct": 0.0,
        "win_rate": 0.0, "win_rate_recent": 0.0, "total_trades": 0,
        "wins": 0, "losses": 0, "consec_losses": 0, "is_paused": False, "drawdown_pct": 0.0,
    }
    last_trade_indicators = {}
    if not df_ops.empty:
        last = df_ops.iloc[-1]
        total = len(df_ops)
        wins  = int((df_ops["result"] == "WIN").sum()) if "result" in df_ops.columns else 0
        balance      = float(last.get("balance_after", 0))
        balance_init = float(df_ops.iloc[0].get("balance_before", balance))
        pnl_today    = 0.0
        if "timestamp" in df_ops.columns:
            try:
                df_ops["_ts"] = pd.to_datetime(df_ops["timestamp"], errors="coerce")
                today = pd.Timestamp.now().normalize()
                today_df = df_ops[df_ops["_ts"] >= today]
                pnl_today = float(today_df["profit"].sum()) if "profit" in today_df.columns else 0.0
            except Exception:
                pass
        summary = {
            "balance":         round(balance, 2),
            "balance_initial": round(balance_init, 2),
            "pnl_today":       round(pnl_today, 2),
            "pnl_pct":         round((pnl_today / balance_init * 100), 2) if balance_init != 0 else 0.0,
            "win_rate":        round(wins / total * 100, 1) if total > 0 else 0.0,
            "win_rate_recent": float(last.get("win_rate_recent", 0.0)),
            "total_trades":    total,
            "wins":            wins,
            "losses":          total - wins,
            "consec_losses":   int(last.get("consec_losses", 0)),
            "is_paused":       int(last.get("consec_losses", 0)) >= 3,
            "drawdown_pct":    round(float(last.get("drawdown_pct", 0.0)), 2),
        }
        last_trade_indicators = {
            "ema9": float(last.get("ema9", 0)), "ema21": float(last.get("ema21", 0)),
            "rsi": float(last.get("rsi", 0)), "adx": float(last.get("adx", 0)),
            "macd_hist": float(last.get("macd_hist", 0)),
            "ai_confidence": float(last.get("ai_confidence", 0)),
            "ai_score": float(last.get("ai_score", 0)),
            "market_condition": str(last.get("market_condition", "unknown")),
        }

    # — Indicadores ao vivo (últimos 500 ticks) —
    live_indicators = last_trade_indicators.copy()
    try:
        from feature_engine import compute_feature_map
        df_ticks = _tail_csv(TICKS_CSV, 500)
        if not df_ticks.empty:
            if "price" not in df_ticks.columns and df_ticks.shape[1] >= 2:
                df_ticks.columns = ["epoch", "price"] if df_ticks.shape[1] == 2 else list(df_ticks.columns)
            prices = df_ticks["price"].astype(float).tolist() if "price" in df_ticks.columns else []
            if len(prices) >= 50:
                fm = compute_feature_map(prices)
                if fm is not None:
                    live_indicators = {
                        "ema9": round(fm["ema9"], 5), "ema21": round(fm["ema21"], 5),
                        "rsi": round(fm["rsi"], 2), "adx": round(fm["adx"], 2),
                        "macd_hist": round(fm["macd_hist"], 6),
                        "ai_confidence": last_trade_indicators.get("ai_confidence", 0),
                        "ai_score": last_trade_indicators.get("ai_score", 0),
                        "market_condition": last_trade_indicators.get("market_condition", "unknown"),
                    }
    except Exception:
        pass

    # — Modelo —
    model_info = {
        "model_exists": MODEL_PKL.exists(),
        "tft_exists": TFT_PKL.exists(),
        "duration_exists": DURATION_PKL.exists(),
        "model_mtime": int(MODEL_PKL.stat().st_mtime) if MODEL_PKL.exists() else None,
        "ticks_count": 0,
        "dataset_rows": 0,
        "min_ticks_target": 500,
    }
    try:
        df_t = _read_csv_safe(TICKS_CSV)
        if not df_t.empty:
            if session_start_str and "epoch" in df_t.columns:
                try:
                    _ss_epoch = int(pd.Timestamp(session_start_str).timestamp())
                    model_info["ticks_count"] = int((df_t["epoch"].astype(int) >= _ss_epoch).sum())
                except Exception:
                    model_info["ticks_count"] = len(df_t)
            else:
                model_info["ticks_count"] = len(df_t)
        df_d = _read_csv_safe(DATASET_CSV)
        if not df_d.empty:
            model_info["dataset_rows"] = len(df_d)
    except Exception:
        pass

    try:
        model_info["min_ticks_target"] = int(active_state_pre.get("min_ticks_target", 500) or 500)
    except Exception:
        model_info["min_ticks_target"] = 500

    # — Métricas do último treino (model_metrics.json) —
    try:
        _mm_data = _read_json_safe(MODEL_METRICS_JSON)
        if isinstance(_mm_data, list) and _mm_data:
            _mm = _mm_data[-1]
            model_info["last_metrics"] = {
                "best_model": _mm.get("best_model"),
                "accuracy":   _mm.get("accuracy"),
                "auc":        _mm.get("auc"),
                "f1":         _mm.get("f1"),
                "n_train":    _mm.get("n_train"),
                "n_test":     _mm.get("n_test"),
                "timestamp":  _mm.get("timestamp"),
            }
        else:
            model_info["last_metrics"] = {}
    except Exception:
        model_info["last_metrics"] = {}

    # — Risk state e state.json —
    risk_state = _read_json_safe(RISK_STATE_JSON)
    active_state = active_state_pre  # já lido acima para session_start
    active_contract = active_state.get("active_contract")
    if not isinstance(active_contract, dict):
        active_contract = {"has_active": False}

    # — Bot status —
    uptime_sec = None
    if running and pid and _PSUTIL:
        try:
            uptime_sec = int(time.time() - psutil.Process(pid).create_time())
        except Exception:
            pass

    return jsonify({
        "bot": {"running": running, "pid": pid, "uptime_sec": uptime_sec},
        "summary": summary,
        "indicators": live_indicators,
        "model": model_info,
        "risk_state": risk_state,
        "active_symbol": active_state.get("symbol", ""),
        "active_contract": active_contract,
    })


# ─── API: Contrato ativo ─────────────────────────────────────────────────────

@app.route("/api/contract-active")
def api_contract_active():
    state = _read_json_safe(STATE_JSON)
    ac = state.get("active_contract")
    if not isinstance(ac, dict) or not ac.get("has_active"):
        return jsonify({"has_active": False})

    payload = {
        "has_active": True,
        "contract_id": str(ac.get("contract_id", "")),
        "symbol": str(ac.get("symbol", state.get("symbol", ""))),
        "direction": str(ac.get("direction", "")),
        "duration": int(ac.get("duration", 0) or 0),
        "buy_timestamp": float(ac.get("buy_timestamp", 0.0) or 0.0),
        "entry_epoch": int(ac.get("entry_epoch", 0) or 0),
        "entry_price": float(ac.get("entry_price", 0.0) or 0.0),
    }
    return jsonify(payload)


# ─── API: Séries históricas de indicadores (/api/indicator-series) ────────────

@app.route("/api/indicator-series")
def api_indicator_series():
    """Calcula indicadores para os últimos N candles de tempo."""
    n_candles = min(int(request.args.get("n", 60)), 300)
    try:
        from feature_engine import compute_feature_map
        from config import CANDLE_SIZE, CANDLE_TIMEFRAME_SEC
        import indicators as ind

        window_size = 50  # mínimo de candles para indicadores
        ticks_needed = (n_candles + window_size + 5) * max(CANDLE_TIMEFRAME_SEC, CANDLE_SIZE)
        df = _tail_csv(TICKS_CSV, ticks_needed)

        if df.empty:
            return jsonify([])

        if "price" not in df.columns and df.shape[1] >= 2:
            df.columns = ["epoch", "price"] if df.shape[1] == 2 else list(df.columns)

        prices_raw = df["price"].astype(float).tolist() if "price" in df.columns else []
        epochs_raw = df["epoch"].astype(int).tolist() if "epoch" in df.columns else list(range(len(prices_raw)))

        # Determina se dados são ticks brutos ou closes de candles
        if len(epochs_raw) >= 2:
            avg_gap = (epochs_raw[-1] - epochs_raw[0]) / max(len(epochs_raw) - 1, 1)
            if avg_gap >= CANDLE_TIMEFRAME_SEC * 0.8:
                prices = prices_raw
                epochs = epochs_raw
            else:
                ticks_list = [{"epoch": e, "price": p} for e, p in zip(epochs_raw, prices_raw)]
                candles_agg = ind.ticks_to_candles_by_time(ticks_list, CANDLE_TIMEFRAME_SEC)
                prices = [c["close"] for c in candles_agg]
                epochs = [c["epoch"] for c in candles_agg]
        else:
            prices = prices_raw
            epochs = epochs_raw

        if len(prices) < window_size + 1:
            return jsonify([])

        result = []
        for end in range(window_size, len(prices)):
            window = prices[max(0, end - window_size):end]
            fm = compute_feature_map(window)
            if fm is None:
                continue
            result.append({
                "epoch":     epochs[end - 1] if end - 1 < len(epochs) else 0,
                "rsi":       round(fm["rsi"], 2),
                "adx":       round(fm["adx"], 2),
                "macd_hist": round(fm["macd_hist"], 6),
                "ema9":      round(fm["ema9"], 5),
                "ema21":     round(fm["ema21"], 5),
            })

        return jsonify(result[-n_candles:])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── API: Histórico de métricas do modelo ─────────────────────────────────────

@app.route("/api/model-history")
def api_model_history():
    data = _read_json_safe(MODEL_METRICS_JSON)
    resp = jsonify(data) if isinstance(data, list) else jsonify([])
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    return resp


# ─── API: Métricas do último treino ───────────────────────────────────────────

@app.route("/api/model-metrics")
def api_model_metrics():
    data = _read_json_safe(MODEL_METRICS_JSON)
    if not isinstance(data, list) or not data:
        return jsonify({})
    # Prefere a última entrada com stage "final" (que contém models_comparison).
    # Fallback para a última entrada disponível.
    final_entries = [e for e in data if e.get("stage") == "final"]
    entry = final_entries[-1] if final_entries else data[-1]
    resp = jsonify(entry)
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    return resp


# ─── API: Uso de CPU/RAM do sistema e do bot ──────────────────────────────────

@app.route("/api/system")
def api_system():
    result: dict = {
        "cpu_pct":     None,
        "ram_pct":     None,
        "ram_used_mb": None,
        "disk_pct":    None,
        "bot_cpu_pct": None,
        "bot_ram_mb":  None,
    }
    if not _PSUTIL:
        return jsonify(result)

    try:
        result["cpu_pct"]     = psutil.cpu_percent(interval=0.1)
        vm                    = psutil.virtual_memory()
        result["ram_pct"]     = vm.percent
        result["ram_used_mb"] = round(vm.used / 1024 / 1024, 1)
        du                    = psutil.disk_usage("/")
        result["disk_pct"]    = du.percent
    except Exception:
        pass

    running, pid = _bot_running()
    if running and pid:
        try:
            proc = psutil.Process(pid)
            result["bot_cpu_pct"] = round(proc.cpu_percent(interval=0.1), 1)
            result["bot_ram_mb"]  = round(proc.memory_info().rss / 1024 / 1024, 1)
        except Exception:
            pass

    return jsonify(result)


# ─── API: Estatísticas avançadas de operações ─────────────────────────────────

@app.route("/api/stats")
def api_stats():
    df = _read_csv_safe(OPERATIONS_CSV)
    if df.empty:
        return jsonify({})

    try:
        profits = df["profit"].astype(float) if "profit" in df.columns else pd.Series([], dtype=float)
        results = df["result"].astype(str) if "result" in df.columns else pd.Series([], dtype=str)
        dirs    = df["direction"].astype(str) if "direction" in df.columns else pd.Series([], dtype=str)

        wins_mask  = results == "WIN"
        loss_mask  = results == "LOSS"

        total_wins  = int(wins_mask.sum())
        total_loss  = int(loss_mask.sum())
        total       = len(df)

        avg_win  = float(profits[wins_mask].mean())  if total_wins  > 0 else 0.0
        avg_loss = float(profits[loss_mask].mean())  if total_loss  > 0 else 0.0
        gross_profit = float(profits[wins_mask].sum()) if total_wins > 0 else 0.0
        gross_loss   = abs(float(profits[loss_mask].sum())) if total_loss > 0 else 0.0

        profit_factor = round(gross_profit / gross_loss, 3) if gross_loss > 0 else None
        wr = total_wins / total if total > 0 else 0
        lr = total_loss / total if total > 0 else 0
        expectancy = round(wr * avg_win + lr * avg_loss, 4) if total > 0 else 0.0

        # Distribuição de profit (10 bins)
        hist_bins = [-5, -2, -1, -0.5, 0, 0.5, 1, 2, 5, 100]
        hist_labels = ["<-5", "-5a-2", "-2a-1", "-1a-0.5", "-0.5a0", "0a0.5", "0.5a1", "1a2", "2a5", ">5"]
        hist_counts = []
        for i in range(len(hist_bins) - 1):
            lo, hi = hist_bins[i], hist_bins[i + 1]
            cnt = int(((profits >= lo) & (profits < hi)).sum())
            hist_counts.append(cnt)

        # Por direção
        dir_stats = {}
        for d_val in ["CALL", "PUT", "BUY", "SELL"]:
            mask = dirs == d_val
            if not mask.any():
                continue
            dw = int((results[mask] == "WIN").sum())
            dt = int(mask.sum())
            dir_stats[d_val] = {
                "total":  dt,
                "wins":   dw,
                "wr":     round(dw / dt * 100, 1) if dt > 0 else 0.0,
                "profit": round(float(profits[mask].sum()), 2),
            }

        return jsonify({
            "total":           total,
            "avg_win":         round(avg_win, 4),
            "avg_loss":        round(avg_loss, 4),
            "gross_profit":    round(gross_profit, 2),
            "gross_loss":      round(gross_loss, 2),
            "profit_factor":   profit_factor,
            "expectancy":      expectancy,
            "hist_labels":     hist_labels,
            "hist_counts":     hist_counts,
            "by_direction":    dir_stats,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── API: Candles OHLC para gráfico de candlestick ────────────────────────────

@app.route("/api/candles")
def api_candles():
    n = min(int(request.args.get("n", 100)), 500)
    symbol = str(request.args.get("symbol", "")).strip()
    from_epoch_raw = request.args.get("from_epoch")
    from_epoch = None
    if from_epoch_raw not in (None, ""):
        try:
            from_epoch = int(from_epoch_raw)
        except (TypeError, ValueError):
            from_epoch = None
    try:
        from config import CANDLE_SIZE, CANDLE_TIMEFRAME_SEC
        import indicators as ind

        # Para candles por tempo precisamos de (n + margem) * max_ticks_por_candle
        # Estimativa: 1 tick/s × CANDLE_TIMEFRAME_SEC por vela
        ticks_needed = (n + 5) * max(CANDLE_TIMEFRAME_SEC, CANDLE_SIZE)
        df = _tail_csv(TICKS_CSV, ticks_needed)
        if df.empty:
            return jsonify([])

        if "price" not in df.columns and df.shape[1] >= 2:
            df.columns = ["epoch", "price"] if df.shape[1] == 2 else list(df.columns)

        if symbol and "symbol" in df.columns:
            df = df[df["symbol"].astype(str) == symbol]

        if from_epoch is not None and "epoch" in df.columns:
            try:
                df = df[df["epoch"].astype(int) >= from_epoch]
            except Exception:
                pass

        if df.empty:
            return jsonify([])

        epochs = df["epoch"].astype(int).tolist() if "epoch" in df.columns else list(range(len(df)))
        prices = df["price"].astype(float).tolist() if "price" in df.columns else []

        # Tenta candles por tempo primeiro (quando há epoch e dados consistentes)
        candles = []
        if "epoch" in df.columns and len(epochs) >= 2:
            avg_gap = (epochs[-1] - epochs[0]) / max(len(epochs) - 1, 1)
            if avg_gap >= CANDLE_TIMEFRAME_SEC * 0.8:
                # Dados já são closes de candles — formata diretamente
                result = []
                for i, (ep, pr) in enumerate(zip(epochs[-n:], prices[-n:])):
                    result.append({
                        "t": ep * 1000,
                        "o": round(pr, 5), "h": round(pr, 5),
                        "l": round(pr, 5), "c": round(pr, 5),
                    })
                return jsonify(result)
            else:
                # Agrega ticks em candles de tempo
                ticks_list = [{"epoch": e, "price": p} for e, p in zip(epochs, prices)]
                candles = ind.ticks_to_candles_by_time(ticks_list, CANDLE_TIMEFRAME_SEC)

        # Fallback: candles por contagem de ticks
        if not candles:
            candles = ind.ticks_to_candles(prices, CANDLE_SIZE)
            result = []
            for idx, c in enumerate(candles[-n:]):
                tick_idx = idx * CANDLE_SIZE
                epoch = epochs[tick_idx] if tick_idx < len(epochs) else 0
                result.append({
                    "t": epoch * 1000,
                    "o": round(c["open"],  5), "h": round(c["high"],  5),
                    "l": round(c["low"],   5), "c": round(c["close"], 5),
                })
            return jsonify(result)

        result = []
        for c in candles[-n:]:
            result.append({
                "t": int(c["epoch"]) * 1000,
                "o": round(c["open"],  5), "h": round(c["high"],  5),
                "l": round(c["low"],   5), "c": round(c["close"], 5),
            })
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Deriv Bot Dashboard")
    print("  http://localhost:5055")
    if _DASHBOARD_TOKEN:
        print(f"  Auth: X-Auth-Token requerido")
    else:
        print("  Auth: desativada (defina DASHBOARD_TOKEN no .env para ativar)")
    print("  Ctrl+C para encerrar")
    print("=" * 55)
    app.run(host="127.0.0.1", port=5055, debug=False, use_reloader=False)
