#!/usr/bin/env python3
"""
dashboard/server.py — Backend Flask do Dashboard do Bot Deriv.

Uso:
    cd /caminho/para/StanDeriv
    .venv/bin/python3 dashboard/server.py
    # ou: python3 dashboard/server.py (com venv ativo)

Acesso: http://localhost:5055
"""

import collections
import datetime as dt
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from functools import wraps
from pathlib import Path

# Garante que imports do bot funcionam mesmo rodando de dentro de dashboard/
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import pandas as pd
import websocket
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from model_contract import FEATURE_WINDOW, minimum_history_candles

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

TICKS_CSV      = _ROOT / "ticks_v2.csv"
OPERATIONS_CSV = _ROOT / "operacoes_v2.csv"
DATASET_CSV    = _ROOT / "dataset_v2.csv"
MODEL_PKL      = _ROOT / "model.pkl"
TFT_PKL        = _ROOT / "transformer_model.pkl"
DURATION_PKL   = _ROOT / "duration_model.pkl"
PIPELINE_PY    = _ROOT / "pipeline.py"
BACKTEST_PY    = _ROOT / "backtest.py"
PID_FILE       = _ROOT / "dashboard" / "bot.pid"
STATIC_DIR     = Path(__file__).resolve().parent
RISK_STATE_JSON = _ROOT / "risk_state.json"
STATE_JSON      = _ROOT / "state.json"
MODEL_METRICS_JSON = _ROOT / "model_metrics.json"
PIPELINE_LOG_TXT = _ROOT / "pipeline.log"
BOT_STATUS_JSON = _ROOT / "bot_status.json"
BACKTEST_ROOT = _ROOT / "backtest_reports" / "dashboard"
BACKTEST_STATE_JSON = BACKTEST_ROOT / "state.json"
MIN_HISTORY_CANDLES = minimum_history_candles()

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
_bot_status_lock = threading.RLock()
_deriv_contract_cache: dict = {"ts": 0.0, "records": [], "error": ""}
_DERIV_CONTRACT_CACHE_TTL = 15.0

# Backtests rodam em subprocesso para não bloquear o servidor Flask nem
# compartilhar estado dos modelos com o processo do bot.
_backtest_process: subprocess.Popen | None = None
_backtest_log_buffer: collections.deque = collections.deque(maxlen=200)
_backtest_lock = threading.RLock()
_backtest_state: dict = {}


def _read_bot_output(proc: subprocess.Popen) -> None:
    """Lê stdout/stderr do pipeline em thread e preenche o buffer circular."""
    stdout = proc.stdout
    if stdout is None:
        return
    returncode = None
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
    finally:
        try:
            returncode = proc.wait(timeout=0)
        except subprocess.TimeoutExpired:
            returncode = proc.poll()
        except Exception:
            returncode = None
        if not isinstance(returncode, int):
            returncode = None
        if returncode is not None:
            _mark_bot_stopped(returncode, requested=_get_bot_status().get("status") == "stopping")


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
        # Tentativa padrão (arquivo com cabeçalho)
        df = pd.read_csv(path, **kwargs)
        if "price" in df.columns or path != TICKS_CSV:
            return df

        # Se não houver 'price' e estivermos lendo ticks.csv, é provável que o CSV
        # não tenha cabeçalho. Recarrega sem header e atribui nomes de coluna conforme
        # o número de colunas detectadas.
        df_no_header = pd.read_csv(path, header=None, **kwargs)
        cols = df_no_header.shape[1]
        if cols >= 4:
            # epoch, datetime, symbol, price (+ possíveis colunas extra)
            names = ["epoch", "datetime", "symbol", "price"]
            if cols > 4:
                names += [f"col{i}" for i in range(4, cols)]
            df_no_header.columns = names
        elif cols == 2:
            df_no_header.columns = ["epoch", "price"]
        else:
            # fallback: manter nomes numéricos
            df_no_header.columns = list(df_no_header.columns)
        return df_no_header
    except Exception:
        return pd.DataFrame()


def _last_local_contract() -> dict | None:
    df = _read_csv_safe(OPERATIONS_CSV)
    if df.empty:
        return None
    row = df.iloc[-1].to_dict()

    def _float(name: str) -> float:
        try:
            value = float(row.get(name, 0) or 0)
            return value if pd.notna(value) else 0.0
        except Exception:
            return 0.0

    def _int(name: str) -> int:
        try:
            value = int(float(row.get(name, 0) or 0))
            return value if value > 0 else 0
        except Exception:
            return 0

    return {
        "has_active": False,
        "closed": True,
        "contract_id": str(row.get("contract_id", "") or ""),
        "symbol": str(row.get("symbol", "") or ""),
        "direction": str(row.get("direction", "") or ""),
        "duration": int(_float("duration")),
        "duration_unit": str(row.get("duration_unit", "m") or "m"),
        "entry_epoch": _int("entry_epoch"),
        "exit_epoch": _int("exit_epoch"),
        "entry_price": _float("entry_price"),
        "exit_price": _float("exit_price"),
        "payout": _float("payout"),
        "buy_price": _float("buy_price") or _float("stake"),
        "profit": _float("profit"),
        "result": str(row.get("result", "") or ""),
        "timestamp": str(row.get("timestamp", "") or ""),
    }


def _deriv_credentials_available() -> bool:
    try:
        from config import TOKEN
        token = str(TOKEN or "").strip()
    except Exception:
        return False
    if not token:
        return False
    lowered = token.lower()
    return not (
        lowered.startswith("test_")
        or lowered.startswith("offline_")
        or lowered in {"test_token", "test_token_placeholder", "offline_test_token"}
    )


def _request_deriv_profit_table(limit: int = 100) -> list[dict]:
    """Busca o relatório oficial de contratos na Deriv sem persistir no repo."""
    if not _deriv_credentials_available():
        return []

    from config import APP_ID, DEMO_MODE, DERIV_AUTH_MODE, TOKEN
    from deriv_session import is_bearer_token, setup_bearer_options_session

    token = str(TOKEN).strip()
    app_id = str(APP_ID).strip() or "1089"
    ws_url = f"wss://ws.derivws.com/websockets/v3?app_id={app_id}"
    needs_authorize = True

    if is_bearer_token(token, DERIV_AUTH_MODE):
        session = setup_bearer_options_session(
            app_id=app_id,
            token=token,
            demo=bool(DEMO_MODE),
            timeout_sec=15,
        )
        ws_url = str(session["ws_url"])
        needs_authorize = False

    ws = websocket.create_connection(ws_url, timeout=15)
    try:
        if needs_authorize:
            ws.send(json.dumps({"authorize": token}))
            auth = json.loads(ws.recv())
            if auth.get("error"):
                raise RuntimeError(str(auth["error"].get("message", "Falha ao autenticar na Deriv")))

        ws.send(json.dumps({
            "profit_table": 1,
            "description": 1,
            "limit": max(1, min(int(limit), 500)),
            "sort": "DESC",
        }))
        response = json.loads(ws.recv())
        if response.get("error"):
            raise RuntimeError(str(response["error"].get("message", "Falha ao consultar profit_table")))
        table = response.get("profit_table") or {}
        transactions = table.get("transactions") or []
        return [item for item in transactions if isinstance(item, dict)]
    finally:
        try:
            ws.close()
        except Exception:
            pass


def _contract_direction(transaction: dict) -> str:
    contract_type = str(transaction.get("contract_type", "")).upper()
    if contract_type in {"CALL", "MULTUP", "ASIANU", "DIGITOVER"}:
        return "BUY"
    if contract_type in {"PUT", "MULTDOWN", "ASIAND", "DIGITUNDER"}:
        return "SELL"
    longcode = str(transaction.get("longcode") or transaction.get("description") or "").lower()
    if "higher" in longcode or "rise" in longcode or "call" in longcode:
        return "BUY"
    if "lower" in longcode or "fall" in longcode or "put" in longcode:
        return "SELL"
    return contract_type or "—"


def _normalize_deriv_contract(transaction: dict) -> dict:
    buy_price = float(transaction.get("buy_price", 0.0) or 0.0)
    sell_price = float(transaction.get("sell_price", 0.0) or 0.0)
    profit = float(transaction.get("profit", sell_price - buy_price) or 0.0)
    purchase_time = int(float(transaction.get("purchase_time", transaction.get("date_start", 0)) or 0))
    sell_time = int(float(transaction.get("sell_time", transaction.get("date_expiry", purchase_time)) or purchase_time))
    duration_sec = max(0, sell_time - purchase_time)
    duration_min = int(round(duration_sec / 60)) if duration_sec else 0
    result = "WIN" if profit > 0 else "LOSS" if profit < 0 else "DRAW"
    contract_id = str(transaction.get("contract_id") or transaction.get("transaction_id") or "")

    return {
        "timestamp": dt.datetime.fromtimestamp(purchase_time).isoformat() if purchase_time else "",
        "symbol": str(transaction.get("underlying_symbol") or transaction.get("symbol") or ""),
        "direction": _contract_direction(transaction),
        "stake": buy_price,
        "sell_price": sell_price,
        "duration": duration_min,
        "result": result,
        "profit": round(profit, 2),
        "balance_after": None,
        "ema9": 0.0,
        "ema21": 0.0,
        "rsi": 0.0,
        "adx": 0.0,
        "macd_hist": 0.0,
        "ai_confidence": 0.0,
        "ai_score": 0.0,
        "market_condition": "",
        "contract_id": contract_id,
        "payout": float(transaction.get("payout", 0.0) or 0.0),
    }


def _deriv_contract_report(limit: int = 100, *, force: bool = False) -> list[dict]:
    global _deriv_contract_cache
    now = time.time()
    if (
        not force
        and now - float(_deriv_contract_cache.get("ts", 0.0) or 0.0) < _DERIV_CONTRACT_CACHE_TTL
    ):
        return list(_deriv_contract_cache.get("records", []))

    try:
        transactions = _request_deriv_profit_table(limit=limit)
        records = [_normalize_deriv_contract(item) for item in transactions]
        _deriv_contract_cache = {"ts": now, "records": records, "error": ""}
        return records
    except Exception as exc:
        _deriv_contract_cache = {"ts": now, "records": [], "error": str(exc)}
        return []


def _contracts_frame(limit: int = 500) -> pd.DataFrame:
    return pd.DataFrame(_deriv_contract_report(limit=limit))


def _bot_pid() -> int | None:
    """Retorna PID salvo em bot.pid ou None."""
    try:
        if PID_FILE.exists():
            pid = int(PID_FILE.read_text().strip())
            return pid
    except (ValueError, OSError):
        pass
    try:
        status = _get_bot_status()
        if (
            bool(status.get("running"))
            or str(status.get("status", "")) in {"starting", "running", "stopping"}
        ):
            pid = int(status.get("pid") or 0)
            return pid if pid > 0 else None
    except (TypeError, ValueError):
        pass
    return None


def _clear_stale_bot_pid(pid: int | None = None) -> None:
    try:
        if PID_FILE.exists():
            current = int(PID_FILE.read_text().strip())
            if pid is None or current == int(pid):
                PID_FILE.unlink()
    except Exception:
        pass
    _mark_stale_bot_status(pid)


def _mark_stale_bot_status(pid: int | None = None) -> None:
    try:
        state = _get_bot_status()
        status_pid = int(state.get("pid") or 0)
        if pid is not None and status_pid and status_pid != int(pid):
            return
        if (
            not state.get("running")
            and str(state.get("status", "")) not in {"starting", "running", "stopping"}
        ):
            return
        now = _utc_now_iso()
        state.update(
            status="crashed",
            running=False,
            stopped_at=now,
            finished_at=now,
            returncode=state.get("returncode"),
            last_error=state.get("last_error") or "processo ausente",
        )
        _persist_bot_status(state)
    except Exception:
        pass


def _is_pipeline_process(proc) -> bool:
    try:
        if not proc.is_running() or proc.status() == psutil.STATUS_ZOMBIE:
            return False
        cmdline = " ".join(proc.cmdline()).replace("\\", "/")
        cwd = ""
        try:
            cwd = str(proc.cwd())
        except Exception:
            pass
        return "pipeline.py" in cmdline and (not cwd or Path(cwd).resolve() == _ROOT.resolve())
    except Exception:
        return False


def _bot_running() -> tuple[bool, int | None]:
    """(is_running, pid) — verifica se o processo pipeline.py existe."""
    pid = _bot_pid()
    if pid is None:
        return False, None
    if _PSUTIL:
        try:
            proc = psutil.Process(pid)
            if _is_pipeline_process(proc):
                return True, pid
            _clear_stale_bot_pid(pid)
            return False, None
        except psutil.NoSuchProcess:
            _clear_stale_bot_pid(pid)
            return False, None
    else:
        # Fallback sem psutil: valida /proc quando disponível; não confia em PID
        # reutilizado por outro processo.
        cmdline_path = Path("/proc") / str(pid) / "cmdline"
        try:
            if cmdline_path.exists():
                cmdline = cmdline_path.read_text(errors="replace").replace("\x00", " ")
                if "pipeline.py" in cmdline:
                    return True, pid
                _clear_stale_bot_pid(pid)
                return False, None
        except Exception:
            pass
        try:
            os.kill(pid, 0)
            _clear_stale_bot_pid(pid)
            return False, None
        except (ProcessLookupError, PermissionError):
            _clear_stale_bot_pid(pid)
            return False, None


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _append_bot_log(message: str) -> None:
    try:
        with PIPELINE_LOG_TXT.open("a", encoding="utf-8") as log_file:
            log_file.write(message + "\n")
    except Exception:
        pass
    with _bot_log_lock:
        _bot_log_buffer.append(message)


def _persist_bot_status(status: dict) -> None:
    with _bot_status_lock:
        temp_path = BOT_STATUS_JSON.with_name(
            f".{BOT_STATUS_JSON.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        try:
            temp_path.write_text(json.dumps(status, ensure_ascii=False), encoding="utf-8")
            os.replace(temp_path, BOT_STATUS_JSON)
        finally:
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except Exception:
                pass


def _get_bot_status() -> dict:
    with _bot_status_lock:
        data = _read_json_safe(BOT_STATUS_JSON)
    return data if isinstance(data, dict) else {}


def _set_bot_status(**updates) -> dict:
    state = _get_bot_status()
    state.update(updates)
    _persist_bot_status(state)
    return state


def _mark_bot_stopped(returncode: int | None = None, *, requested: bool = False) -> dict:
    global _bot_process
    now = _utc_now_iso()
    if not isinstance(returncode, int):
        returncode = None
    status = "stopped" if requested or returncode == 0 else "crashed"
    reason = "parada solicitada" if requested else (
        f"processo encerrado com returncode {returncode}" if returncode is not None else "processo ausente"
    )
    state = _set_bot_status(
        status=status,
        running=False,
        stopped_at=now,
        finished_at=now,
        returncode=returncode,
        last_error="" if status == "stopped" else reason,
    )
    _clear_stale_bot_pid(state.get("pid"))
    _bot_process = None
    if status == "crashed":
        _append_bot_log(f"[BOT] Processo encerrado inesperadamente: {reason}")
    else:
        _append_bot_log(f"[BOT] Processo encerrado: {reason}")
    return state


def _refresh_bot_status(running: bool | None = None, pid: int | None = None) -> dict:
    if running is None:
        running, pid = _bot_running()
    state = _get_bot_status()
    status = str(state.get("status", "idle"))
    known_pid = state.get("pid")
    try:
        known_pid_int = int(known_pid or 0)
    except (TypeError, ValueError):
        known_pid_int = 0
    should_mark_crashed = (
        (status in {"starting", "running"} or bool(state.get("running")))
        and not running
        and known_pid_int
        and (pid is None or known_pid_int != int(pid or 0))
    )
    if should_mark_crashed:
        return _mark_bot_stopped(state.get("returncode"), requested=False)
    if running:
        return _set_bot_status(status="running", running=True, pid=pid, last_seen_at=_utc_now_iso())
    if state.get("running") and status != "stopping":
        return _mark_bot_stopped(state.get("returncode"), requested=False)
    return state


def _persist_backtest_state(state: dict) -> None:
    """Persiste o estado da execução com substituição atômica."""
    BACKTEST_ROOT.mkdir(parents=True, exist_ok=True)
    temp_path = BACKTEST_STATE_JSON.with_suffix(".tmp")
    temp_path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    os.replace(temp_path, BACKTEST_STATE_JSON)


def _get_backtest_state() -> dict:
    global _backtest_state
    with _backtest_lock:
        if not _backtest_state and BACKTEST_STATE_JSON.exists():
            try:
                loaded = json.loads(BACKTEST_STATE_JSON.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    _backtest_state = loaded
            except (OSError, json.JSONDecodeError):
                pass
        return dict(_backtest_state)


def _set_backtest_state(**updates) -> dict:
    global _backtest_state
    with _backtest_lock:
        current = _get_backtest_state()
        current.update(updates)
        _backtest_state = current
        _persist_backtest_state(current)
        return dict(current)


def _backtest_running() -> tuple[bool, int | None]:
    with _backtest_lock:
        if _backtest_process is not None and _backtest_process.poll() is None:
            return True, _backtest_process.pid
        state = _get_backtest_state()
    pid = state.get("pid")
    # Após reinício do servidor, não confia apenas em um PID persistido: ele
    # pode já ter sido reutilizado por outro processo. O status será marcado
    # como interrompido, sem risco de terminar um processo alheio.
    return False, int(pid) if isinstance(pid, int) else None


def _read_backtest_output(proc: subprocess.Popen, run_id: str, run_dir: Path) -> None:
    """Captura logs e finaliza o estado persistido do backtest."""
    log_path = run_dir / "backtest.log"
    try:
        stdout = proc.stdout
        with log_path.open("a", encoding="utf-8") as log_file:
            if stdout is not None:
                for line in stdout:
                    decoded = line.rstrip()
                    if not decoded:
                        continue
                    log_file.write(decoded + "\n")
                    log_file.flush()
                    with _backtest_lock:
                        _backtest_log_buffer.append(decoded)
                stdout.close()
        returncode = proc.wait()
    except Exception as exc:
        returncode = proc.poll()
        with _backtest_lock:
            _backtest_log_buffer.append(f"Falha ao ler processo: {exc}")

    report_path = run_dir / "backtest_report.json"
    current = _get_backtest_state()
    if current.get("run_id") != run_id:
        return
    requested_stop = current.get("status") == "stopping"
    completed = returncode == 0 and report_path.exists() and not requested_stop
    error = ""
    if not completed and not requested_stop:
        error = f"Backtest finalizado com código {returncode}"
    try:
        _set_backtest_state(
            status="completed" if completed else ("stopped" if requested_stop else "failed"),
            running=False,
            returncode=returncode,
            finished_at=dt.datetime.now(dt.timezone.utc).isoformat(),
            report_available=report_path.exists(),
            error=error,
        )
    except OSError:
        pass


def _validated_number(
    data: dict,
    key: str,
    default: float,
    minimum: float,
    maximum: float,
    integer: bool = False,
):
    try:
        value = int(data.get(key, default)) if integer else float(data.get(key, default))
    except (TypeError, ValueError):
        raise ValueError(f"{key} deve ser numérico") from None
    if not minimum <= value <= maximum:
        raise ValueError(f"{key} deve estar entre {minimum} e {maximum}")
    return value


def _safe_report_path(state: dict) -> Path | None:
    raw = state.get("report_path")
    if not raw:
        return None
    try:
        path = Path(raw).resolve()
        root = BACKTEST_ROOT.resolve()
        if not path.is_relative_to(root):
            return None
        return path
    except (OSError, TypeError, ValueError):
        return None


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
    risk_state = _read_json_safe(RISK_STATE_JSON)
    summary = _summary_from_contract_report(_deriv_contract_report(limit=500), risk_state)
    summary["risk_state"] = risk_state
    summary["source"] = "deriv_profit_table"
    if _deriv_contract_cache.get("error"):
        summary["source_error"] = _deriv_contract_cache["error"]
    return jsonify(summary)


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
    force = str(request.args.get("force", "")).lower() in {"1", "true", "yes"}
    records = _deriv_contract_report(limit=n, force=force)[:n]
    return jsonify({
        "source": "deriv_profit_table",
        "error": str(_deriv_contract_cache.get("error", "")),
        "trades": records,
    })


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
    status = _refresh_bot_status(running, pid)

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
        "status":     status.get("status", "running" if running else "idle"),
        "started_at": status.get("started_at"),
        "stopped_at": status.get("stopped_at"),
        "last_tick_at": status.get("last_tick_at"),
        "last_loop_at": status.get("last_loop_at"),
        "last_error": status.get("last_error", ""),
        "returncode": status.get("returncode"),
        "last_decision": status.get("last_decision", {}),
        "decision_counts": status.get("decision_counts", {}),
        "no_entry_for_sec": status.get("no_entry_for_sec", 0),
    })


# ─── API: Iniciar bot ─────────────────────────────────────────────────────────

@app.route("/api/bot/start", methods=["POST"])
@_require_token
def api_bot_start():
    running, pid = _bot_running()
    if running:
        return jsonify({"ok": False, "msg": f"Bot já está rodando (PID {pid})."}), 409
    backtest_running, backtest_pid = _backtest_running()
    if backtest_running:
        return jsonify({
            "ok": False,
            "msg": f"Backtest em execução (PID {backtest_pid}). Aguarde ou cancele antes de iniciar o bot.",
        }), 409

    data = request.get_json(silent=True) or {}
    mode = data.get("mode", "demo")
    if mode not in ("demo", "real"):
        return jsonify({"ok": False, "msg": "mode deve ser 'demo' ou 'real'"}), 400
    try:
        balance = float(data.get("balance", 1000.0))
        if not _math.isfinite(balance) or balance <= 0:
            raise ValueError()
    except (TypeError, ValueError):
        return jsonify({"ok": False, "msg": "balance deve ser um número positivo"}), 400
    skip_collect = bool(data.get("skip_collect", False))
    no_scan = bool(data.get("no_scan", False))
    min_ticks = int(data.get("min_ticks", 200))
    history_count = int(data.get("history_count", MIN_HISTORY_CANDLES))
    force_retrain = bool(data.get("force_retrain", False))
    use_trained_model = bool(data.get("use_trained_model", True))
    use_weighted_signal = bool(data.get("use_weighted_signal", True))
    use_price_action_filter = bool(data.get("use_price_action_filter", False))
    use_candle_pattern_filter = bool(data.get("use_candle_pattern_filter", False))
    if not any((
        use_trained_model,
        use_weighted_signal,
        use_price_action_filter,
        use_candle_pattern_filter,
    )):
        return jsonify({
            "ok": False,
            "msg": "Ative pelo menos um módulo de decisão.",
        }), 400
    execution_signal_mode = "trained_model"

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

    # Reiniciar processo não redefine o risco diário nem apaga contratos pendentes.
    # Registra timestamp de início da sessão em state.json
    try:
        import datetime as _dt2
        _cur_state = _read_json_safe(STATE_JSON)
        _cur_state["session_start"] = _dt2.datetime.now().isoformat()
        _cur_state["min_ticks_target"] = int(min_ticks_target)
        STATE_JSON.write_text(json.dumps(_cur_state))
    except Exception:
        pass

    try:
        global _bot_process
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["STANDERIV_USE_AI_MODEL"] = "1" if use_trained_model else "0"
        env["STANDERIV_EXECUTION_SIGNAL_MODE"] = execution_signal_mode
        env["STANDERIV_USE_WEIGHTED_SIGNAL"] = "1" if use_weighted_signal else "0"
        env["STANDERIV_PRICE_ACTION_FILTER"] = "1" if use_price_action_filter else "0"
        env["STANDERIV_CANDLE_PATTERN_FILTER"] = "1" if use_candle_pattern_filter else "0"
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
        _set_bot_status(
            status="starting",
            running=True,
            pid=proc.pid,
            mode=mode,
            started_at=_utc_now_iso(),
            stopped_at=None,
            finished_at=None,
            returncode=None,
            last_error="",
            command=cmd,
            execution_signal_mode=execution_signal_mode,
            use_trained_model=use_trained_model,
            use_weighted_signal=use_weighted_signal,
            use_price_action_filter=use_price_action_filter,
            use_candle_pattern_filter=use_candle_pattern_filter,
        )
        threading.Thread(target=_read_bot_output, args=(proc,), daemon=True).start()
        PID_FILE.write_text(str(proc.pid))
        return jsonify({
            "ok": True,
            "pid": proc.pid,
            "mode": mode,
            "execution_signal_mode": execution_signal_mode,
            "use_trained_model": use_trained_model,
            "use_weighted_signal": use_weighted_signal,
            "use_price_action_filter": use_price_action_filter,
            "use_candle_pattern_filter": use_candle_pattern_filter,
        })
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

        _set_bot_status(
            status="stopping",
            running=False,
            pid=pid,
            stopped_at=_utc_now_iso(),
            last_error="",
        )

        # O encerramento do processo não liquida o contrato na corretora.
        return jsonify({"ok": True, "pid": pid})
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)}), 500


def _reset_operations_and_state():
    """Reseta apenas a visualização; ledger, PnL diário e contrato são preservados."""
    if _bot_running()[0] or _read_json_safe(STATE_JSON).get('active_contract'):
        return False, 'Contrato pendente ou bot ativo; não é possível limpar a sessão'
    from model_contract import write_json_atomic
    state=_read_json_safe(STATE_JSON)
    state['session_start']=dt.datetime.now().isoformat()
    write_json_atomic(STATE_JSON,state)
    return True, ''


@app.route("/api/bot/data/clear", methods=["POST"])
@_require_token
def api_bot_data_clear():
    """Limpa os dados das operações (operacoes_log.csv, risk_state.json, state.json).

    Protegido por token via `_require_token`.
    """
    try:
        ok, msg = _reset_operations_and_state()
        if not ok:
            return jsonify({"ok": False, "msg": msg}), 409
        return jsonify({"ok": True, "msg": "Visualização reiniciada; histórico e risco diário preservados"})
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


# ─── API: Backtesting offline ────────────────────────────────────────────────

@app.route("/api/backtest/status")
def api_backtest_status():
    running, pid = _backtest_running()
    state = _get_backtest_state()
    with _backtest_lock:
        process_known = _backtest_process is not None
    if (
        state.get("status") in {"running", "stopping"}
        and not running
        and not process_known
    ):
        state = _set_backtest_state(
            status="failed",
            running=False,
            finished_at=dt.datetime.now(dt.timezone.utc).isoformat(),
            error="Processo interrompido antes de concluir",
        )
    with _backtest_lock:
        logs = list(_backtest_log_buffer)[-50:]
    return jsonify({
        "status": state.get("status", "idle"),
        "running": running,
        "pid": pid,
        "run_id": state.get("run_id"),
        "started_at": state.get("started_at"),
        "finished_at": state.get("finished_at"),
        "params": state.get("params", {}),
        "report_available": bool(state.get("report_available", False)),
        "error": state.get("error", ""),
        "logs": logs,
    })


@app.route("/api/backtest/start", methods=["POST"])
@_require_token
def api_backtest_start():
    global _backtest_process

    bot_is_running, _ = _bot_running()
    if bot_is_running:
        return jsonify({
            "ok": False,
            "msg": "Pare o bot antes de executar o backtest.",
        }), 409
    running, pid = _backtest_running()
    if running:
        return jsonify({
            "ok": False,
            "msg": f"Backtest já está em execução (PID {pid}).",
        }), 409
    if not BACKTEST_PY.exists() or not TICKS_CSV.exists():
        return jsonify({
            "ok": False,
            "msg": "backtest.py ou ticks.csv não encontrado.",
        }), 404

    data = request.get_json(silent=True) or {}
    try:
        symbol = str(data.get("symbol", "")).strip()
        if symbol and not re.fullmatch(r"[A-Za-z0-9_.-]{1,32}", symbol):
            raise ValueError("symbol contém caracteres inválidos")
        balance = _validated_number(data, "balance", 1000.0, 0.35, 1_000_000_000)
        payout = _validated_number(data, "payout", 0.85, 0.0, 10.0)
        cost_rate = _validated_number(data, "cost_rate", 0.0, 0.0, 1.0)
        fixed_cost = _validated_number(data, "fixed_cost", 0.0, 0.0, 1_000_000)
        duration = _validated_number(data, "duration", 15, 1, 1440, integer=True)
        timeframe = _validated_number(data, "timeframe", 60, 1, 86400, integer=True)
        max_ticks = _validated_number(
            data,
            "max_ticks",
            100_000,
            500,
            2_000_000,
            integer=True,
        )
        duration_unit = str(data.get("duration_unit", "m"))
        if duration_unit not in {"m", "s", "t"}:
            raise ValueError("duration_unit deve ser 'm', 's' ou 't'")
        for key in ("with_ai", "dynamic_duration", "candle_patterns"):
            if key in data and not isinstance(data[key], bool):
                raise ValueError(f"{key} deve ser booleano")
    except ValueError as exc:
        return jsonify({"ok": False, "msg": str(exc)}), 400

    params = {
        "symbol": symbol,
        "balance": balance,
        "payout": payout,
        "cost_rate": cost_rate,
        "fixed_cost": fixed_cost,
        "duration": duration,
        "duration_unit": duration_unit,
        "timeframe": timeframe,
        "max_ticks": max_ticks,
        "with_ai": bool(data.get("with_ai", False)),
        "dynamic_duration": bool(data.get("dynamic_duration", False)),
        "candle_patterns": bool(data.get("candle_patterns", True)),
    }
    run_id = (
        dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + uuid.uuid4().hex[:8]
    )
    run_dir = BACKTEST_ROOT / run_id
    report_path = run_dir / "backtest_report.json"
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        return jsonify({"ok": False, "msg": str(exc)}), 500

    cmd = [
        sys.executable,
        "-u",
        str(BACKTEST_PY),
        "--input",
        str(TICKS_CSV),
        "--output-dir",
        str(run_dir),
        "--balance",
        str(balance),
        "--payout",
        str(payout),
        "--cost-rate",
        str(cost_rate),
        "--fixed-cost",
        str(fixed_cost),
        "--duration",
        str(duration),
        "--duration-unit",
        duration_unit,
        "--timeframe",
        str(timeframe),
        "--max-ticks",
        str(max_ticks),
    ]
    if symbol:
        cmd += ["--symbol", symbol]
    if params["with_ai"]:
        cmd.append("--with-ai")
    if params["dynamic_duration"]:
        cmd.append("--dynamic-duration")
    if not params["candle_patterns"]:
        cmd.append("--no-candle-patterns")

    proc = None
    try:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env.pop("DASHBOARD_TOKEN", None)
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
        _backtest_process = proc
        with _backtest_lock:
            _backtest_log_buffer.clear()
        _set_backtest_state(
            run_id=run_id,
            status="running",
            running=True,
            pid=proc.pid,
            started_at=dt.datetime.now(dt.timezone.utc).isoformat(),
            finished_at=None,
            params=params,
            report_path=str(report_path),
            report_available=False,
            returncode=None,
            error="",
        )
        threading.Thread(
            target=_read_backtest_output,
            args=(proc, run_id, run_dir),
            daemon=True,
            name=f"Backtest-{run_id}",
        ).start()
        return jsonify({"ok": True, "run_id": run_id, "pid": proc.pid}), 202
    except Exception as exc:
        if proc is not None and proc.poll() is None:
            proc.terminate()
        try:
            _set_backtest_state(
                run_id=run_id,
                status="failed",
                running=False,
                finished_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                params=params,
                report_path=str(report_path),
                report_available=False,
                error=str(exc),
            )
        except OSError:
            pass
        return jsonify({"ok": False, "msg": str(exc)}), 500


@app.route("/api/backtest/stop", methods=["POST"])
@_require_token
def api_backtest_stop():
    running, pid = _backtest_running()
    if not running or pid is None:
        return jsonify({"ok": False, "msg": "Backtest não está em execução."}), 409
    try:
        _set_backtest_state(status="stopping", running=True)
        if _PSUTIL:
            proc = psutil.Process(pid)
            for child in proc.children(recursive=True):
                child.terminate()
            proc.terminate()
        else:
            os.kill(pid, signal.SIGTERM)
        return jsonify({"ok": True, "pid": pid})
    except Exception as exc:
        return jsonify({"ok": False, "msg": str(exc)}), 500


@app.route("/api/backtest/result")
def api_backtest_result():
    state = _get_backtest_state()
    report_path = _safe_report_path(state)
    if report_path is None or not report_path.exists():
        return jsonify({"ok": False, "msg": "Nenhum relatório disponível."}), 404
    try:
        try:
            n = min(max(int(request.args.get("n", 200)), 1), 1000)
        except (TypeError, ValueError):
            return jsonify({"ok": False, "msg": "n deve ser inteiro"}), 400
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            raise ValueError("formato de relatório inválido")
        trades = report.pop("trades", [])
        if not isinstance(trades, list):
            trades = []
        initial_balance = float(report.get("metrics", {}).get("initial_balance", 0.0))
        equity_curve = []
        if trades:
            equity_curve.append({
                "epoch": int(trades[0].get("entry_epoch", 0)),
                "balance": initial_balance,
            })
            equity_curve.extend({
                "epoch": int(trade.get("expiry_epoch", 0)),
                "balance": float(trade.get("balance_after", 0.0)),
            } for trade in trades)
        if len(equity_curve) > 500:
            step = _math.ceil(len(equity_curve) / 500)
            sampled = equity_curve[::step]
            if sampled[-1] != equity_curve[-1]:
                sampled.append(equity_curve[-1])
            equity_curve = sampled
        report["trades"] = list(reversed(trades[-n:]))
        report["equity_curve"] = equity_curve
        return jsonify({"ok": True, "report": report})
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return jsonify({"ok": False, "msg": str(exc)}), 500


# ─── API: Equity curve ────────────────────────────────────────────────────────

@app.route("/api/equity")
def api_equity():
    records = list(reversed(_deriv_contract_report(limit=500)))
    if not records:
        return jsonify([])
    risk_state = _read_json_safe(RISK_STATE_JSON)
    current_balance = float(risk_state.get("balance", 0.0) or 0.0)
    total_profit = sum(float(record.get("profit", 0.0) or 0.0) for record in records)
    running_balance = current_balance - total_profit if current_balance else 0.0
    result = []
    for record in records:
        running_balance += float(record.get("profit", 0.0) or 0.0)
        result.append({
            "timestamp": str(record.get("timestamp", "")),
            "balance_after": round(running_balance, 2),
            "result": str(record.get("result", "")),
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


def _summary_from_risk_state(risk_state: dict) -> dict:
    """Monta resumo operacional quando ainda não há operações na sessão filtrada."""
    recent_results = risk_state.get("recent_results", [])
    if not isinstance(recent_results, list):
        recent_results = []
    total = len(recent_results)
    wins = sum(1 for result in recent_results if int(result or 0) == 1)
    daily_start = float(risk_state.get("daily_start_balance", risk_state.get("initial_balance", 0.0)) or 0.0)
    balance = float(risk_state.get("balance", daily_start) or 0.0)
    daily_pnl = float(risk_state.get("daily_pnl", 0.0) or 0.0)
    daily_pnl_pct = float(risk_state.get("daily_pnl_pct", 0.0) or 0.0)
    consec_losses = int(risk_state.get("consec_losses", 0) or 0)
    pause_remaining = int(risk_state.get("pause_remaining_sec", 0) or 0)

    return {
        "balance": round(balance, 2),
        "balance_initial": round(daily_start, 2),
        "pnl_today": round(daily_pnl, 2),
        "pnl_pct": round(daily_pnl_pct, 2),
        "win_rate": round(wins / total * 100, 1) if total else 0.0,
        "win_rate_recent": round(float(risk_state.get("win_rate_recent", 0.0) or 0.0), 1),
        "total_trades": total,
        "wins": wins,
        "losses": total - wins,
        "consec_losses": consec_losses,
        "is_paused": bool(risk_state.get("is_paused", False)) or pause_remaining > 0 or consec_losses >= 3,
        "drawdown_pct": round(float(risk_state.get("drawdown_pct", 0.0) or 0.0), 2),
    }


def _summary_from_contract_report(records: list[dict], risk_state: dict) -> dict:
    balance = float(risk_state.get("balance", 0.0) or 0.0)
    daily_start = float(risk_state.get("daily_start_balance", risk_state.get("initial_balance", balance)) or 0.0)
    today = dt.datetime.now().date()
    today_records = []
    for record in records:
        try:
            if dt.datetime.fromisoformat(str(record.get("timestamp", ""))).date() == today:
                today_records.append(record)
        except Exception:
            pass
    scoped = today_records or records
    total = len(scoped)
    wins = sum(1 for record in scoped if record.get("result") == "WIN")
    pnl_today = sum(float(record.get("profit", 0.0) or 0.0) for record in today_records)
    if not records:
        return _summary_from_risk_state(risk_state)
    return {
        "balance": round(balance, 2),
        "balance_initial": round(daily_start, 2),
        "pnl_today": round(pnl_today, 2),
        "pnl_pct": round((pnl_today / daily_start * 100), 2) if daily_start else 0.0,
        "win_rate": round(wins / total * 100, 1) if total else 0.0,
        "win_rate_recent": round(wins / total * 100, 1) if total else 0.0,
        "total_trades": total,
        "wins": wins,
        "losses": sum(1 for record in scoped if record.get("result") == "LOSS"),
        "consec_losses": int(risk_state.get("consec_losses", 0) or 0),
        "is_paused": bool(risk_state.get("is_paused", False)) or int(risk_state.get("pause_remaining_sec", 0) or 0) > 0,
        "drawdown_pct": round(float(risk_state.get("drawdown_pct", 0.0) or 0.0), 2),
    }


# ─── API: Estado consolidado (/api/state) ─────────────────────────────────────

@app.route("/api/state")
def api_state():
    """Endpoint único que retorna todos os dados do dashboard em um só JSON."""
    running, pid = _bot_running()
    bot_status = _refresh_bot_status(running, pid)

    # — Resumo de operações vindo do relatório oficial de contratos da Deriv —
    contract_records = _deriv_contract_report(limit=500)
    df_ops_all = pd.DataFrame(contract_records)
    active_state_pre = _read_json_safe(STATE_JSON)
    risk_state = _read_json_safe(RISK_STATE_JSON)
    session_start_str = active_state_pre.get("session_start", "")

    df_ops = df_ops_all.copy()
    if session_start_str and not df_ops.empty and "timestamp" in df_ops.columns:
        try:
            _ss = pd.to_datetime(session_start_str, errors="coerce")
            df_ops["_ts"] = pd.to_datetime(df_ops["timestamp"], errors="coerce")
            df_ops = df_ops[df_ops["_ts"] >= _ss].reset_index(drop=True)
        except Exception:
            pass
    summary = _summary_from_contract_report(df_ops.to_dict("records"), risk_state)
    last_trade_indicators = {}
    if not df_ops.empty:
        last = df_ops.iloc[-1]
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

    # Métrica ativa precisa pertencer ao bundle publicado; candidatos ficam separados.
    try:
        from model_contract import fingerprint, usable
        manifest=_read_json_safe(Path(str(MODEL_PKL)+'.metadata.json'))
        symbol=active_state_pre.get('symbol','')
        valid=(MODEL_PKL.exists() and isinstance(manifest,dict) and manifest.get('approved')
               and usable(manifest,symbol) and manifest.get('artifact_sha256')==fingerprint(MODEL_PKL))
        model_info['ready']=bool(valid)
        model_info['last_metrics']=manifest.get('metrics',{}) if valid else {}
        model_info['blocked_reason']='' if valid else 'Modelo sem validação compatível com o ativo e vencimento'
        candidates=_read_json_safe(MODEL_METRICS_JSON)
        model_info['last_candidate_metrics']=candidates[-1] if isinstance(candidates,list) and candidates else {}
    except Exception:
        model_info.update(ready=False,last_metrics={},blocked_reason='Modelo não validado')

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
        "bot": {
            "running": running,
            "pid": pid,
            "uptime_sec": uptime_sec,
            "status": bot_status.get("status", "running" if running else "idle"),
            "started_at": bot_status.get("started_at"),
            "stopped_at": bot_status.get("stopped_at"),
            "last_tick_at": bot_status.get("last_tick_at"),
            "last_loop_at": bot_status.get("last_loop_at"),
            "last_error": bot_status.get("last_error", ""),
            "returncode": bot_status.get("returncode"),
            "last_decision": bot_status.get("last_decision", {}),
            "decision_counts": bot_status.get("decision_counts", {}),
            "no_entry_for_sec": bot_status.get("no_entry_for_sec", 0),
        },
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
        return jsonify({"has_active": False, "last_contract": _last_local_contract()})

    buy_timestamp = float(ac.get("buy_timestamp", 0.0) or 0.0)
    entry_epoch = int(ac.get("entry_epoch", 0) or 0)
    if entry_epoch <= 0 and buy_timestamp > 0:
        entry_epoch = int(buy_timestamp)

    payload = {
        "has_active": True,
        "phase": str(ac.get("phase", "")),
        "contract_id": str(ac.get("contract_id", "")),
        "symbol": str(ac.get("symbol", state.get("symbol", ""))),
        "direction": str(ac.get("direction", "")),
        "duration": int(ac.get("duration", 0) or 0),
        "buy_timestamp": buy_timestamp,
        "entry_epoch": entry_epoch,
        "entry_price": float(ac.get("entry_price", 0.0) or 0.0),
        "last_contract": _last_local_contract(),
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
    # Apenas métricas do artefato aprovado; histórico de candidatos tem rota própria.
    entry = {}
    try:
        from model_contract import fingerprint, usable
        manifest = _read_json_safe(Path(str(MODEL_PKL) + '.metadata.json'))
        state = _read_json_safe(STATE_JSON)
        symbol = state.get('symbol', '') if isinstance(state, dict) else ''
        if (MODEL_PKL.exists() and isinstance(manifest, dict)
                and usable(manifest, symbol)
                and manifest.get('artifact_sha256') == fingerprint(MODEL_PKL)):
            entry = manifest.get('metrics', {})
    except Exception:
        entry = {}
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
    df = _contracts_frame(limit=500)
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
