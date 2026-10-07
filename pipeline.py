"""Orquestrador do pipeline: coleta, dataset, treino, bot e retreino."""
from pathlib import Path
import math
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
    DURATION,
    DURATION_UNIT,
    MODEL_PROMOTION_MIN_AUC_DELTA,
    MODEL_PROMOTION_MAX_ACC_DROP,
    MODEL_PROMOTION_MAX_F1_DROP,
    DEMO_MODE,
    CANDLE_TIMEFRAME_SEC,
    TICK_SPIKE_THRESHOLD,
    RETRAIN_EVAL_WINDOW,
    RETRAIN_WIN_RATE_TRIGGER,
    RETRAIN_CONFIDENCE_TRIGGER,
    RETRAIN_NEW_LABELS_TRIGGER,
    RETRAIN_MIN_INTERVAL_SEC,
    RETRAIN_MAX_INTERVAL_SEC,
    OPERATIONS_LOG,
    BOOTSTRAP_MAX_FAILED_PROMOTIONS,
    MARKET_SWITCH_COOLDOWN_SEC,
    MARKET_SWITCH_MAX_PER_WINDOW,
    MARKET_SWITCH_WINDOW_SEC,
)

from executor import DerivBot
from risk_manager import RiskManager
from market_data import ensure_header, append_record, read_records
from model_contract import (FEATURE_WINDOW, MAX_TICK_GAP_SEC, usable,
                            load_validated_bundle, write_json_atomic,
                            acquire_execution_lock, minimum_history_candles,
                            minimum_training_rows)


# ─────────────────────────────────────────────────────────────────
# Defaults do pipeline
# ─────────────────────────────────────────────────────────────────

_DEFAULT_HISTORY_COUNT = minimum_history_candles()
_DEFAULT_MIN_TICKS = 100
_MIN_LABELABLE_CONTRACTS = minimum_training_rows()
_TICK_LOG_INTERVAL = 200
_HISTORY_LOG_INTERVAL = 10_000

_WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}"

_AUTH_MODE = "legacy"
_PAT_ACCOUNT_ID: Optional[str] = None
_PAT_ACCOUNT: dict = {}
_PAT_WS_URL: Optional[str] = None
_BOT_STATUS_JSON = Path(__file__).resolve().parent / "bot_status.json"
_market_failure_until: dict[str, float] = {}
_market_switch_times: list[float] = []


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


def _runtime_symbol() -> str:
    """Símbolo efetivamente selecionado para o processo atual."""
    import config as _cfg
    return str(globals().get("SYMBOL") or _cfg.get_active_symbol() or _cfg.SYMBOL)


def _set_runtime_symbol(symbol):
    """Sincroniza consumidores mesmo quando o ativo escolhido é o default."""
    import config as cfg
    import executor
    import strategy
    for module in (cfg, executor, strategy, dataset_builder, train_model):
        if hasattr(module, 'SYMBOL'):
            module.SYMBOL = symbol
    cfg.set_active_symbol(symbol)
    globals()['SYMBOL'] = symbol


# Evento global: sinaliza encerramento a todas as threads
_shutdown = threading.Event()
_market_switch = threading.Event()


class MarketUnavailable(Exception):
    """Sinaliza que o mercado ativo deve ser trocado."""
    pass


def _request_market_switch(symbol):
    """Pede troca de mercado se `symbol` for o ativo atual (contrato indisponível)."""
    if symbol == _runtime_symbol():
        if not _market_switch.is_set():
            print(f'\n[MERCADO] {symbol}: contrato indisponível; buscando outro ativo após concluir pendências.')
        _market_switch.set()


def _request_market_switch_for_unfavorable(symbol, reason, streak, indicators=None):
    """Pede troca de mercado após sequência de avaliações desfavoráveis do ativo atual."""
    if symbol == _runtime_symbol():
        if not _market_switch.is_set():
            adx = ""
            if indicators and indicators.get("adx") is not None:
                adx = f" | ADX={indicators.get('adx')}"
            print(
                f"\n[MERCADO] {symbol}: {streak} avaliações sem condição favorável "
                f"({reason}{adx}); buscando outro ativo após concluir pendências."
            )
        _market_switch.set()


def _check_market_switch():
    """Levanta `MarketUnavailable` se houver troca de mercado solicitada."""
    if _market_switch.is_set():
        raise MarketUnavailable(_runtime_symbol())


_pipeline_status_lock = threading.Lock()


def _write_pipeline_status(**updates):
    """Mescla `updates` em `bot_status.json` com escrita atômica protegida por lock."""
    import tempfile

    with _pipeline_status_lock:
        try:
            status = json.loads(_BOT_STATUS_JSON.read_text()) if _BOT_STATUS_JSON.exists() else {}
        except Exception:
            status = {}
        status.update(updates)
        status.setdefault("pid", os.getpid())
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=_BOT_STATUS_JSON.parent,
                prefix=f".{_BOT_STATUS_JSON.name}.{os.getpid()}.",
                suffix=".tmp",
                delete=False,
            ) as temp_file:
                temp_path = Path(temp_file.name)
                temp_file.write(json.dumps(status, ensure_ascii=False))
            os.replace(temp_path, _BOT_STATUS_JSON)
        finally:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass


def _pipeline_heartbeat(stage: str, **updates):
    """Publica no status o estágio atual, o horário e o símbolo."""
    _write_pipeline_status(
        status=stage,
        running=True,
        last_seen_at=datetime.utcnow().isoformat() + "+00:00",
        symbol=_runtime_symbol(),
        **updates,
    )


def _mark_market_failure(symbol: str, reason: str = "") -> None:
    """Coloca o símbolo em quarentena, registra a troca na janela de orçamento e atualiza o status."""
    if not symbol:
        return
    _market_failure_until[str(symbol)] = time.time() + max(0, int(MARKET_SWITCH_COOLDOWN_SEC or 0))
    _market_switch_times.append(time.time())
    cutoff = time.time() - max(1, int(MARKET_SWITCH_WINDOW_SEC or 1))
    del _market_switch_times[: len([t for t in _market_switch_times if t < cutoff])]
    if reason:
        _pipeline_heartbeat("switching_market", last_error=reason)


def _market_available_for_switch(symbol: str) -> bool:
    """Indica se a quarentena do símbolo já terminou."""
    until = float(_market_failure_until.get(str(symbol), 0.0) or 0.0)
    return until <= time.time()


def _switch_budget_exhausted() -> bool:
    """Indica se o limite de trocas de mercado na janela recente foi atingido."""
    cutoff = time.time() - max(1, int(MARKET_SWITCH_WINDOW_SEC or 1))
    recent = [t for t in _market_switch_times if t >= cutoff]
    _market_switch_times[:] = recent
    limit = int(MARKET_SWITCH_MAX_PER_WINDOW or 0)
    return limit > 0 and len(recent) >= limit


# ─────────────────────────────────────────────────────────────────
# Utilitários
# ─────────────────────────────────────────────────────────────────

def _count_records(kinds=('tick',)):
    """Conta registros identificados do ativo ativo para os tipos informados."""
    if not os.path.exists(TICKS_CSV):
        return 0
    symbol = _runtime_symbol()
    with open(TICKS_CSV,newline='') as f:
        allowed = set(kinds)
        return sum(row.get('symbol')==symbol and row.get('kind') in allowed
                   for row in csv.DictReader(f))


def _count_ticks():
    """Contagem de ticks reais do ativo; candles históricos não rotulam contratos."""
    return _count_records(('tick',))


def _count_market_records():
    """Contagem para status operacional: ticks reais e candles históricos fechados."""
    return _count_records(('tick', 'candle'))


def _count_candles():
    """Contagem de candles históricos do ativo."""
    return _count_records(('candle',))


def _minimum_history_candles():
    """Reserva features, vencimento, embargo e quatro particoes temporais."""
    return minimum_history_candles()


def _effective_history_count(requested):
    """Quantidade de candles a baixar: 0 se inválida/não positiva, senão ao menos o mínimo necessário."""
    try:
        requested = int(requested)
    except (TypeError, ValueError):
        requested = 0
    if requested <= 0:
        return 0
    return max(requested, _minimum_history_candles())


def _history_fetch_count_for_start(requested):
    """Na inicialização, sempre atualiza os candles mais recentes do símbolo ativo."""
    return _effective_history_count(requested)


def _ensure_ticks_header():
    """Garante o cabeçalho do CSV de ticks."""
    ensure_header(TICKS_CSV)


def _prepare_training_start(*, reuse=False, skip_collect=False, pending=None):
    """Archive previous learning artifacts while the caller owns executor.lock."""
    if pending or reuse or skip_collect:
        print('[PIPELINE] Dados preservados: reconciliacao pendente ou reutilizacao solicitada.')
        return None
    import tempfile
    import config

    bases = [TICKS_CSV, DATASET_CSV, AI_MODEL_PATH, config.LEGACY_TICKS_CSV,
             'dataset.csv', config.TRANSFORMER_MODEL_PATH, config.DURATION_MODEL_PATH,
             'model_metrics.json']
    paths = set()
    for base in bases:
        for suffix in ('', '.metadata.json', '.new', '.new.metadata.json',
                       '.candidate', '.candidate.metadata.json', '.tmp', '.train_snapshot'):
            path = Path(str(base) + suffix).absolute()
            if path.is_file():
                paths.add(path)
    if not paths:
        print('[PIPELINE] Nova coleta: nenhum dado ou modelo anterior para reutilizar.')
        return None
    archive_root = Path(TICKS_CSV).absolute().parent / 'recovered_data' / 'training_resets'
    archive_root.mkdir(parents=True, exist_ok=True)
    archive = Path(tempfile.mkdtemp(prefix=datetime.now().strftime('%Y%m%d-%H%M%S-'),
                                    dir=archive_root))
    moved = []
    try:
        for index, source in enumerate(sorted(paths)):
            destination = archive / f'{index:02d}-{source.name}'
            os.replace(source, destination)
            moved.append((source, destination))
        write_json_atomic(archive / 'manifest.json',
                          [{'source': str(src), 'backup': dst.name} for src, dst in moved])
    except Exception:
        for source, destination in reversed(moved):
            os.replace(destination, source)
        raise
    _reset_ai_predictor()
    print(f'[PIPELINE] Nova coleta e novo treino. {len(moved)} arquivos antigos arquivados em {archive}')
    return archive


def _uses_bearer_auth() -> bool:
    """Indica se o pipeline usa autenticação Bearer com conta definida."""
    return (
        _AUTH_MODE == "bearer"
        and bool(_PAT_ACCOUNT_ID)
    )


def _get_ws_url() -> str:
    """URL do WebSocket: a fixa legada ou, em Bearer, a pré-obtida (consumida) ou nova solicitada à API."""
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
_TOP_TRENDING_SYMBOLS = 5


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
        """Solicita as últimas `count` velas do símbolo."""
        ws.send(
            json.dumps(
                {
                    "ticks_history": symbol,
                    "end": "latest",
                    "count": count,
                    "style": "candles",
                    "granularity": CANDLE_TIMEFRAME_SEC,
                    "adjust_start_time": 1,
                }
            )
        )

    def _on_open(ws) -> None:
        """Pede o histórico (Bearer) ou envia `authorize` (modo legado)."""
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
        """Coleta fechamentos de velas já encerradas e fecha a conexão."""
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

        if data.get("msg_type") == "candles":
            try:
                cutoff=int(time.time())//CANDLE_TIMEFRAME_SEC*CANDLE_TIMEFRAME_SEC
                received.extend(float(c['close']) for c in data.get('candles', [])
                                if int(c['epoch'])+CANDLE_TIMEFRAME_SEC<=cutoff)
            except (TypeError, ValueError):
                pass

            ws.close()

    def _on_error(ws, err) -> None:
        """Fecha a conexão em caso de erro."""
        ws.close()

    def _on_close(ws, *_args) -> None:
        """Sinaliza o término da coleta."""
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
        if (ema9 - ema21) * macd_hist > 0
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
        """Pontua um símbolo (0 se fechado ou com poucos dados), guarda em `results` e imprime o resultado."""
        if not _local_market_open(symbol):
            with lock:
                results[symbol] = 0.0
            print(f"  {symbol:<12} fechado pelo calendário local")
            return

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



def _scan_candidates() -> list:
    """Lista ordenada de todos os símbolos candidatos ao scan (forex, primários, secundários)."""
    return (
        list(_SCAN_SYMBOLS_FOREX)
        + list(_SCAN_SYMBOLS_PRIMARY)
        + list(_SCAN_SYMBOLS_SECONDARY)
    )


def _local_market_open(symbol):
    """Consulta o calendário local de mercado; assume aberto em caso de falha."""
    try:
        import config as _cfg
        return _cfg.is_market_open(symbol)
    except Exception:
        return True


def _probe_symbol(symbol, timeout=20):
    """Check the actual account feed and contract catalogue without placing orders."""
    import config as cfg
    done = threading.Event()
    result = {
        'tick': False,
        'contracts': False,
        'quotes': set(),
        'proposal_errors': {},
        'reason': 'tempo limite sem tick/catalogo/propostas validos',
    }

    def request(ws):
        """Assina ticks e solicita catálogo de contratos e propostas CALL/PUT de teste."""
        ws.send(json.dumps({'ticks': symbol, 'subscribe': 1}))
        ws.send(json.dumps({'contracts_for': symbol}))
        for req_id, direction in ((910001, 'CALL'), (910002, 'PUT')):
            request = dict(proposal=1, amount=1, basis='stake', contract_type=direction,
                           currency=cfg.CURRENCY, duration=cfg.DURATION,
                           duration_unit=cfg.DURATION_UNIT, req_id=req_id)
            request['underlying_symbol' if _uses_bearer_auth() else 'symbol'] = symbol
            ws.send(json.dumps(request))

    def on_open(ws):
        """Inicia as requisições de sondagem (após `authorize`, no modo legado)."""
        if _uses_bearer_auth():
            request(ws)
        else:
            ws.send(json.dumps({'authorize': TOKEN}))

    def on_message(ws, raw):
        """Acumula tick, catálogo e propostas em `result` e encerra ao concluir ou errar."""
        try:
            data = json.loads(raw)
            if 'error' in data:
                error = data['error']
                req_id = data.get('req_id')
                message = str(error.get('message') or 'erro API')
                code = str(error.get('code') or '')
                if req_id in (910001, 910002):
                    result['proposal_errors'][req_id] = message
                    result['reason'] = f'{code}: {message}' if code else message
                    if len(result['proposal_errors']) == 2:
                        done.set()
                        ws.close()
                    return
                result['reason'] = f'{code}: {message}' if code else message
                done.set()
                ws.close()
                return
            if data.get('msg_type') == 'authorize':
                request(ws)
            if 'tick' in data:
                tick = data['tick']
                price, epoch = float(tick['quote']), int(tick['epoch'])
                result['tick'] = (tick.get('symbol') == symbol and math.isfinite(price)
                                  and price > 0 and 0 <= time.time() - epoch <= MAX_TICK_GAP_SEC)
            if 'contracts_for' in data:
                kinds = {c.get('contract_type') for c in data['contracts_for'].get('available', [])}
                result['contracts'] = {'CALL', 'PUT'} <= kinds
                if not result['contracts']:
                    result['reason'] = 'CALL/PUT indisponiveis nesta sessao'
                    done.set()
                    ws.close()
                    return
            if 'proposal' in data and data.get('req_id') in (910001, 910002):
                proposal = data['proposal']
                price, payout = float(proposal['ask_price']), float(proposal['payout'])
                if proposal.get('id') and math.isfinite(price) and math.isfinite(payout) and 0 < price < payout:
                    result['quotes'].add(data['req_id'])
            if result['tick'] and result['contracts'] and len(result['quotes']) == 2:
                result['reason'] = ''
                done.set()
                ws.close()
        except (ValueError, TypeError, KeyError):
            result['reason'] = 'resposta de mercado invalida'
            done.set()
            ws.close()

    def on_error(ws, error):
        """Registra falha de conexão como motivo e encerra a sondagem."""
        result['reason'] = 'falha de conexao na validacao do ativo'
        done.set()
        ws.close()

    try:
        ws = websocket.WebSocketApp(_get_ws_url(), on_open=on_open, on_message=on_message,
                                    on_error=on_error, on_close=lambda *_: done.set())
        worker = threading.Thread(target=ws.run_forever, daemon=True)
        worker.start()
        done.wait(timeout)
        ws.close()
        worker.join(timeout=1)
    except Exception as exc:
        return False, f'falha ao validar ativo ({type(exc).__name__})'
    return result['tick'] and result['contracts'] and len(result['quotes']) == 2 and not result['reason'], result['reason']


def _best_live_from(results):
    """Primeiro símbolo, por score decrescente e acima do piso, que passa na sondagem; retorna (símbolo, score) ou ('', 0.0)."""
    for symbol in sorted(results, key=results.get, reverse=True):
        if results[symbol] < _ADX_TREND_MIN:
            break
        available, reason = _probe_symbol(symbol)
        if available:
            return symbol, results[symbol]
        print(f'[TENDENCIA] {symbol} descartado: {reason}')
    return '', 0.0


def _top_live_from(results, limit=_TOP_TRENDING_SYMBOLS):
    """Retorna os melhores símbolos operáveis acima do piso de tendência."""
    selected = []
    for symbol in sorted(results, key=results.get, reverse=True):
        if len(selected) >= limit:
            break
        score = results[symbol]
        if score < _ADX_TREND_MIN:
            break
        available, reason = _probe_symbol(symbol)
        if available:
            selected.append((symbol, score))
        else:
            print(f'[TENDENCIA] {symbol} descartado: {reason}')
    return selected


def _first_live_symbol(candidates):
    """Retorna o primeiro símbolo com tick recente e CALL/PUT disponíveis."""
    seen = set()
    for symbol in candidates:
        if symbol in seen:
            continue
        seen.add(symbol)
        available, reason = _probe_symbol(symbol)
        if available:
            return symbol
        print(f'[TENDENCIA] {symbol} indisponivel para fallback: {reason}')
    return ''


def _first_live_symbol_with_compatible_model(candidates):
    """Retorna um símbolo operável que já possui bundle ativo aprovado."""
    seen = set()
    for symbol in candidates:
        if symbol in seen:
            continue
        seen.add(symbol)
        try:
            _payload, metadata = load_validated_bundle(AI_MODEL_PATH, symbol)
            if not usable(metadata, symbol):
                continue
        except Exception:
            continue
        available, reason = _probe_symbol(symbol)
        if available:
            return symbol
        print(f'[TENDENCIA] {symbol} possui modelo compatível, mas está indisponivel: {reason}')
    return ''


def _require_live_symbol(symbol, *, allow_fallback=False):
    """Retorna o símbolo se operável; com `allow_fallback` tenta outro candidato, senão levanta RuntimeError."""
    available, reason = _probe_symbol(symbol)
    if available:
        return symbol
    if allow_fallback:
        print(f'[TENDENCIA] {symbol} indisponivel para operar: {reason}')
        fallback = _first_live_symbol(
            [s for s in _scan_candidates() if s != symbol]
        )
        if fallback:
            print(
                "\n[TENDÊNCIA] ✔ Ativo indisponível substituído por: "
                f"{_symbol_display(fallback)}"
            )
            return fallback
    raise RuntimeError(f'Ativo {symbol} indisponivel para operar: {reason}')


def _ensure_operable_symbol(symbol):
    """Confirma mercado aberto e CALL/PUT antes de iniciar coleta de dados."""
    return _require_live_symbol(symbol, allow_fallback=True)


def _detect_top_trending_symbols(
    no_scan: bool = False,
    limit: int = _TOP_TRENDING_SYMBOLS,
) -> list:
    """Seleciona os melhores símbolos operáveis para o torneio inicial."""
    import config as _cfg

    if limit <= 0:
        return []

    if no_scan:
        return [_require_live_symbol(_cfg.SYMBOL, allow_fallback=True)]

    all_scores = {}

    print(
        "\n[TENDÊNCIA] "
        "── Torneio: scan forex (pares de moedas) ───────"
    )
    print(
        "[TENDÊNCIA] Candidatos: "
        + ", ".join(_SCAN_SYMBOLS_FOREX)
    )
    all_scores.update(_scan_group(_SCAN_SYMBOLS_FOREX))

    print(
        "\n[TENDÊNCIA] "
        "── Torneio: scan primário (volatilidade) ───────"
    )
    print(
        "[TENDÊNCIA] Candidatos: "
        + ", ".join(_SCAN_SYMBOLS_PRIMARY)
    )
    all_scores.update(_scan_group(_SCAN_SYMBOLS_PRIMARY))

    print(
        "\n[TENDÊNCIA] "
        "── Torneio: scan secundário (boom/crash/jump) ──"
    )
    print(
        "[TENDÊNCIA] Candidatos: "
        + ", ".join(_SCAN_SYMBOLS_SECONDARY)
    )
    all_scores.update(_scan_group(_SCAN_SYMBOLS_SECONDARY))

    selected = _top_live_from(all_scores, limit)
    seen = {symbol for symbol, _score in selected}

    if len(selected) < limit:
        fallback_candidates = (
            [_cfg.SYMBOL]
            + [s for s in _SCAN_SYMBOLS_PRIMARY if s != _cfg.SYMBOL]
            + [s for s in _SCAN_SYMBOLS_SECONDARY if s != _cfg.SYMBOL]
        )
        for symbol in fallback_candidates:
            if len(selected) >= limit:
                break
            if symbol in seen:
                continue
            available, reason = _probe_symbol(symbol)
            if available:
                seen.add(symbol)
                selected.append((symbol, 0.0))
            else:
                print(f'[TENDENCIA] {symbol} indisponivel para fallback: {reason}')

    symbols = [symbol for symbol, _score in selected]
    if symbols:
        print(
            "\n[TENDÊNCIA] Top símbolos para treino: "
            + ", ".join(_symbol_display(symbol) for symbol in symbols)
        )
    return symbols


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
        return _require_live_symbol(_cfg.SYMBOL, allow_fallback=True)

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

    best_fx, score_fx = _best_live_from(
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

    best, score = _best_live_from(
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

    best2, score2 = _best_live_from(
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
        "Procurando sintético operável..."
    )

    synthetic_fallback = _first_live_symbol(
        [_cfg.SYMBOL]
        + [s for s in _SCAN_SYMBOLS_PRIMARY if s != _cfg.SYMBOL]
        + [s for s in _SCAN_SYMBOLS_SECONDARY if s != _cfg.SYMBOL]
    )
    if synthetic_fallback:
        print(
            "\n[TENDÊNCIA] ✔ Fallback sintético aberto: "
            f"{_symbol_display(synthetic_fallback)}"
        )
        return synthetic_fallback

    raise RuntimeError("nenhum índice sintético com tick recente e CALL/PUT disponível")


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
    symbol = _runtime_symbol()
    available, reason = _probe_symbol(symbol)
    if not available:
        print(
            f"\n[HISTÓRICO] {symbol} indisponível para contrato "
            f"CALL/PUT de {DURATION}{DURATION_UNIT} ({reason}); "
            "histórico não será baixado."
        )
        return 0

    print(
        f"\n[HISTÓRICO] Buscando os últimos "
        f"{count:,} candles de {tf_min} min "
        f"de '{_symbol_display(symbol)}'..."
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
                        if row.get("kind") != "candle":
                            continue
                        existing_epochs.add(
                            (
                                int(row["epoch"]),
                                row.get(
                                    "symbol",
                                    "",
                                ),
                                int(row.get("timeframe") or 0),
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
    next_end = ["latest"]

    def _send_history_request(ws) -> None:
        """Pede uma página (até 1000) de velas anterior à mais antiga já recebida."""
        remaining = max(1, count - len(received))
        ws.send(
            json.dumps(
                {
                    "ticks_history": symbol,
                    "end": next_end[0],
                    "count": min(remaining, 1000),
                    "style": "candles",
                    "granularity": CANDLE_TIMEFRAME_SEC,
                    "adjust_start_time": 1,
                }
            )
        )

    def _on_open(ws) -> None:
        """Pede o histórico (Bearer) ou envia `authorize` (modo legado)."""
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
        """Acumula páginas de velas fechadas sem duplicar e pede a próxima página até atingir `count`."""
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
            page = []
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

                    if epoch > 0 and epoch+CANDLE_TIMEFRAME_SEC <= int(time.time()):
                        page.append((epoch,close,candle))

                except (
                    TypeError,
                    ValueError,
                ):
                    continue

            if page:
                known = {item[0] for item in received}
                for item in page:
                    if item[0] not in known:
                        received.append(item)
                        known.add(item[0])
                oldest = min(item[0] for item in page)
                next_end[0] = oldest - 1

            if len(received) >= count or not page:
                ws.close()
            else:
                _send_history_request(ws)

    def _on_error(ws, err) -> None:
        """Guarda a mensagem de erro e fecha a conexão."""
        error_msg.append(str(err))
        ws.close()

    def _on_close(
        ws,
        code,
        msg,
    ) -> None:
        """Sinaliza o término do download."""
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

    for epoch,close,candle in received:
        key=(epoch,symbol,CANDLE_TIMEFRAME_SEC)
        if key in existing_epochs:
            continue
        append_record(TICKS_CSV,epoch,symbol,close,kind='candle',timeframe=CANDLE_TIMEFRAME_SEC,candle=candle)
        existing_epochs.add(key)
        written+=1

    print(
        f"[HISTÓRICO] {written:,} candles "
        f"de {tf_min}min gravados — "
        f"total do ativo no CSV: {_count_market_records():,}"
    )

    return written


def _banner(
    history_count: int,
    min_ticks: int,
    demo: bool,
    symbol: Optional[str] = None,
) -> None:
    """Imprime o resumo da configuração do pipeline."""
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
        f"{_symbol_display(symbol or SYMBOL)}"
    )
    print(
        f"  Candles históricos : "
        f"{history_count:,} × {tf_min} min"
    )
    print(
        f"  Candles p/ treino  : "
        f"{_MIN_LABELABLE_CONTRACTS:,} contratos rotuláveis "
        f"({min_ticks:,} ticks reais mínimos)"
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
        f"+{RETRAIN_NEW_LABELS_TRIGGER} contratos rotulados | "
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

    print(
        "\n[AUTH] Validando token Deriv..."
    )

    max_attempts = 3

    for attempt in range(1, max_attempts + 1):
        done = threading.Event()

        result = {
            "ok": False,
            "message": "",
            "code": "",
            "loginid": "",
            "currency": "",
        }

        def _on_open(ws) -> None:
            """Envia `authorize` com o token."""
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
            """Interpreta a resposta de `authorize`, validando o tipo de conta contra o modo, e fecha a conexão."""
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

                if auth.get('is_virtual') not in (0, 1) or bool(auth['is_virtual']) != demo:
                    result['message'] = 'Tipo de conta autenticada não corresponde ao modo solicitado.'
                    ws.close()
                    return
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
            """Guarda o erro como mensagem e fecha a conexão."""
            result["message"] = str(err)
            ws.close()

        def _on_close(
            ws,
            *_args,
        ) -> None:
            """Sinaliza o fim da validação."""
            done.set()

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

            result["message"] = (
                f"Timeout após {timeout_sec}s ao validar o token."
            )

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

        transient = (
            not result["code"]
            and any(
                marker in message.lower()
                for marker in (
                    "temporary failure in name resolution",
                    "name or service not known",
                    "network is unreachable",
                    "connection timed out",
                    "timed out",
                )
            )
        )

        if transient and attempt < max_attempts:
            wait_sec = min(10, 2 * attempt)
            print(
                f"[AUTH] Falha transitória de rede/DNS: "
                f"{message}. Nova tentativa em {wait_sec}s "
                f"({attempt}/{max_attempts})."
            )
            time.sleep(wait_sec)
            continue

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
    """Imprime orientações de verificação de App ID e token Bearer/PAT."""
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

    def __init__(self, symbol: str) -> None:
        """Prepara estado, gravador de cotações de payout e último epoch já coletado do símbolo."""
        super().__init__(
            daemon=True,
            name="Collector",
        )

        self._symbol = symbol
        self.fatal_error = ''
        self._stopped = threading.Event()
        self._first_tick_timer = None
        self.history_ready = threading.Event()
        self._history_start = None
        self._history_end = None
        self._history_seen = set()
        self._last_history_log_count = 0
        from payout_quotes import ProposalRecorder
        import config as cfg
        self._payout_recorder = ProposalRecorder(Path(TICKS_CSV).with_name('payout_quotes.csv'),
            symbol, cfg.DURATION, cfg.DURATION_UNIT, cfg.CURRENCY,
            on_unavailable=_request_market_switch)

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
                        if row.get("symbol") != self._symbol or row.get("kind") != "tick":
                            continue
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
        """Mantém a conexão WebSocket, reconectando a cada 5s, até parada, encerramento ou erro fatal."""
        _ensure_ticks_header()

        while not _shutdown.is_set() and not self._stopped.is_set() and not self.fatal_error:
            try:
                ws = websocket.WebSocketApp(
                    _get_ws_url(),
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )

                self._ws = ws

                # OTP URLs are single-use: reconnect through a new session URL.
                ws.run_forever(ping_interval=30, ping_timeout=10)

            except Exception as exc:
                if not _shutdown.is_set():
                    print(
                        f"\n[COLETOR] Exceção inesperada: "
                        f"{exc} — reconectando em 5s"
                    )

            finally:
                if self._first_tick_timer:
                    self._first_tick_timer.cancel()
            if self.fatal_error or self._stopped.wait(5):
                break

    def stop(self) -> None:
        """Interrompe a coleta e fecha o WebSocket."""
        self._stopped.set()
        if self._first_tick_timer:
            self._first_tick_timer.cancel()
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass

    def _on_open(self, ws) -> None:
        """Arma o timer de primeiro tick, carrega épocas já gravadas e autoriza ou assina ticks."""
        self._first_tick_timer = threading.Timer(30, ws.close)
        self._first_tick_timer.daemon = True
        self._first_tick_timer.start()
        self.history_ready.clear()
        self._history_start = None
        records, _ = read_records(TICKS_CSV)
        self._history_seen = {r['epoch'] for r in records
                              if r['symbol'] == self._symbol and r['kind'] == 'tick'}
        self._last_history_log_count = len(self._history_seen)
        print(
            f"\n[COLETOR] Conectado | "
            f"{_symbol_display(self._symbol)}"
        )

        if not _uses_bearer_auth():
            ws.send(
                json.dumps(
                    {
                        "authorize": TOKEN
                    }
                )
            )
            return
        self._subscribe(ws)

    def _subscribe(self, ws):
        """Assina o fluxo de ticks do símbolo."""
        ws.send(
            json.dumps(
                {
                    "ticks": self._symbol,
                    "subscribe": 1,
                }
            )
        )

    def _on_message(
        self,
        ws,
        message: str,
    ) -> None:
        """Trata erros, autorização, páginas de histórico e ticks ao vivo, gravando ticks válidos em `ticks.csv`."""
        try:
            data = json.loads(message)
        except (
            TypeError,
            json.JSONDecodeError,
        ):
            return

        if self._payout_recorder.handle(data):
            return

        if "error" in data:
            error = data['error']
            from payout_quotes import market_unavailable
            if market_unavailable(error):
                _request_market_switch(self._symbol)
            if (error.get('code') in ('InvalidSymbol', 'MarketIsClosed', 'AuthorizationRequired', 'InvalidToken')
                    or 'invalid symbol' in error.get('message', '').lower()):
                self.fatal_error = f"{self._symbol}: {error.get('message', 'erro API')}"
            print(
                "\n[COLETOR] Erro API: "
                f"{data['error'].get('message', 'Erro desconhecido')}"
            )
            ws.close()
            return

        if data.get(
            "msg_type"
        ) == "authorize":
            print(
                "[COLETOR] Autorizado. "
                "Coleta iniciada..."
            )
            self._subscribe(ws)
            return

        if data.get('msg_type') == 'history':
            if self._history_start is None:
                return
            history = data.get('history', {})
            try:
                times, prices = history['times'], history['prices']
                if len(times) != len(prices):
                    raise ValueError('historico com tamanhos divergentes')
                page = sorted((int(e), float(p)) for e, p in zip(times, prices))
                if any(e <= 0 or not math.isfinite(p) or p <= 0 for e, p in page):
                    raise ValueError('tick historico invalido')
            except (KeyError, TypeError, ValueError, OverflowError):
                print('[COLETOR] Resposta historica invalida; reconectando para recuperar.')
                ws.close()
                return
            if not page or page[0][0] > self._history_end:
                print('[COLETOR] Historico incompleto; aguardando dados continuos.')
                self.history_ready.set()
                return
            for epoch, price in page:
                if (self._history_start <= epoch <= self._history_end
                        and epoch not in self._history_seen
                        and math.isfinite(price) and price > 0):
                    append_record(TICKS_CSV, epoch, self._symbol, price)
                    self._history_seen.add(epoch)
            history_count = len(self._history_seen)
            should_log_history = (
                self._last_history_log_count == 0
                or history_count - self._last_history_log_count >= _HISTORY_LOG_INTERVAL
            )
            if should_log_history:
                print(
                    f'[COLETOR] Historico: {history_count:,} ticks identificados; '
                    'coleta ao vivo ativa.'
                )
                self._last_history_log_count = history_count
            if page[0][0] <= self._history_start + MAX_TICK_GAP_SEC:
                print('[COLETOR] Ticks historicos recuperados; fluxo ao vivo permanece ativo.')
                self.history_ready.set()
            else:
                self._history_end = page[0][0] - 1
                self._request_tick_history(ws)
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

        if tick.get("symbol") != self._symbol or not math.isfinite(price) or price<=0:
            return

        if self._first_tick_timer:
            self._first_tick_timer.cancel()

        if self._history_start is None:
            self._history_start = (epoch // CANDLE_TIMEFRAME_SEC - _minimum_history_candles()) * CANDLE_TIMEFRAME_SEC
            self._history_end = epoch - 1
            print(f'[COLETOR] Recuperando ticks dos ultimos {_minimum_history_candles():,} candles para treino inicial.')
            self._request_tick_history(ws)

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

        try:
            self._payout_recorder.sample(ws, bearer=_uses_bearer_auth())
        except Exception as exc:
            print(f'[PAYOUT] Amostragem indisponível: {type(exc).__name__}')

        self._last_epoch = epoch
        self._last_price = price

        dt_str = datetime.fromtimestamp(
            epoch
        ).isoformat()

        self._local_count += 1

        append_record(TICKS_CSV,epoch,self._symbol,price)
        self._history_seen.add(epoch)

        if self._local_count == 1 or self._local_count % _TICK_LOG_INTERVAL == 0:
            print(
                f"[COLETOR] Ticks: "
                f"{self._local_count:>7,} | "
                f"{price} @ {dt_str}",
                flush=True,
            )

    def _request_tick_history(self, ws):
        """Solicita até 5000 ticks históricos do intervalo ainda faltante."""
        ws.send(json.dumps({'ticks_history': self._symbol, 'style': 'ticks',
                            'start': self._history_start, 'end': self._history_end,
                            'count': 5000}))

    def _on_error(
        self,
        ws,
        error,
    ) -> None:
        """Imprime o erro e fecha o WebSocket para reconectar."""
        if not _shutdown.is_set():
            print(
                f"\n[COLETOR] Erro WebSocket: "
                f"{error}"
            )
        ws.close()

    def _on_close(
        self,
        ws,
        code,
        msg,
    ) -> None:
        """Informa o encerramento da conexão quando não foi intencional."""
        if not _shutdown.is_set() and not self._stopped.is_set():
            action = 'coleta bloqueada' if self.fatal_error else 'reconectando...'
            print(
                f"\n[COLETOR] Conexão encerrada "
                f"(código: {code}) — {action}"
            )


# ─────────────────────────────────────────────────────────────────
# FASE 2 — Aguardar ticks mínimos
# ─────────────────────────────────────────────────────────────────

def _wait_for_ticks(
    min_ticks: int,
    poll_sec: int = 10,
    collector=None,
) -> None:
    """Bloqueia até haver ticks e contratos rotuláveis para treino temporal."""
    def check_collector():
        """Levanta se houver troca de mercado ou se o coletor falhou ou terminou."""
        _check_market_switch()
        if collector is not None and (collector.fatal_error or not collector.is_alive()):
            raise RuntimeError(collector.fatal_error or 'coletor encerrado sem dados')

    check_collector()
    _pipeline_heartbeat("collecting")
    current = _count_ticks()
    labelable = _count_labelable_contracts_for_training()

    if current >= min_ticks and labelable >= _MIN_LABELABLE_CONTRACTS:
        print(
            f"[PIPELINE] {current:,} ticks reais "
            f"e {labelable:,} contratos rotuláveis "
            "já disponíveis — pronto para treino."
        )
        return

    print(
        f"\n[PIPELINE] Aguardando "
        f"{min_ticks:,} ticks reais para o primeiro treino "
        f"e {_MIN_LABELABLE_CONTRACTS:,} contratos rotuláveis "
        f"(atual: {current:,} ticks, {labelable:,} contratos)..."
    )

    while not _shutdown.is_set():
        _pipeline_heartbeat("collecting")
        time.sleep(poll_sec)
        check_collector()

        current = _count_ticks()
        labelable = _count_labelable_contracts_for_training()

        print(
            f"\r[PIPELINE] Ticks reais: "
            f"{current:,} / {min_ticks:,} | "
            f"contratos rotuláveis: {labelable:,} / {_MIN_LABELABLE_CONTRACTS:,}   ",
            end="",
            flush=True,
        )

        if current >= min_ticks and labelable >= _MIN_LABELABLE_CONTRACTS:
            print()
            return

    sys.exit(0)


def _count_labelable_contracts_for_training() -> int:
    """Conta contratos rotuláveis do ativo no CSV de ticks; 0 se ausente."""
    if not os.path.exists(TICKS_CSV):
        return 0
    try:
        return dataset_builder.count_labelable_contracts(
            TICKS_CSV,
            symbol=_runtime_symbol(),
            window_size=FEATURE_WINDOW,
        )
    except Exception as exc:
        print(f'[PIPELINE] Dados ainda indisponiveis para treino: {exc}')
        return 0


# ─────────────────────────────────────────────────────────────────
# FASES 3 e 4 — Dataset + treino
# ─────────────────────────────────────────────────────────────────

def _load_latest_final_metrics():
    """Métricas do bundle ativo; None quando o bundle não existe ou não é utilizável."""
    global _last_model_load_error
    try:
        if not os.path.exists(AI_MODEL_PATH):
            _last_model_load_error = ""
            return None
        if not os.path.exists(AI_MODEL_PATH + '.metadata.json'):
            msg = '[PIPELINE] Modelo antigo sem metadados ignorado; treino inicial automatico necessario.'
            if msg != _last_model_load_error:
                print(msg)
                _last_model_load_error = msg
            return None
        from model_contract import load_validated_bundle
        symbol = _runtime_symbol()
        payload, metadata = load_validated_bundle(AI_MODEL_PATH, symbol)
        if usable(metadata, symbol):
            _last_model_load_error = ""
            return metadata.get('metrics')
        msg = (f"[PIPELINE] Bundle existente inutilizável: symbol={metadata.get('symbol')} "
               f"approved={metadata.get('approved')} "
               f"deps={metadata.get('dependencies')}")
    except Exception as exc:
        msg = f"[PIPELINE] Falha ao carregar modelo existente: {exc}"
    if msg != _last_model_load_error:
        print(msg)
        _last_model_load_error = msg
    return None


def _should_promote_model(current,challenger):
    """Decide se o candidato substitui o modelo atual; retorna (promover, motivo)."""
    if not challenger or not challenger.get('eligible'):
        reasons = (challenger or {}).get('rejection_reasons')
        return False, '; '.join(reasons) if reasons else 'teste final não comprova qualidade mínima no vencimento configurado'
    # A comparação usa reavaliação do bundle ativo no mesmo período do candidato.
    baseline=challenger.get('champion_metrics')
    if current is not None and baseline is None:
        return False,'champion não pôde ser reavaliado fora da amostra no mesmo período'
    if baseline is None:
        return True,'primeiro bundle compatível com teste final aprovado'
    for key in ('evaluation_start','evaluation_end','n_test'):
        if baseline.get(key)!=challenger.get(key):
            return False,'períodos de avaliação diferentes'
    ok=(challenger['auc']>=baseline['auc']+MODEL_PROMOTION_MIN_AUC_DELTA
        and challenger['accuracy']>=baseline['accuracy']-MODEL_PROMOTION_MAX_ACC_DROP
        and challenger['f1']>=baseline['f1']-MODEL_PROMOTION_MAX_F1_DROP)
    return ok,'comparação temporal no mesmo conjunto'


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
        import fcntl
        with open(TICKS_CSV, 'rb') as source, open(tmp_ticks, 'wb') as target:
            fcntl.flock(source, fcntl.LOCK_SH)
            shutil.copyfileobj(source, target)

        # Filtra o snapshot para o símbolo ativo.
        try:
            import pandas as _pd
            _active_sym = (
                _runtime_symbol()
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

        except Exception as exc:
            raise ValueError("snapshot sem esquema válido") from exc

        # Preserve the collected temporal coverage. A tick-count cap shrinks
        # evaluation differently for each feed and prevents learning from growing.

        n = dataset_builder.build_dataset(
            tmp_ticks,
            dataset_path,
            window_size=FEATURE_WINDOW,
            symbol=_runtime_symbol(),
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
        from model_contract import fingerprint, specification
        dataset_key = (fingerprint(dataset_path), json.dumps(specification(_runtime_symbol()), sort_keys=True), test_ratio)
        if getattr(_run_training, '_last_dataset_key', None) == dataset_key:
            print('[PIPELINE] Dataset rotulado inalterado; aguardando novos contratos vencidos para retreinar.')
            return False, {"status": "unchanged_dataset"}
        metrics = train_model.train(
            dataset_path,
            model_path,
            test_ratio,
            tft_output_path=tft_output_path,
            duration_output_path=duration_output_path,
            payout_quotes_path=Path(TICKS_CSV).with_name("payout_quotes.csv"),
        )
        _run_training._last_dataset_key = dataset_key
        _retrain_state["last_label_count"] = n
        print(f"[PIPELINE] Modelo treinado: {metrics.get('best_model')} | "
              f"teste={metrics.get('n_test')} amostras | contratos={metrics.get('trades')} | "
              f"AUC={metrics.get('auc', 0):.4f} | aprovado={metrics.get('eligible')}")

    except SystemExit:
        return False, None

    except Exception as exc:
        print(
            f"[PIPELINE] Erro ao treinar "
            f"modelo: {exc}"
        )

        return False, None

    return True, metrics


def _safe_symbol_slug(symbol: str) -> str:
    """Converte o símbolo em texto seguro para nomes de arquivo."""
    return "".join(ch if ch.isalnum() else "_" for ch in str(symbol)).strip("_") or "symbol"


def _training_rank_key(metrics: dict) -> tuple:
    """Chave de ordenação de candidatos: elegibilidade, expectativa, win rate, AUC, F1 e número de contratos."""
    observed = metrics.get("observed_payout_evaluation") or {}
    payout_metrics = observed if observed.get("eligible") is not None else metrics
    return (
        bool(metrics.get("eligible")),
        float(payout_metrics.get("expectancy", metrics.get("expectancy", 0.0)) or 0.0),
        float(metrics.get("win_rate", 0.0) or 0.0),
        float(metrics.get("auc", 0.0) or 0.0),
        float(metrics.get("f1", 0.0) or 0.0),
        int(metrics.get("trades", 0) or 0),
    )


def _run_initial_symbol_tournament(
    candidates: list,
    *,
    history_count: int,
    min_ticks: int,
    resources: dict,
) -> tuple[str, Optional[dict]]:
    """Treina candidatos isolados e promove somente o melhor aprovado."""
    if not candidates:
        return "", None

    print(
        "\n[TORNEIO] Iniciando treino comparativo dos "
        f"{len(candidates)} melhores símbolos."
    )

    previous_symbol = _runtime_symbol()
    results = []

    for index, symbol in enumerate(candidates, start=1):
        if _shutdown.is_set():
            break

        print(
            f"\n[TORNEIO] ({index}/{len(candidates)}) Preparando {_symbol_display(symbol)}"
        )
        _set_runtime_symbol(symbol)

        collector = None
        try:
            _ensure_ticks_header()
            collector = _CollectorThread(symbol)
            collector.start()
            _fetch_historical_ticks(history_count)
            _wait_for_ticks(min_ticks, collector=collector)

            slug = _safe_symbol_slug(symbol)
            tmp_dataset = f"{DATASET_CSV}.candidate.{slug}"
            tmp_model = f"{AI_MODEL_PATH}.candidate.{slug}"
            tmp_tft = f"{TRANSFORMER_MODEL_PATH}.candidate.{slug}"
            tmp_duration = f"{DURATION_MODEL_PATH}.candidate.{slug}"

            success, metrics = _run_training(
                dataset_path=tmp_dataset,
                model_path=tmp_model,
                tft_output_path=tmp_tft,
                duration_output_path=tmp_duration,
            )
            if success and metrics:
                promote = bool(metrics.get("eligible"))
                reason = "aprovado no teste final do torneio" if promote else "; ".join(metrics.get("rejection_reasons") or [])
                metrics = dict(metrics, tournament_promotion_reason=reason)
                print(
                    f"[TORNEIO] {symbol}: aprovado={metrics.get('eligible')} "
                    f"AUC={metrics.get('auc', 0):.4f} "
                    f"win_rate={metrics.get('win_rate', 0):.2%} "
                    f"trades={metrics.get('trades', 0)} | {reason}"
                )
                if promote:
                    results.append((symbol, tmp_model, metrics))
                else:
                    print(f"[TORNEIO] {symbol}: candidato não elegível para promoção.")
            else:
                print(f"[TORNEIO] {symbol}: treino não gerou candidato utilizável.")

        except RuntimeError as exc:
            print(f"[TORNEIO] {symbol}: coleta/treino bloqueado: {exc}")
        finally:
            if collector is not None:
                collector.stop()
                while collector.is_alive():
                    collector.join(timeout=5)

    if not results:
        _set_runtime_symbol(previous_symbol)
        print(
            "[TORNEIO] Nenhum dos símbolos gerou bundle aprovado; "
            "pipeline seguirá no fluxo adaptativo padrão."
        )
        return "", None

    winner_symbol, winner_model, winner_metrics = max(
        results,
        key=lambda item: _training_rank_key(item[2]),
    )
    _set_runtime_symbol(winner_symbol)

    os.replace(winner_model, AI_MODEL_PATH)
    winner_metadata = winner_model + ".metadata.json"
    if os.path.exists(winner_metadata):
        os.replace(winner_metadata, AI_MODEL_PATH + ".metadata.json")

    _reset_ai_predictor()
    _model_ready.set()

    print(
        "\n[TORNEIO] Vencedor promovido: "
        f"{_symbol_display(winner_symbol)} | "
        f"AUC={winner_metrics.get('auc', 0):.4f} | "
        f"win_rate={winner_metrics.get('win_rate', 0):.2%} | "
        f"trades={winner_metrics.get('trades', 0)}"
    )
    return winner_symbol, winner_metrics


# ─────────────────────────────────────────────────────────────────
# FASE 6 — Re-treino adaptativo
# ─────────────────────────────────────────────────────────────────

def _reset_ai_predictor():
    """Reinicia o estado do preditor de IA."""
    import ai_predictor
    ai_predictor.reset()


_retrain_state: dict = {
    "last_epoch": 0.0,
    "last_tick_count": 0,
    "last_label_count": 0,
    "bootstrap_failed_promotions": 0,
}

_model_ready = threading.Event()
_last_model_load_error = ""


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

    import config as cfg
    labels = _count_labelable_contracts_for_training()
    new_labels = labels - _retrain_state.get('last_label_count', 0)
    if new_labels < cfg.RETRAIN_NEW_LABELS_TRIGGER:
        return False, f'{new_labels} novos contratos rotulados; aguardando {cfg.RETRAIN_NEW_LABELS_TRIGGER}'

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

    return True, f'{new_labels} novos contratos rotulados disponíveis'


def _retrain_loop(bootstrap_until_ready: bool = False) -> None:
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

    if not _retrain_state.get('last_label_count'):
        _retrain_state['last_label_count'] = _count_labelable_contracts_for_training()

    while not _shutdown.is_set():
        _pipeline_heartbeat("training")

        for _ in range(12):
            if _shutdown.is_set():
                return

            _pipeline_heartbeat("training")
            time.sleep(5)

        if bootstrap_until_ready and _load_latest_final_metrics() is None:
            labels = _count_labelable_contracts_for_training()
            required = minimum_training_rows()
            if labels >= required:
                new_labels = labels - _retrain_state.get("last_label_count", 0)
                already_tried_dataset = getattr(_run_training, "_last_dataset_key", None) is not None
                if already_tried_dataset and new_labels < RETRAIN_NEW_LABELS_TRIGGER:
                    should, reason = False, (
                        "sem modelo compatível; "
                        f"{new_labels} novos contratos rotulados; "
                        f"aguardando {RETRAIN_NEW_LABELS_TRIGGER}"
                    )
                else:
                    should, reason = True, (
                        "sem modelo compatível; "
                        f"{labels} contratos rotulados disponíveis"
                    )
            else:
                should, reason = False, (
                    f"{labels} contratos rotulados; "
                    f"aguardando {required} para treino inicial"
                )
        else:
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

                metadata=tmp_model+'.metadata.json'
                if os.path.exists(metadata):
                    os.replace(metadata,AI_MODEL_PATH+'.metadata.json')

                _reset_ai_predictor()
                _model_ready.set()

                print(
                    "[RE-TREINO] model.pkl "
                    "atualizado com sucesso "
                    f"({promo_reason})."
                )
                _retrain_state["bootstrap_failed_promotions"] = 0

            else:
                # Keep the last trained candidate reviewable without publishing it.
                os.replace(tmp_model, AI_MODEL_PATH + '.candidate')
                if os.path.exists(tmp_model + '.metadata.json'):
                    os.replace(tmp_model + '.metadata.json', AI_MODEL_PATH + '.candidate.metadata.json')
                print(
                    "[RE-TREINO] Challenger "
                    f"não promovido: {promo_reason}"
                )
                if bootstrap_until_ready and _load_latest_final_metrics() is None:
                    failures = int(_retrain_state.get("bootstrap_failed_promotions", 0) or 0) + 1
                    _retrain_state["bootstrap_failed_promotions"] = failures
                    if failures >= BOOTSTRAP_MAX_FAILED_PROMOTIONS:
                        print(
                            "[RE-TREINO] Bootstrap sem modelo compatível após "
                            f"{failures} tentativas; solicitando troca de ativo."
                        )
                        _request_market_switch_for_unfavorable(
                            _runtime_symbol(),
                            "bootstrap_model_not_promoted",
                            failures,
                        )

        else:
            if (challenger_metrics or {}).get("status") == "unchanged_dataset":
                print(
                    "[RE-TREINO] Dataset inalterado — coleta continua "
                    "até surgirem novos contratos rotulados."
                )
            elif _load_latest_final_metrics() is None:
                print(
                    "[RE-TREINO] Falhou — ainda sem modelo compatível; "
                    "coleta e bootstrap de treino continuam ativos."
                )
                if bootstrap_until_ready:
                    failures = int(_retrain_state.get("bootstrap_failed_promotions", 0) or 0) + 1
                    _retrain_state["bootstrap_failed_promotions"] = failures
                    if failures >= BOOTSTRAP_MAX_FAILED_PROMOTIONS:
                        print(
                            "[RE-TREINO] Bootstrap falhou repetidamente sem modelo compatível; "
                            "solicitando troca de ativo."
                        )
                        _request_market_switch_for_unfavorable(
                            _runtime_symbol(),
                            "bootstrap_training_failed",
                            failures,
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


def _wait_for_compatible_model(collector=None, poll_sec: int = 10) -> bool:
    """Aguarda promoção de bundle antes de permitir entrada no executor."""
    print(
        "[PIPELINE] Aguardando primeiro modelo compatível "
        "antes de iniciar o bot de trading..."
    )
    while not _shutdown.is_set():
        _pipeline_heartbeat("waiting_model")
        _check_market_switch()
        if _load_latest_final_metrics() is not None:
            _model_ready.set()
            return True
        if collector is not None and (collector.fatal_error or not collector.is_alive()):
            print(
                "[PIPELINE] Coletor encerrado antes de existir modelo compatível; "
                "bot não será iniciado."
            )
            return False
        _model_ready.wait(timeout=poll_sec)
        if _model_ready.is_set() and _load_latest_final_metrics() is not None:
            return True
    return False


# ─────────────────────────────────────────────────────────────────
# Confirmação modo real
# ─────────────────────────────────────────────────────────────────

def _confirm_real_mode() -> bool:
    """Exibe aviso do modo real e retorna True somente se o usuário digitar 'sim'."""
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

def _main_session(resources, forced_symbol=None) -> None:
    """Executa uma sessão do pipeline para um mercado: argumentos, coleta, treino, bot e retreino."""
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

    parser.add_argument('--reuse-training-data', action='store_true',
                        help='Reutiliza dados/modelos anteriores; por padrao inicia uma nova coleta e treino.')

    parser.add_argument(
        "--no-scan",
        action="store_true",
        help=(
            "Pula o scan de tendência e "
            "prefere SYMBOL de config.py, com fallback se indisponível."
        ),
    )

    parser.add_argument(
        "--top-symbols",
        type=int,
        default=_TOP_TRENDING_SYMBOLS,
        metavar="N",
        help=(
            "Quantidade de símbolos operáveis a treinar no torneio inicial "
            f"(1-{_TOP_TRENDING_SYMBOLS}; padrão: {_TOP_TRENDING_SYMBOLS})."
        ),
    )

    args = parser.parse_args()
    if not 1 <= args.top_symbols <= _TOP_TRENDING_SYMBOLS:
        parser.error(f"--top-symbols deve estar entre 1 e {_TOP_TRENDING_SYMBOLS}")
    execution_lock=resources["lock"]

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

    # ── Ctrl+C ────────────────────────────────────────────────

    def _handle_interrupt(
        sig,
        frame,
    ) -> None:
        """Handler de Ctrl+C: sinaliza o encerramento e sai."""
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
    model_required = bool(_cfg.USE_AI_MODEL)
    active_modules = [
        label for enabled, label in (
            (_cfg.USE_WEIGHTED_SIGNAL, 'score ponderado'),
            (_cfg.USE_AI_MODEL, 'modelo treinado'),
            (_cfg.PRICE_ACTION_FILTER, 'Price Action'),
            (_cfg.CANDLE_PATTERN_FILTER, 'padrões de vela'),
        ) if enabled
    ]
    print('[PIPELINE] Módulos de decisão ativos: ' + ', '.join(active_modules or ['nenhum']))

    state_path=Path(__file__).resolve().parent/'state.json'
    prior=json.loads(state_path.read_text()) if state_path.exists() else {}
    pending=prior.get('active_contract')
    tournament_candidates = []
    try:
        if pending:
            detected = pending.get('symbol')
        elif forced_symbol:
            detected = forced_symbol
        elif args.skip_collect:
            detected = _cfg.SYMBOL
        else:
            tournament_candidates = _detect_top_trending_symbols(no_scan=args.no_scan, limit=args.top_symbols)
            detected = tournament_candidates[0] if tournament_candidates else _detect_trending_symbol(no_scan=args.no_scan)

        if not pending and not args.skip_collect:
            detected = _ensure_operable_symbol(detected)
    except RuntimeError as exc:
        print(f'[PIPELINE] Sem mercado disponível na seleção: {exc}')
        raise MarketUnavailable(_runtime_symbol()) from exc

    if detected != _cfg.SYMBOL:
        print(
            f"[TENDÊNCIA] Símbolo alterado: "
            f"{_cfg.SYMBOL} → {detected}"
        )

    if (
        not pending
        and not args.force_retrain
        and forced_symbol is None
        and not tournament_candidates
        and model_required
    ):
        model_symbol = _first_live_symbol_with_compatible_model(
            [detected] + [s for s in _scan_candidates() if s != detected]
        )
        if model_symbol and model_symbol != detected:
            print(
                "[TENDÊNCIA] Scan escolheu ativo sem bundle aprovado; "
                f"mantendo modelo compatível em {_symbol_display(model_symbol)}."
            )
            detected = model_symbol

    _set_runtime_symbol(detected)

    history_count = _history_fetch_count_for_start(args.history_count)

    if model_required:
        _prepare_training_start(reuse=args.reuse_training_data or forced_symbol is not None,
                                skip_collect=args.skip_collect, pending=pending)
    else:
        print('[PIPELINE] IA desligada; preservando dados e artefatos de treinamento.')

    if model_required and tournament_candidates and not pending and not args.skip_collect:
        winner_symbol, _winner_metrics = _run_initial_symbol_tournament(
            tournament_candidates,
            history_count=history_count,
            min_ticks=args.min_ticks,
            resources=resources,
        )
        if winner_symbol:
            detected = winner_symbol
        else:
            _set_runtime_symbol(detected)

    _banner(
        history_count,
        args.min_ticks,
        is_demo,
        symbol=detected,
    )

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

    collector = None
    if not args.skip_collect:
        _ensure_ticks_header()
        collector = _CollectorThread(detected)
        resources["collector"] = collector
        collector.start()

    if not args.skip_collect:
        n_hist = (
            _fetch_historical_ticks(
                history_count
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

    if not args.skip_collect:
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

    model_exists = not model_required or (
        _load_latest_final_metrics() is not None and not args.force_retrain
    )

    if not model_required:
        print('\n[PIPELINE] Modelo treinado desativado; treino inicial ignorado.')

    elif model_exists or pending:
        print(
            "\n[PIPELINE] Fases 2-4: "
            "treino inicial ignorado: modelo compatível ou contrato pendente a reconciliar."
        )

        print(
            "           Use --force-retrain "
            "para forçar novo treino."
        )

    else:
        # O treino exige ticks contínuos do ativo ativo (gaps <= MAX_TICK_GAP_SEC);
        # histórico legado não qualifica. Tenta a cada ciclo até um treino elegível.
        success, _metrics = False, None

        while not _shutdown.is_set() and not success:
            try:
                _wait_for_ticks(args.min_ticks, collector=collector)
            except RuntimeError as exc:
                print(f'[PIPELINE] Coleta interrompida; treino bloqueado: {exc}')
                if collector:
                    collector.stop()
                return

            print(
                "[PIPELINE] Tentativa de treino inicial "
                "com os ticks contínuos disponíveis..."
            )

            success, _metrics = (
                _run_training(model_path=AI_MODEL_PATH+".new")
            )

            if not success and not _shutdown.is_set():
                print(
                    "[PIPELINE] Ainda sem dados contínuos suficientes "
                    "(ticks ao vivo identificados são necessários). "
                    "Nova tentativa após mais coleta."
                )

                if _market_switch.wait(300):
                    _check_market_switch()
                if _shutdown.is_set():
                    return

        if not success:
            return

        _check_market_switch()
        champion_metrics = _load_latest_final_metrics()

        if success and _metrics and not _metrics.get('champion_metrics'):
            # Reavalia o champion no mesmo período do candidato para comparação justa.
            try:
                import train_model as _tm
                from model_contract import load_validated_bundle

                _symbol = _runtime_symbol()
                _champion_bundle, _cm = load_validated_bundle(AI_MODEL_PATH, _symbol)
                _cm = _champion_bundle.get('metadata', {})

                if usable(_cm, _symbol):
                    _champ_model = _champion_bundle['model']

                    import pandas as _pd
                    _df = _pd.read_csv(DATASET_CSV)

                    _eval = _df[
                        (_df['entry_epoch'] >= _metrics.get('evaluation_start', 0))
                        & (_df['entry_epoch'] <= _metrics.get('evaluation_end', 1 << 40))
                    ]

                    if len(_eval) >= 30:
                        _metrics['champion_metrics'] = _tm._metrics(_champ_model, _eval)
            except Exception:
                pass

        promote, reason = _should_promote_model(champion_metrics, _metrics)
        if not success or not promote:
            print(
                "\n[PIPELINE] Treino inicial "
                "produziu modelo não elegível. "
                f"Motivo: {reason}. Aguardando mais dados para nova tentativa."
            )

            _retrain_state["last_epoch"] = time.time()
            _retrain_state["last_tick_count"] = _count_ticks()
            model_exists = False
            print('[PIPELINE] Bootstrap de re-treino assumirá a promoção do primeiro bundle elegível')
        else:
            os.replace(AI_MODEL_PATH+'.new',AI_MODEL_PATH)
            if os.path.exists(AI_MODEL_PATH+'.new.metadata.json'):
                os.replace(AI_MODEL_PATH+'.new.metadata.json',AI_MODEL_PATH+'.metadata.json')
            model_exists=True
            _model_ready.set()
            _reset_ai_predictor()

    # ── FASE 5: Re-treino ─────────────────────────────────────

    if model_required and model_exists:
        retrain_thread = threading.Thread(
            target=_retrain_loop,
            daemon=True,
            name="Retrain",
        )
        resources["retrain"] = retrain_thread
        retrain_thread.start()
        print('[PIPELINE] Re-treino adaptativo iniciado para o bundle compatível')
    elif not model_required:
        print('[PIPELINE] IA desligada; retreino e promoção de modelo não bloqueiam a execução.')
    else:
        retrain_thread = threading.Thread(
            target=_retrain_loop,
            kwargs={'bootstrap_until_ready': True},
            daemon=True,
            name="RetrainBootstrap",
        )
        resources["retrain"] = retrain_thread
        retrain_thread.start()
        if pending:
            print(
                '[PIPELINE] Contrato pendente sem bundle compatível; '
                'bot reconciliará a operação enquanto o bootstrap treina em paralelo.'
            )
        else:
            print('[PIPELINE] Sem modelo compatível; entradas permanecem bloqueadas até promoção.')
            print('[PIPELINE] Coleta e treino inicial adaptativo ativos para o símbolo atual')
            if not _wait_for_compatible_model(collector=collector):
                return
            model_exists = True

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

    risk_manager._authenticated_initial_balance = initial_balance

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

    bot._execution_lock=execution_lock
    bot.market_switch_requested = _market_switch.is_set
    bot.on_market_unavailable = _request_market_switch
    bot.on_market_unfavorable = _request_market_switch_for_unfavorable
    bot.run()
    _check_market_switch()


def _finish_market_session(resources):
    """Encerra o coletor e aguarda o retreino em andamento antes de trocar de ativo."""
    # Never change module symbols while the previous training job can still write.
    _shutdown.set()
    collector = resources.pop('collector', None)
    if collector:
        collector.stop()
        while collector.is_alive():
            collector.join(timeout=30)
    retrain = resources.pop('retrain', None)
    if retrain:
        while retrain.is_alive():
            print('[MERCADO] Aguardando treino atual terminar antes da troca de ativo...')
            retrain.join(timeout=30)


def main() -> None:
    """Ponto de entrada: obtém o lock de execução e repete sessões, trocando de mercado quando indisponível."""
    resources = {'lock': acquire_execution_lock(Path(__file__).resolve().parent/'executor.lock')}
    forced_symbol = None
    try:
        while True:
            _shutdown.clear()
            _market_switch.clear()
            try:
                _main_session(resources, forced_symbol)
                return
            except MarketUnavailable as exc:
                old_symbol = str(exc)
            finally:
                _finish_market_session(resources)
            # Keep risk, historical data and account state. Only the market changes.
            _mark_market_failure(old_symbol, "market_unavailable")
            _shutdown.clear()
            forced_symbol = ''
            while not forced_symbol and not _shutdown.is_set():
                if _switch_budget_exhausted():
                    _write_pipeline_status(
                        status="degraded",
                        running=False,
                        stopped_at=datetime.utcnow().isoformat() + "+00:00",
                        finished_at=datetime.utcnow().isoformat() + "+00:00",
                        returncode=None,
                        last_error="limite de trocas de mercado atingido; aguardando intervenção",
                    )
                    print('[MERCADO] Limite de trocas atingido; pipeline entrou em estado degradado.')
                    return
                candidates = [
                    s for s in _scan_candidates()
                    if s != old_symbol and _market_available_for_switch(s)
                ]
                forced_symbol = _first_live_symbol_with_compatible_model(candidates)
                if forced_symbol:
                    print(
                        f'[MERCADO] {old_symbol} → {forced_symbol}; '
                        'usando bundle compatível já aprovado.'
                    )
                    break
                forced_symbol = _first_live_symbol(candidates)
                if forced_symbol:
                    print(
                        f'[MERCADO] {old_symbol} → {forced_symbol}; '
                        'sem bundle aprovado, coleta e treino serão iniciados.'
                    )
                    break
                print('[MERCADO] Nenhum alternativo disponível; nova consulta em 60 segundos.')
                _shutdown.wait(60)
            if not forced_symbol:
                return
            _model_ready.clear()
            _retrain_state.update(
                last_epoch=0,
                last_tick_count=0,
                last_label_count=0,
                bootstrap_failed_promotions=0,
            )
            _reset_ai_predictor()
    finally:
        resources['lock'].close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        _write_pipeline_status(
            status="stopped",
            running=False,
            stopped_at=datetime.utcnow().isoformat() + "+00:00",
            finished_at=datetime.utcnow().isoformat() + "+00:00",
            returncode=0,
            last_error="",
        )
        raise
    except Exception as exc:
        _write_pipeline_status(
            status="crashed",
            running=False,
            stopped_at=datetime.utcnow().isoformat() + "+00:00",
            finished_at=datetime.utcnow().isoformat() + "+00:00",
            returncode=1,
            last_error=f"{type(exc).__name__}: {exc}",
        )
        raise
    else:
        _write_pipeline_status(
            status="stopped",
            running=False,
            stopped_at=datetime.utcnow().isoformat() + "+00:00",
            finished_at=datetime.utcnow().isoformat() + "+00:00",
            returncode=0,
            last_error="",
        )
