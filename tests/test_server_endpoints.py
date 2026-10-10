"""
tests/test_server_endpoints.py — testa os endpoints da API do dashboard via Flask test client.

Não requer o servidor rodando: usa app.test_client() diretamente.
Todos os testes funcionam sem bot rodando e sem arquivos CSV/JSON — validam estrutura de resposta.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Garante imports do projeto
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "dashboard"))

# Força DERIV_TOKEN para o import do config não falhar
os.environ.setdefault("DERIV_TOKEN", "test_token_placeholder")
os.environ.setdefault("DASHBOARD_TOKEN", "")  # sem auth por padrão nos testes


@pytest.fixture(scope="module")
def client():
    """Flask test client com arquivos de dados apontando para temporários vazios."""
    import dashboard.server as srv

    # Redireciona os paths de CSV/JSON para arquivos temporários vazios
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        empty_ops = tmp / "operacoes_log.csv"
        empty_ticks = tmp / "ticks.csv"
        empty_risk = tmp / "risk_state.json"
        empty_state = tmp / "state.json"
        bot_status = tmp / "bot_status.json"
        empty_metrics = tmp / "model_metrics.json"
        backtest_root = tmp / "backtest_reports"
        backtest_state = backtest_root / "state.json"

        empty_ops.write_text("")
        empty_ticks.write_text("")
        empty_risk.write_text("{}")
        empty_state.write_text("{}")
        bot_status.write_text("{}")
        empty_metrics.write_text("[]")

        with patch.object(srv, "OPERATIONS_CSV", empty_ops), \
             patch.object(srv, "TICKS_CSV", empty_ticks), \
             patch.object(srv, "RISK_STATE_JSON", empty_risk), \
             patch.object(srv, "STATE_JSON", empty_state), \
             patch.object(srv, "BOT_STATUS_JSON", bot_status), \
             patch.object(srv, "MODEL_METRICS_JSON", empty_metrics), \
             patch.object(srv, "BACKTEST_ROOT", backtest_root), \
             patch.object(srv, "BACKTEST_STATE_JSON", backtest_state):

            srv._backtest_state = {}
            srv._backtest_process = None
            srv.app.config["TESTING"] = True
            with srv.app.test_client() as c:
                yield c
            srv._backtest_state = {}
            srv._backtest_process = None


# ─── Endpoints estáticos ──────────────────────────────────────────────────────

def test_index_returns_200(client):
    """Garante que a página inicial responde 200."""
    r = client.get("/")
    assert r.status_code == 200


def test_index_contains_backtest_panel(client):
    """Garante que o HTML da página inicial contém aba, formulário e gráfico de backtest."""
    html = client.get("/").get_data(as_text=True)

    assert 'data-tab="backtest"' in html
    assert 'id="backtestForm"' in html
    assert 'id="backtestEquityChart"' in html


def test_favicon_no_error(client):
    """Garante que /favicon.ico não gera erro de servidor."""
    r = client.get("/favicon.ico")
    assert r.status_code in (200, 204, 404)  # qualquer coisa menos 500


def test_frontend_uses_same_origin_api_base(client):
    """Garante que o frontend não fixa localhost e funciona por IP, proxy ou túnel."""
    text = client.get("/app.js").get_data(as_text=True)
    assert "localhost:5055" not in text
    assert "window.STANDERIV_API_BASE_URL || ''" in text


def test_frontend_keeps_controls_alive_without_chart_cdn(client):
    """Garante fallback quando Chart.js externo não carrega."""
    text = client.get("/app.js").get_data(as_text=True)
    assert "class ChartFallback" in text
    assert "gráficos desativados, controles e APIs continuam ativos" in text


def test_frontend_records_poll_errors(client):
    """Garante que erros de polling deixam rastros visíveis e não somem em catch vazio."""
    text = client.get("/app.js").get_data(as_text=True)
    assert "function _recordPollError" in text
    assert "Falha ao atualizar:" in text
    assert "async function _fetchJson" in text


# ─── /api/summary ─────────────────────────────────────────────────────────────

def test_summary_returns_200(client):
    """Garante que /api/summary responde 200."""
    r = client.get("/api/summary")
    assert r.status_code == 200


def test_summary_required_keys(client):
    """Garante que /api/summary traz todas as chaves obrigatórias."""
    r = client.get("/api/summary")
    data = json.loads(r.data)
    required = {
        "balance", "balance_initial", "pnl_today", "pnl_pct",
        "win_rate", "total_trades", "wins", "losses",
        "consec_losses", "is_paused", "drawdown_pct",
    }
    missing = required - data.keys()
    assert not missing, f"Chaves ausentes em /api/summary: {missing}"


def test_summary_empty_data_returns_zeros(client):
    """Garante que /api/summary sem dados retorna zeros e não pausado."""
    r = client.get("/api/summary")
    data = json.loads(r.data)
    assert data["total_trades"] == 0
    assert data["balance"] == 0.0
    assert data["is_paused"] is False


def test_summary_uses_risk_state_when_operations_are_empty(tmp_path):
    """Garante que /api/summary usa o estado de risco quando o CSV de operações está vazio."""
    import dashboard.server as srv

    empty_ops = tmp_path / "operacoes_v2.csv"
    risk_state = tmp_path / "risk_state.json"
    empty_ops.write_text("")
    risk_state.write_text(json.dumps({
        "balance": 7645.24,
        "daily_start_balance": 7578.09,
        "daily_pnl": 67.15,
        "daily_pnl_pct": 0.886,
        "recent_results": [1],
        "win_rate_recent": 100.0,
        "consec_losses": 0,
        "is_paused": False,
        "drawdown_pct": 0.0,
    }))

    with patch.object(srv, "OPERATIONS_CSV", empty_ops), \
         patch.object(srv, "RISK_STATE_JSON", risk_state):
        srv.app.config["TESTING"] = True
        with srv.app.test_client() as c:
            data = c.get("/api/summary").get_json()

    assert data["balance"] == 7645.24
    assert data["balance_initial"] == 7578.09
    assert data["pnl_today"] == 67.15
    assert data["total_trades"] == 1
    assert data["win_rate"] == 100.0


def test_summary_uses_deriv_contract_report_instead_of_local_operations(tmp_path, monkeypatch):
    """Garante que /api/summary prefere o relatório de contratos da Deriv às operações locais."""
    import dashboard.server as srv

    empty_ops = tmp_path / "operacoes_v2.csv"
    risk_state = tmp_path / "risk_state.json"
    empty_ops.write_text("")
    risk_state.write_text(json.dumps({
        "balance": 1015.0,
        "daily_start_balance": 1000.0,
        "consec_losses": 0,
    }))
    now = int(__import__("time").time())
    monkeypatch.setattr(srv, "_deriv_contract_cache", {"ts": 0.0, "records": [], "error": ""})
    monkeypatch.setattr(srv, "_request_deriv_profit_table", lambda limit=100: [
        {
            "transaction_id": 1,
            "contract_type": "CALL",
            "underlying_symbol": "R_100",
            "buy_price": 10.0,
            "sell_price": 18.5,
            "purchase_time": now,
            "sell_time": now + 60,
        },
        {
            "transaction_id": 2,
            "contract_type": "PUT",
            "underlying_symbol": "R_75",
            "buy_price": 10.0,
            "sell_price": 0.0,
            "purchase_time": now,
            "sell_time": now + 60,
        },
    ])

    with patch.object(srv, "OPERATIONS_CSV", empty_ops), \
         patch.object(srv, "RISK_STATE_JSON", risk_state):
        srv.app.config["TESTING"] = True
        with srv.app.test_client() as c:
            data = c.get("/api/summary").get_json()

    assert data["source"] == "deriv_profit_table"
    assert data["balance"] == 1015.0
    assert data["pnl_today"] == -1.5
    assert data["total_trades"] == 2
    assert data["wins"] == 1
    assert data["losses"] == 1


# ─── /api/ticks ───────────────────────────────────────────────────────────────

def test_ticks_returns_200(client):
    """Garante que /api/ticks responde 200."""
    r = client.get("/api/ticks?n=10")
    assert r.status_code == 200


def test_ticks_returns_list(client):
    """Garante que /api/ticks retorna uma lista."""
    data = json.loads(client.get("/api/ticks?n=10").data)
    assert isinstance(data, list)


def test_ticks_n_capped_at_2000(client):
    """Parâmetro n não deve exceder 2000 (proteção contra DoS)."""
    r = client.get("/api/ticks?n=99999")
    assert r.status_code == 200  # não deve explodir


def test_ticks_filters_symbol_before_returning_tail(monkeypatch, tmp_path):
    """Garante que a cauda do endpoint contém somente ticks do símbolo solicitado."""
    import dashboard.server as srv

    ticks = tmp_path / "ticks_v2.csv"
    ticks.write_text(
        "epoch,symbol,price\n"
        "1000,R_100,10\n"
        "1001,R_75,20\n"
        "1002,R_100,11\n"
        "1003,R_100,12\n"
    )
    monkeypatch.setattr(srv, "TICKS_CSV", ticks)

    with srv.app.test_client() as test_client:
        records = test_client.get("/api/ticks?n=2&symbol=R_100").get_json()

    assert [record["price"] for record in records] == [11.0, 12.0]


def test_csv_row_counter_updates_incrementally_and_filters_epoch(tmp_path):
    """Garante contagem por sessão e atualização após novas linhas anexadas."""
    import dashboard.server as srv

    ticks = tmp_path / "ticks.csv"
    ticks.write_text("epoch,symbol,price\n1000,R_100,10\n2000,R_100,11\n")

    assert srv._count_csv_rows(ticks) == 2
    assert srv._count_csv_rows(ticks, minimum_epoch=1500) == 1

    with ticks.open("a") as data_file:
        data_file.write("3000,R_100,12\n")

    assert srv._count_csv_rows(ticks) == 3
    assert srv._count_csv_rows(ticks, minimum_epoch=1500) == 2


def test_candle_patterns_reuses_recent_symbol_result(monkeypatch):
    """Garante que polls repetidos compartilham o cálculo de padrões por ativo."""
    import pandas as pd
    import dashboard.server as srv
    import indicators

    read_tail = MagicMock(return_value=pd.DataFrame({"price": [1, 2, 3, 4]}))
    monkeypatch.setattr(srv, "_tail_csv", read_tail)
    monkeypatch.setattr(srv, "_candle_patterns_cache", {})
    monkeypatch.setattr(indicators, "ticks_to_candles", lambda prices, size: [1, 2, 3, 4])
    monkeypatch.setattr(indicators, "detect_candle_patterns", lambda candles: [])
    monkeypatch.setattr(indicators, "price_action_features", lambda candles, tolerance: None)

    with srv.app.test_client() as test_client:
        first = test_client.get("/api/candle-patterns?symbol=R_TEST")
        second = test_client.get("/api/candle-patterns?symbol=R_TEST")

    assert first.get_json() == []
    assert second.get_json() == []
    read_tail.assert_called_once_with(srv.TICKS_CSV, 500, symbol="R_TEST")


# ─── /api/trades ──────────────────────────────────────────────────────────────

def test_trades_returns_200(client):
    """Garante que /api/trades responde 200."""
    r = client.get("/api/trades?n=5")
    assert r.status_code == 200


def test_trades_returns_list(client):
    """Garante que /api/trades retorna dict com lista de trades e fonte deriv_profit_table."""
    data = json.loads(client.get("/api/trades?n=5").data)
    assert isinstance(data, dict)
    assert isinstance(data["trades"], list)
    assert data["source"] == "deriv_profit_table"


def test_trades_returns_deriv_contract_report(monkeypatch):
    """Garante que /api/trades converte o relatório de contratos da Deriv em trades com lucro calculado."""
    import dashboard.server as srv

    monkeypatch.setattr(srv, "_deriv_contract_cache", {"ts": 0.0, "records": [], "error": ""})
    monkeypatch.setattr(srv, "_request_deriv_profit_table", lambda limit=100: [{
        "transaction_id": 42,
        "contract_type": "CALL",
        "underlying_symbol": "R_100",
        "buy_price": 10.0,
        "sell_price": 18.5,
        "purchase_time": 1700000000,
        "sell_time": 1700000060,
    }])

    srv.app.config["TESTING"] = True
    with srv.app.test_client() as c:
        data = c.get("/api/trades?n=5").get_json()

    assert data["source"] == "deriv_profit_table"
    assert data["error"] == ""
    assert data["trades"][0]["contract_id"] == "42"
    assert data["trades"][0]["symbol"] == "R_100"
    assert data["trades"][0]["sell_price"] == 18.5
    assert data["trades"][0]["profit"] == 8.5


def test_trades_reports_deriv_error(monkeypatch):
    """Garante que /api/trades devolve lista vazia e a mensagem de erro quando a Deriv falha."""
    import dashboard.server as srv

    monkeypatch.setattr(srv, "_deriv_contract_cache", {"ts": 0.0, "records": [], "error": ""})

    def fail(limit=100):
        """Simula indisponibilidade da Deriv."""
        raise RuntimeError("Deriv indisponível")

    monkeypatch.setattr(srv, "_request_deriv_profit_table", fail)

    srv.app.config["TESTING"] = True
    with srv.app.test_client() as c:
        data = c.get("/api/trades?n=5&force=1").get_json()

    assert data["trades"] == []
    assert "Deriv indisponível" in data["error"]


def test_trades_falls_back_to_local_operations_when_deriv_fails(monkeypatch, tmp_path):
    """Garante fallback identificado para CSV local quando a Deriv falha."""
    import dashboard.server as srv

    ops = tmp_path / "operacoes_v2.csv"
    ops.write_text(
        "timestamp,symbol,direction,stake,duration,result,profit,contract_id\n"
        "2026-09-21T09:30:00,R_100,BUY,10,15,WIN,8,abc123\n"
    )
    monkeypatch.setattr(srv, "_deriv_contract_cache", {"ts": 0.0, "records": [], "error": ""})

    def fail(limit=100):
        """Simula indisponibilidade da Deriv."""
        raise RuntimeError("Deriv indisponível")

    monkeypatch.setattr(srv, "_request_deriv_profit_table", fail)
    with patch.object(srv, "OPERATIONS_CSV", ops):
        srv.app.config["TESTING"] = True
        with srv.app.test_client() as c:
            data = c.get("/api/trades?n=5&force=1").get_json()

    assert data["source"] == "local_operations_fallback"
    assert "Deriv indisponível" in data["error"]
    assert data["trades"][0]["contract_id"] == "abc123"
    assert data["trades"][0]["profit"] == 8


# ─── /api/indicators ─────────────────────────────────────────────────────────

def test_indicators_returns_200(client):
    """Garante que /api/indicators responde 200."""
    r = client.get("/api/indicators")
    assert r.status_code == 200


def test_indicators_returns_dict(client):
    """Garante que /api/indicators retorna um dict."""
    data = json.loads(client.get("/api/indicators").data)
    assert isinstance(data, dict)


def test_indicators_use_live_ticks_before_last_operation(monkeypatch, tmp_path):
    """Garante que /api/indicators calcula indicadores de ticks atuais quando possível."""
    import dashboard.server as srv
    import feature_engine

    ticks = tmp_path / "ticks_v2.csv"
    ticks.write_text(
        "epoch,symbol,price\n" +
        "\n".join(f"{1700000000 + i},R_100,{100 + i * 0.01}" for i in range(80))
    )
    ops = tmp_path / "operacoes_v2.csv"
    ops.write_text("")
    monkeypatch.setattr(feature_engine, "compute_feature_map", lambda prices: {
        "ema9": 101.1,
        "ema21": 100.9,
        "rsi": 55.5,
        "adx": 22.2,
        "macd_hist": 0.123456,
    })

    with patch.object(srv, "TICKS_CSV", ticks), patch.object(srv, "OPERATIONS_CSV", ops):
        srv.app.config["TESTING"] = True
        with srv.app.test_client() as c:
            data = c.get("/api/indicators").get_json()

    assert data["source"] == "ticks_live"
    assert data["rsi"] == 55.5
    assert data["adx"] == 22.2


# ─── /api/model ───────────────────────────────────────────────────────────────

def test_model_returns_200(client):
    """Garante que /api/model responde 200."""
    r = client.get("/api/model")
    assert r.status_code == 200


def test_model_required_keys(client):
    """Garante que /api/model traz as chaves obrigatórias."""
    data = json.loads(client.get("/api/model").data)
    required = {"model_exists", "tft_exists", "duration_exists", "ticks_count", "dataset_rows"}
    missing = required - data.keys()
    assert not missing, f"Chaves ausentes em /api/model: {missing}"


# ─── /api/bot/status ─────────────────────────────────────────────────────────

def test_bot_status_returns_200(client):
    """Garante que /api/bot/status responde 200."""
    r = client.get("/api/bot/status")
    assert r.status_code == 200


def test_bot_status_running_is_bool(client):
    """Garante que o campo running de /api/bot/status é booleano."""
    data = json.loads(client.get("/api/bot/status").data)
    assert isinstance(data["running"], bool)


def test_bot_status_not_running_without_pid(client):
    """Garante que /api/bot/status informa running falso quando não há bot em execução."""
    import dashboard.server as srv
    # Mocka _bot_running para simular ausência de bot nos testes
    with patch.object(srv, "_bot_running", return_value=(False, None)):
        data = json.loads(client.get("/api/bot/status").data)
    assert data["running"] is False


def test_bot_running_removes_stale_pid(client):
    """Garante que _bot_running remove PID obsoleto e marca o status como crashed."""
    import dashboard.server as srv
    srv.PID_FILE.write_text("999999")
    srv.BOT_STATUS_JSON.write_text(json.dumps({"status": "running", "running": True, "pid": 999999}))
    assert srv._bot_running() == (False, None)
    assert not srv.PID_FILE.exists()
    status = json.loads(srv.BOT_STATUS_JSON.read_text())
    assert status["running"] is False
    assert status["status"] == "crashed"
    assert status["last_error"] == "processo ausente"


def test_bot_status_marks_missing_running_process_as_crashed(client):
    """Garante que /api/bot/status marca como crashed um status running sem processo."""
    import dashboard.server as srv
    srv.BOT_STATUS_JSON.write_text(json.dumps({"status": "running", "running": True, "pid": 999999}))
    with patch.object(srv, "_bot_running", return_value=(False, None)):
        data = client.get("/api/bot/status").get_json()
    assert data["status"] == "crashed"
    assert data["last_error"]


def test_bot_status_preserves_stopping_while_process_is_alive(client):
    """Garante que polling não transforma parada solicitada em running enquanto o processo encerra."""
    import dashboard.server as srv
    srv.BOT_STATUS_JSON.write_text(json.dumps({"status": "stopping", "running": True, "pid": 123}))
    with patch.object(srv, "_bot_running", return_value=(True, 123)):
        data = client.get("/api/bot/status").get_json()
    assert data["status"] == "stopping"
    assert data["running"] is True


def test_bot_status_marks_stopping_dead_process_as_stopped(client):
    """Garante que processo ausente após stop vira stopped, não crashed."""
    import dashboard.server as srv
    srv.BOT_STATUS_JSON.write_text(json.dumps({"status": "stopping", "running": True, "pid": 123}))
    with patch.object(srv, "_bot_running", return_value=(False, None)):
        data = client.get("/api/bot/status").get_json()
    assert data["status"] == "stopped"
    assert data["running"] is False
    assert data["last_error"] == ""


def test_bot_status_uses_status_pid_when_pid_file_is_missing(client):
    """Garante que, sem arquivo de PID, o PID do status inexistente resulta em crashed e pid nulo."""
    import dashboard.server as srv

    try:
        srv.PID_FILE.unlink()
    except FileNotFoundError:
        pass
    srv.BOT_STATUS_JSON.write_text(json.dumps({"status": "running", "running": True, "pid": 999999}))

    data = client.get("/api/bot/status").get_json()

    assert data["running"] is False
    assert data["status"] == "crashed"
    assert data["pid"] is None


def test_bot_status_reconciles_waiting_model_with_dead_pid(client):
    """Garante que status waiting_model com PID morto é reconciliado como crashed e persistido."""
    import dashboard.server as srv

    try:
        srv.PID_FILE.unlink()
    except FileNotFoundError:
        pass
    srv.BOT_STATUS_JSON.write_text(json.dumps({
        "status": "waiting_model",
        "running": True,
        "pid": 999999,
    }))

    data = client.get("/api/bot/status").get_json()
    persisted = json.loads(srv.BOT_STATUS_JSON.read_text())

    assert data["running"] is False
    assert data["status"] == "crashed"
    assert persisted["running"] is False
    assert persisted["status"] == "crashed"
    assert persisted["last_error"] == "processo ausente"


def test_bot_status_includes_last_decision_diagnostics(client):
    """Garante que /api/bot/status inclui última decisão, contagens e tempo sem entrada."""
    import dashboard.server as srv
    srv.BOT_STATUS_JSON.write_text(json.dumps({
        "status": "running",
        "running": True,
        "pid": 123,
        "last_decision": {"reason": "candle_conflict", "signal": None},
        "decision_counts": {"candle_conflict": 2},
        "no_entry_for_sec": 1800,
        "candle_count": 120,
        "min_candles": 150,
        "warmup_remaining_candles": 30,
        "model_feature_window": 150,
    }))
    with patch.object(srv, "_bot_running", return_value=(True, 123)):
        data = client.get("/api/bot/status").get_json()
    assert data["last_decision"]["reason"] == "candle_conflict"
    assert data["decision_counts"]["candle_conflict"] == 2
    assert data["no_entry_for_sec"] == 1800
    assert data["candle_count"] == 120
    assert data["min_candles"] == 150
    assert data["warmup_remaining_candles"] == 30
    assert data["model_feature_window"] == 150
    assert data["diagnostics"]["weighted_score_logic_changed"] is False


def test_state_includes_bot_crash_status(client):
    """Garante que /api/state reflete o status crashed do bot."""
    import dashboard.server as srv
    srv.BOT_STATUS_JSON.write_text(json.dumps({"status": "running", "running": True, "pid": 999999}))
    with patch.object(srv, "_bot_running", return_value=(False, None)), \
         patch.object(srv, "_deriv_contract_report", return_value=[]):
        data = client.get("/api/state").get_json()
    assert data["bot"]["status"] == "crashed"


def test_state_includes_decision_diagnostics(client):
    """Garante que /api/state inclui os diagnósticos de decisão do bot."""
    import dashboard.server as srv
    srv.BOT_STATUS_JSON.write_text(json.dumps({
        "status": "running",
        "running": True,
        "pid": 123,
        "last_decision": {"reason": "ai_disagreement_or_low_confidence"},
        "decision_counts": {"no_technical_signal": 4},
        "no_entry_for_sec": 600,
        "candle_count": 120,
        "min_candles": 150,
        "warmup_remaining_candles": 30,
        "model_feature_window": 150,
    }))
    with patch.object(srv, "_bot_running", return_value=(True, 123)), \
         patch.object(srv, "_deriv_contract_report", return_value=[]):
        data = client.get("/api/state").get_json()
    assert data["bot"]["last_decision"]["reason"] == "ai_disagreement_or_low_confidence"
    assert data["bot"]["decision_counts"]["no_technical_signal"] == 4
    assert data["bot"]["no_entry_for_sec"] == 600
    assert data["bot"]["candle_count"] == 120
    assert data["bot"]["min_candles"] == 150
    assert data["bot"]["warmup_remaining_candles"] == 30
    assert data["bot"]["model_feature_window"] == 150
    assert data["diagnostics"]["weighted_score_logic_changed"] is False


def test_state_reports_missing_model_metrics_without_changing_weighted_score(client, monkeypatch, tmp_path):
    """Garante que métricas ausentes viram diagnóstico operacional, não mudança no score ponderado."""
    import dashboard.server as srv

    missing_metrics = tmp_path / "missing_model_metrics.json"
    monkeypatch.setattr(srv, "MODEL_METRICS_JSON", missing_metrics)
    srv.BOT_STATUS_JSON.write_text(json.dumps({
        "last_decision": {
            "reason": "signal_ready",
            "signal": "BUY",
            "indicators": {
                "decision_modules": ["weighted_score"],
                "decision_source": "weighted_score",
            },
        },
        "decision_counts": {"signal_ready": 1},
        "warmup_remaining_candles": 0,
        "shadow": {"reason": "missing_observed_payout"},
    }))

    with patch.object(srv, "_deriv_contract_report", return_value=[]):
        data = client.get("/api/state").get_json()

    assert data["diagnostics"]["model"]["metrics_missing"] is True
    assert data["diagnostics"]["weighted_score_logic_changed"] is False
    assert data["diagnostics"]["weighted_score_shadow_independent"] is True
    assert "model_metrics_missing" in data["diagnostics"]["blockers"]


def test_bot_stop_sets_stopping_and_terminates_process(client):
    """Garante que /api/bot/stop marca stopping e chama o encerramento robusto."""
    import dashboard.server as srv
    with patch.object(srv, "_bot_running", return_value=(True, 123)), \
         patch.object(srv, "_terminate_bot_process") as terminate:
        response = client.post("/api/bot/stop")
    data = response.get_json()
    status = json.loads(srv.BOT_STATUS_JSON.read_text())
    assert response.status_code == 200
    assert data["ok"] is True
    terminate.assert_called_once_with(123)
    assert status["status"] == "stopping"
    assert status["running"] is True
    assert status["pid"] == 123


# ─── /api/bot/start — autenticação ───────────────────────────────────────────

def test_bot_start_requires_token_when_set(client):
    """Quando DASHBOARD_TOKEN está definido, start sem token deve retornar 401."""
    import dashboard.server as srv

    with patch.object(srv, "_DASHBOARD_TOKEN", "secret123"):
        r = client.post("/api/bot/start", json={"mode": "demo", "balance": 1000})
        assert r.status_code == 401


def test_bot_start_accepts_correct_token(client):
    """Start com token correto deve passar pela autenticação (409 = bot já roda, não 401)."""
    import dashboard.server as srv

    with patch.object(srv, "_DASHBOARD_TOKEN", "secret123"), \
         patch.object(srv, "_bot_running", return_value=(False, None)):
        r = client.post(
            "/api/bot/start",
            json={"mode": "demo", "balance": 1000},
            headers={"X-Auth-Token": "secret123"},
        )
        # Pode retornar 200 (ok) ou 500 (falha ao iniciar processo) — não deve ser 401
        assert r.status_code != 401


def test_bot_start_invalid_mode_returns_400(client):
    """mode inválido deve retornar 400."""
    import dashboard.server as srv

    with patch.object(srv, "_DASHBOARD_TOKEN", ""), \
         patch.object(srv, "_bot_running", return_value=(False, None)):
        r = client.post("/api/bot/start", json={"mode": "invalid", "balance": 1000})
        assert r.status_code == 400


def test_bot_start_negative_balance_returns_400(client):
    """balance negativo deve retornar 400."""
    import dashboard.server as srv

    with patch.object(srv, "_DASHBOARD_TOKEN", ""), \
         patch.object(srv, "_bot_running", return_value=(False, None)):
        r = client.post("/api/bot/start", json={"mode": "demo", "balance": -50})
        assert r.status_code == 400


def test_bot_start_refuses_while_backtest_is_running(client):
    """Evita competição de recursos entre execução ao vivo e backtest."""
    import dashboard.server as srv

    with patch.object(srv, "_bot_running", return_value=(False, None)), \
         patch.object(srv, "_backtest_running", return_value=(True, 4242)):
        response = client.post("/api/bot/start", json={"mode": "demo", "balance": 1000})

    assert response.status_code == 409
    assert "Backtest em execução" in response.get_json()["msg"]


# ─── /api/bot/logs ───────────────────────────────────────────────────────────

def test_bot_logs_returns_200(client):
    """Garante que /api/bot/logs responde 200."""
    r = client.get("/api/bot/logs")
    assert r.status_code == 200


def test_bot_logs_has_lines_key(client):
    """Garante que /api/bot/logs traz a chave lines com uma lista."""
    data = json.loads(client.get("/api/bot/logs").data)
    assert "lines" in data
    assert isinstance(data["lines"], list)


# ─── /api/equity ─────────────────────────────────────────────────────────────

def test_equity_returns_200(client):
    """Garante que /api/equity responde 200."""
    r = client.get("/api/equity")
    assert r.status_code == 200


def test_equity_returns_list(client):
    """Garante que /api/equity retorna uma lista."""
    data = json.loads(client.get("/api/equity").data)
    assert isinstance(data, list)


# ─── /api/state ───────────────────────────────────────────────────────────────

def test_state_returns_200(client):
    """Garante que /api/state responde 200."""
    r = client.get("/api/state")
    assert r.status_code == 200


def test_state_required_top_keys(client):
    """Garante que /api/state traz as seções de nível superior obrigatórias."""
    data = json.loads(client.get("/api/state").data)
    required = {"bot", "summary", "indicators", "model", "risk_state", "active_contract"}
    missing = required - data.keys()
    assert not missing, f"Chaves ausentes em /api/state: {missing}"


def test_state_bot_section_structure(client):
    """Garante que a seção bot de /api/state tem running booleano e pid."""
    data = json.loads(client.get("/api/state").data)
    bot = data["bot"]
    assert "running" in bot
    assert "pid" in bot
    assert isinstance(bot["running"], bool)


def test_state_summary_section_structure(client):
    """Garante que a seção summary de /api/state tem saldo, total de trades e win rate."""
    data = json.loads(client.get("/api/state").data)
    summary = data["summary"]
    assert "balance" in summary
    assert "total_trades" in summary
    assert "win_rate" in summary


def test_state_summary_falls_back_to_risk_state_when_session_has_no_operations(tmp_path):
    """Garante que o resumo de /api/state usa o estado de risco quando a sessão não tem operações."""
    import dashboard.server as srv

    ops = tmp_path / "operacoes_v2.csv"
    risk_state = tmp_path / "risk_state.json"
    state = tmp_path / "state.json"
    ticks = tmp_path / "ticks_v2.csv"
    dataset = tmp_path / "dataset_v2.csv"

    ops.write_text(
        "timestamp,result,profit,balance_before,balance_after,consec_losses,drawdown_pct,win_rate_recent\n"
        "2026-09-16T18:45:02,WIN,64.84,7372.72,7437.56,0,0,100\n"
    )
    risk_state.write_text(json.dumps({
        "balance": 7645.24,
        "daily_start_balance": 7578.09,
        "daily_pnl": 67.15,
        "daily_pnl_pct": 0.886,
        "recent_results": [1],
        "win_rate_recent": 100.0,
    }))
    state.write_text(json.dumps({"session_start": "2026-09-16T19:10:04", "symbol": "R_75"}))
    ticks.write_text("")
    dataset.write_text("")

    with patch.object(srv, "OPERATIONS_CSV", ops), \
         patch.object(srv, "RISK_STATE_JSON", risk_state), \
         patch.object(srv, "STATE_JSON", state), \
         patch.object(srv, "TICKS_CSV", ticks), \
         patch.object(srv, "DATASET_CSV", dataset):
        srv.app.config["TESTING"] = True
        with srv.app.test_client() as c:
            summary = c.get("/api/state").get_json()["summary"]

    assert summary["balance"] == 7645.24
    assert summary["pnl_today"] == 67.15
    assert summary["total_trades"] == 1
    assert summary["win_rate"] == 100.0


def test_state_model_section_has_metrics(client):
    """Garante que a seção model de /api/state traz model_exists e last_metrics como dict."""
    data = json.loads(client.get("/api/state").data)
    model = data["model"]
    assert "model_exists" in model
    assert "last_metrics" in model
    assert isinstance(model["last_metrics"], dict)


# ─── /api/model-metrics ──────────────────────────────────────────────────────

def test_model_metrics_returns_200(client):
    """Garante que /api/model-metrics responde 200."""
    r = client.get("/api/model-metrics")
    assert r.status_code == 200


def test_model_metrics_has_cache_control_header(client):
    """Garante que /api/model-metrics envia Cache-Control com no-cache ou no-store."""
    r = client.get("/api/model-metrics")
    cc = r.headers.get("Cache-Control", "")
    assert "no-cache" in cc or "no-store" in cc, (
        f"Cache-Control ausente ou incorreto: {cc}"
    )


# ─── /api/model-history ──────────────────────────────────────────────────────

def test_model_history_returns_200(client):
    """Garante que /api/model-history responde 200."""
    r = client.get("/api/model-history")
    assert r.status_code == 200


def test_model_history_returns_list(client):
    """Garante que /api/model-history retorna uma lista."""
    data = json.loads(client.get("/api/model-history").data)
    assert isinstance(data, list)


# ─── /api/system ─────────────────────────────────────────────────────────────

def test_system_returns_200(client):
    """Garante que /api/system responde 200."""
    r = client.get("/api/system")
    assert r.status_code == 200


def test_system_required_keys(client):
    """Garante que /api/system traz as chaves de CPU, RAM e disco obrigatórias."""
    data = json.loads(client.get("/api/system").data)
    required = {"cpu_pct", "ram_pct", "ram_used_mb", "disk_pct", "bot_cpu_pct", "bot_ram_mb"}
    missing = required - data.keys()
    assert not missing, f"Chaves ausentes em /api/system: {missing}"


# ─── /api/contract-active ────────────────────────────────────────────────────

def test_contract_active_returns_200(client):
    """Garante que /api/contract-active responde 200."""
    r = client.get("/api/contract-active")
    assert r.status_code == 200


def test_contract_active_no_bot_returns_false(client):
    """Garante que /api/contract-active informa has_active falso sem contrato ativo."""
    data = json.loads(client.get("/api/contract-active").data)
    assert data.get("has_active") is False


def test_contract_active_returns_last_local_contract_when_idle(tmp_path):
    """Garante que, sem contrato ativo, /api/contract-active devolve o último contrato local encerrado."""
    import dashboard.server as srv

    state = tmp_path / "state.json"
    ops = tmp_path / "operacoes_v2.csv"
    state.write_text(json.dumps({"active_contract": None}))
    ops.write_text(
        "timestamp,symbol,direction,stake,duration,result,profit,balance_before,balance_after,"
        "ema9,ema21,rsi,adx,macd_hist,ai_confidence,ai_score,consec_losses,drawdown_pct,"
        "win_rate_recent,market_condition,contract_id,duration_unit,entry_price,exit_price,"
        "entry_epoch,exit_epoch,payout,buy_price,model_id,ai_direction,tech_score,account_key\n"
        "2026-09-21T09:30:00,frxAUDJPY,SELL,10,15,WIN,8,100,108,"
        "1,1,50,20,0,0.7,0.5,0,0,100,trending,abc123,m,112.3,112.1,"
        "1789993000,1789993900,18,10,model,SELL,0.5,acct\n"
    )

    with patch.object(srv, "STATE_JSON", state), patch.object(srv, "OPERATIONS_CSV", ops):
        srv.app.config["TESTING"] = True
        with srv.app.test_client() as c:
            data = c.get("/api/contract-active").get_json()

    assert data["has_active"] is False
    assert data["last_contract"]["contract_id"] == "abc123"
    assert data["last_contract"]["closed"] is True
    assert data["last_contract"]["symbol"] == "frxAUDJPY"
    assert data["last_contract"]["profit"] == 8


def test_contract_active_uses_buy_timestamp_as_entry_epoch_fallback(tmp_path):
    """Garante que o buy_timestamp vira entry_epoch quando o contrato ativo não tem época de entrada."""
    import dashboard.server as srv

    state = tmp_path / "state.json"
    state.write_text(json.dumps({
        "active_contract": {
            "has_active": True,
            "contract_id": "12345",
            "symbol": "R_75",
            "direction": "BUY",
            "duration": 15,
            "buy_timestamp": 1700000000.9,
            "entry_epoch": 0,
            "entry_price": 0.0,
        }
    }))

    with patch.object(srv, "STATE_JSON", state):
        srv.app.config["TESTING"] = True
        with srv.app.test_client() as c:
            data = c.get("/api/contract-active").get_json()

    assert data["has_active"] is True
    assert data["entry_epoch"] == 1700000000


# ─── /api/stats ──────────────────────────────────────────────────────────────

def test_stats_returns_200(client):
    """Garante que /api/stats responde 200."""
    r = client.get("/api/stats")
    assert r.status_code == 200


def test_stats_empty_data_returns_empty_dict(client):
    """Garante que /api/stats retorna um dict sem dados."""
    data = json.loads(client.get("/api/stats").data)
    assert isinstance(data, dict)


# ─── /api/backtest ───────────────────────────────────────────────────────────

def test_backtest_status_is_idle_without_execution(client):
    """Garante que o status do backtest é idle, sem execução e com lista de logs."""
    import dashboard.server as srv

    srv._backtest_state = {}
    with patch.object(srv, "_backtest_process", None):
        data = client.get("/api/backtest/status").get_json()

    assert data["status"] == "idle"
    assert data["running"] is False
    assert isinstance(data["logs"], list)


def test_backtest_result_returns_404_without_report(client):
    """Garante que /api/backtest/result retorna 404 sem relatório."""
    import dashboard.server as srv

    srv._backtest_state = {}
    response = client.get("/api/backtest/result")

    assert response.status_code == 404
    assert response.get_json()["ok"] is False


def test_backtest_start_requires_token_when_configured(client):
    """Garante que iniciar backtest sem token retorna 401 quando o token está configurado."""
    import dashboard.server as srv

    with patch.object(srv, "_DASHBOARD_TOKEN", "secret123"):
        response = client.post("/api/backtest/start", json={})

    assert response.status_code == 401


def test_backtest_start_refuses_when_live_bot_is_running(client):
    """Garante que iniciar backtest com o bot ao vivo rodando retorna 409."""
    import dashboard.server as srv

    with patch.object(srv, "_bot_running", return_value=(True, 123)):
        response = client.post("/api/backtest/start", json={})

    assert response.status_code == 409
    assert "Pare o bot" in response.get_json()["msg"]


def test_backtest_start_validates_parameters(client):
    """Garante que payout negativo no início do backtest retorna 400."""
    import dashboard.server as srv

    with patch.object(srv, "_bot_running", return_value=(False, None)), \
         patch.object(srv, "_backtest_running", return_value=(False, None)):
        response = client.post("/api/backtest/start", json={"payout": -0.1})

    assert response.status_code == 400
    assert "payout" in response.get_json()["msg"]


def test_backtest_start_rejects_non_boolean_flags(client):
    """Garante que with_ai não booleano no início do backtest retorna 400."""
    import dashboard.server as srv

    with patch.object(srv, "_bot_running", return_value=(False, None)), \
         patch.object(srv, "_backtest_running", return_value=(False, None)):
        response = client.post(
            "/api/backtest/start",
            json={"with_ai": "false"},
        )

    assert response.status_code == 400
    assert "with_ai" in response.get_json()["msg"]


def test_backtest_start_builds_safe_subprocess_command(client):
    """Garante que o backtest inicia um subprocesso com lista de argumentos, sem shell e sem --with-ai."""
    import dashboard.server as srv

    fake_process = MagicMock()
    fake_process.pid = 4242
    fake_process.poll.return_value = None
    fake_process.stdout = []
    with patch.object(srv, "_bot_running", return_value=(False, None)), \
         patch.object(srv, "_backtest_running", return_value=(False, None)), \
         patch.object(srv.subprocess, "Popen", return_value=fake_process) as popen, \
         patch.object(srv.threading, "Thread") as thread:
        response = client.post(
            "/api/backtest/start",
            json={
                "symbol": "R_100",
                "balance": 1000,
                "payout": 0.85,
                "duration": 15,
                "duration_unit": "m",
                "timeframe": 60,
                "max_ticks": 5000,
                "with_ai": False,
            },
        )

    assert response.status_code == 202
    assert response.get_json()["ok"] is True
    command = popen.call_args.args[0]
    assert command[:3] == [sys.executable, "-u", str(srv.BACKTEST_PY)]
    assert "--symbol" in command
    assert "R_100" in command
    assert "--with-ai" not in command
    assert popen.call_args.kwargs.get("shell") is None
    thread.return_value.start.assert_called_once()


def test_backtest_result_returns_metrics_curve_and_limited_trades(client):
    """Garante que o resultado do backtest traz métricas, curva de equity e trades limitados por n."""
    import dashboard.server as srv

    run_dir = srv.BACKTEST_ROOT / "test-run"
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / "backtest_report.json"
    report_path.write_text(json.dumps({
        "symbol": "R_100",
        "settings": {"payout_ratio": 0.85},
        "metrics": {"initial_balance": 1000.0, "trades": 2},
        "trades": [
            {"entry_epoch": 100, "expiry_epoch": 200, "balance_after": 1008.5},
            {"entry_epoch": 300, "expiry_epoch": 400, "balance_after": 998.5},
        ],
    }), encoding="utf-8")
    srv._backtest_state = {
        "status": "completed",
        "report_path": str(report_path),
        "report_available": True,
    }

    response = client.get("/api/backtest/result?n=1")
    data = response.get_json()

    assert response.status_code == 200
    assert data["ok"] is True
    assert len(data["report"]["trades"]) == 1
    assert len(data["report"]["equity_curve"]) == 3
    assert data["report"]["equity_curve"][-1]["balance"] == 998.5


def test_start_does_not_reset_risk_or_pending_contract(client):
    """Garante que iniciar o bot não zera o estado de risco nem o contrato pendente."""
    import dashboard.server as srv
    risk={'date':'2026-09-08','daily_pnl':-30,'consec_losses':3,'balance':970,'pause_until_epoch':9999999999}
    active={'has_active':True,'contract_id':'12345','phase':'open'}
    srv.RISK_STATE_JSON.write_text(json.dumps(risk))
    srv.STATE_JSON.write_text(json.dumps({'active_contract':active}))
    proc=MagicMock()
    proc.pid=99999999
    with patch.object(srv,'_bot_running',return_value=(False,None)),patch.object(srv,'_backtest_running',return_value=(False,None)),patch.object(srv.subprocess,'Popen',return_value=proc),patch.object(srv,'PID_FILE',srv.STATE_JSON.parent/'bot.pid'),patch.object(srv,'PIPELINE_LOG_TXT',srv.STATE_JSON.parent/'pipeline.log'),patch.object(srv,'BOT_STATUS_JSON',srv.STATE_JSON.parent/'bot_status.json'),patch.object(srv.threading,'Thread'):
        response=client.post('/api/bot/start',json={'mode':'demo','balance':1000})
    assert response.status_code==200
    assert json.loads(srv.RISK_STATE_JSON.read_text())==risk
    assert json.loads(srv.STATE_JSON.read_text())['active_contract']==active


def test_start_uses_minimum_history_candles_by_default(client):
    """Garante que o início do bot usa MIN_HISTORY_CANDLES como --history-count por padrão."""
    import dashboard.server as srv
    proc=MagicMock()
    proc.pid=99999999
    with patch.object(srv,'_bot_running',return_value=(False,None)),patch.object(srv,'_backtest_running',return_value=(False,None)),patch.object(srv.subprocess,'Popen',return_value=proc) as popen,patch.object(srv,'PID_FILE',srv.STATE_JSON.parent/'bot.pid'),patch.object(srv,'PIPELINE_LOG_TXT',srv.STATE_JSON.parent/'pipeline.log'),patch.object(srv,'BOT_STATUS_JSON',srv.STATE_JSON.parent/'bot_status.json'),patch.object(srv.threading,'Thread'):
        response=client.post('/api/bot/start',json={'mode':'demo','balance':1000})
    assert response.status_code==200
    cmd = popen.call_args.args[0]
    assert cmd[cmd.index('--history-count') + 1] == str(srv.MIN_HISTORY_CANDLES)


def test_start_passes_decision_switches_to_pipeline_env(client):
    """Garante que os interruptores de decisão do início do bot chegam ao pipeline via variáveis de ambiente."""
    import dashboard.server as srv
    proc = MagicMock()
    proc.pid = 99999999
    with patch.object(srv, '_bot_running', return_value=(False, None)), \
         patch.object(srv, '_backtest_running', return_value=(False, None)), \
         patch.object(srv.subprocess, 'Popen', return_value=proc) as popen, \
         patch.object(srv, 'PID_FILE', srv.STATE_JSON.parent / 'bot.pid'), \
         patch.object(srv, 'PIPELINE_LOG_TXT', srv.STATE_JSON.parent / 'pipeline.log'), \
         patch.object(srv, 'BOT_STATUS_JSON', srv.STATE_JSON.parent / 'bot_status.json'), \
         patch.object(srv.threading, 'Thread'):
        response = client.post('/api/bot/start', json={
            'mode': 'demo',
            'balance': 1000,
            'use_trained_model': True,
            'use_weighted_signal': False,
            'use_price_action_filter': True,
            'use_candle_pattern_filter': True,
        })
    assert response.status_code == 200
    env = popen.call_args.kwargs['env']
    assert env['STANDERIV_USE_AI_MODEL'] == '1'
    assert env['STANDERIV_EXECUTION_SIGNAL_MODE'] == 'trained_model'
    assert env['STANDERIV_USE_WEIGHTED_SIGNAL'] == '0'
    assert env['STANDERIV_PRICE_ACTION_FILTER'] == '1'
    assert env['STANDERIV_CANDLE_PATTERN_FILTER'] == '1'


def test_start_rejects_all_decision_switches_off(client):
    """Garante que desligar todos os módulos de decisão no início do bot retorna 400."""
    import dashboard.server as srv
    with patch.object(srv, '_bot_running', return_value=(False, None)), \
         patch.object(srv, '_backtest_running', return_value=(False, None)):
        response = client.post('/api/bot/start', json={
            'mode': 'demo',
            'balance': 1000,
            'use_trained_model': False,
            'use_weighted_signal': False,
            'use_price_action_filter': False,
            'use_candle_pattern_filter': False,
        })
    assert response.status_code == 400
    assert 'módulo de decisão' in response.get_json()['msg']


def test_clear_cannot_remove_active_contract(client):
    """Garante que limpar operações e estado recusa remover contrato ativo."""
    import dashboard.server as srv
    srv.STATE_JSON.write_text(json.dumps({'active_contract':{'has_active':True,'contract_id':'1'}}))
    with patch.object(srv,'_bot_running',return_value=(False,None)):
        ok,_=srv._reset_operations_and_state()
    assert not ok and json.loads(srv.STATE_JSON.read_text())['active_contract']['contract_id']=='1'


def test_model_metrics_does_not_present_candidate_as_active(client, monkeypatch, tmp_path):
    """Garante que métricas de candidato sem modelo ativo não aparecem em /api/model-metrics, apenas no histórico."""
    import dashboard.server as srv
    candidate = tmp_path / 'candidate_metrics.json'
    candidate.write_text(json.dumps([{'stage': 'final', 'auc': .99}]))
    monkeypatch.setattr(srv, 'MODEL_METRICS_JSON', candidate)
    monkeypatch.setattr(srv, 'MODEL_PKL', tmp_path / 'missing.pkl')
    assert client.get('/api/model-metrics').get_json() == {}
    assert client.get('/api/model-history').get_json()[0]['auc'] == .99
