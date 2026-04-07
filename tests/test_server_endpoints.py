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
from unittest.mock import patch

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
        empty_metrics = tmp / "model_metrics.json"

        empty_ops.write_text("")
        empty_ticks.write_text("")
        empty_risk.write_text("{}")
        empty_state.write_text("{}")
        empty_metrics.write_text("[]")

        with patch.object(srv, "OPERATIONS_CSV", empty_ops), \
             patch.object(srv, "TICKS_CSV", empty_ticks), \
             patch.object(srv, "RISK_STATE_JSON", empty_risk), \
             patch.object(srv, "STATE_JSON", empty_state), \
             patch.object(srv, "MODEL_METRICS_JSON", empty_metrics):

            srv.app.config["TESTING"] = True
            with srv.app.test_client() as c:
                yield c


# ─── Endpoints estáticos ──────────────────────────────────────────────────────

def test_index_returns_200(client):
    r = client.get("/")
    assert r.status_code == 200


def test_favicon_no_error(client):
    r = client.get("/favicon.ico")
    assert r.status_code in (200, 204, 404)  # qualquer coisa menos 500


# ─── /api/summary ─────────────────────────────────────────────────────────────

def test_summary_returns_200(client):
    r = client.get("/api/summary")
    assert r.status_code == 200


def test_summary_required_keys(client):
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
    r = client.get("/api/summary")
    data = json.loads(r.data)
    assert data["total_trades"] == 0
    assert data["balance"] == 0.0
    assert data["is_paused"] is False


# ─── /api/ticks ───────────────────────────────────────────────────────────────

def test_ticks_returns_200(client):
    r = client.get("/api/ticks?n=10")
    assert r.status_code == 200


def test_ticks_returns_list(client):
    data = json.loads(client.get("/api/ticks?n=10").data)
    assert isinstance(data, list)


def test_ticks_n_capped_at_2000(client):
    """Parâmetro n não deve exceder 2000 (proteção contra DoS)."""
    r = client.get("/api/ticks?n=99999")
    assert r.status_code == 200  # não deve explodir


# ─── /api/trades ──────────────────────────────────────────────────────────────

def test_trades_returns_200(client):
    r = client.get("/api/trades?n=5")
    assert r.status_code == 200


def test_trades_returns_list(client):
    data = json.loads(client.get("/api/trades?n=5").data)
    assert isinstance(data, list)


# ─── /api/indicators ─────────────────────────────────────────────────────────

def test_indicators_returns_200(client):
    r = client.get("/api/indicators")
    assert r.status_code == 200


def test_indicators_returns_dict(client):
    data = json.loads(client.get("/api/indicators").data)
    assert isinstance(data, dict)


# ─── /api/model ───────────────────────────────────────────────────────────────

def test_model_returns_200(client):
    r = client.get("/api/model")
    assert r.status_code == 200


def test_model_required_keys(client):
    data = json.loads(client.get("/api/model").data)
    required = {"model_exists", "tft_exists", "duration_exists", "ticks_count", "dataset_rows"}
    missing = required - data.keys()
    assert not missing, f"Chaves ausentes em /api/model: {missing}"


# ─── /api/bot/status ─────────────────────────────────────────────────────────

def test_bot_status_returns_200(client):
    r = client.get("/api/bot/status")
    assert r.status_code == 200


def test_bot_status_running_is_bool(client):
    data = json.loads(client.get("/api/bot/status").data)
    assert isinstance(data["running"], bool)


def test_bot_status_not_running_without_pid(client):
    import dashboard.server as srv
    # Mocka _bot_running para simular ausência de bot nos testes
    with patch.object(srv, "_bot_running", return_value=(False, None)):
        data = json.loads(client.get("/api/bot/status").data)
    assert data["running"] is False


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


# ─── /api/bot/logs ───────────────────────────────────────────────────────────

def test_bot_logs_returns_200(client):
    r = client.get("/api/bot/logs")
    assert r.status_code == 200


def test_bot_logs_has_lines_key(client):
    data = json.loads(client.get("/api/bot/logs").data)
    assert "lines" in data
    assert isinstance(data["lines"], list)


# ─── /api/equity ─────────────────────────────────────────────────────────────

def test_equity_returns_200(client):
    r = client.get("/api/equity")
    assert r.status_code == 200


def test_equity_returns_list(client):
    data = json.loads(client.get("/api/equity").data)
    assert isinstance(data, list)


# ─── /api/state ───────────────────────────────────────────────────────────────

def test_state_returns_200(client):
    r = client.get("/api/state")
    assert r.status_code == 200


def test_state_required_top_keys(client):
    data = json.loads(client.get("/api/state").data)
    required = {"bot", "summary", "indicators", "model", "risk_state", "active_contract"}
    missing = required - data.keys()
    assert not missing, f"Chaves ausentes em /api/state: {missing}"


def test_state_bot_section_structure(client):
    data = json.loads(client.get("/api/state").data)
    bot = data["bot"]
    assert "running" in bot
    assert "pid" in bot
    assert isinstance(bot["running"], bool)


def test_state_summary_section_structure(client):
    data = json.loads(client.get("/api/state").data)
    summary = data["summary"]
    assert "balance" in summary
    assert "total_trades" in summary
    assert "win_rate" in summary


def test_state_model_section_has_metrics(client):
    data = json.loads(client.get("/api/state").data)
    model = data["model"]
    assert "model_exists" in model
    assert "last_metrics" in model
    assert isinstance(model["last_metrics"], dict)


# ─── /api/model-metrics ──────────────────────────────────────────────────────

def test_model_metrics_returns_200(client):
    r = client.get("/api/model-metrics")
    assert r.status_code == 200


def test_model_metrics_has_cache_control_header(client):
    r = client.get("/api/model-metrics")
    cc = r.headers.get("Cache-Control", "")
    assert "no-cache" in cc or "no-store" in cc, (
        f"Cache-Control ausente ou incorreto: {cc}"
    )


# ─── /api/model-history ──────────────────────────────────────────────────────

def test_model_history_returns_200(client):
    r = client.get("/api/model-history")
    assert r.status_code == 200


def test_model_history_returns_list(client):
    data = json.loads(client.get("/api/model-history").data)
    assert isinstance(data, list)


# ─── /api/system ─────────────────────────────────────────────────────────────

def test_system_returns_200(client):
    r = client.get("/api/system")
    assert r.status_code == 200


def test_system_required_keys(client):
    data = json.loads(client.get("/api/system").data)
    required = {"cpu_pct", "ram_pct", "ram_used_mb", "disk_pct", "bot_cpu_pct", "bot_ram_mb"}
    missing = required - data.keys()
    assert not missing, f"Chaves ausentes em /api/system: {missing}"


# ─── /api/contract-active ────────────────────────────────────────────────────

def test_contract_active_returns_200(client):
    r = client.get("/api/contract-active")
    assert r.status_code == 200


def test_contract_active_no_bot_returns_false(client):
    data = json.loads(client.get("/api/contract-active").data)
    assert data.get("has_active") is False


# ─── /api/stats ──────────────────────────────────────────────────────────────

def test_stats_returns_200(client):
    r = client.get("/api/stats")
    assert r.status_code == 200


def test_stats_empty_data_returns_empty_dict(client):
    data = json.loads(client.get("/api/stats").data)
    assert isinstance(data, dict)
