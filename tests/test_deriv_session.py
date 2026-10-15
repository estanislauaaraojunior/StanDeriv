"""Regressões offline do fluxo REST/Bearer da Deriv."""

import deriv_session


def test_get_options_ws_url_backs_off_on_rate_limit(monkeypatch):
    """Garante backoff maior quando o OTP REST recebe HTTP 429."""
    calls = []
    sleeps = []

    def fake_request_json(*_args, **_kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise deriv_session.DerivAPIError(
                "Rate limit exceeded. Please try again later",
                status=429,
            )
        return {"data": {"url": "wss://example.test/options"}}

    monkeypatch.setattr(deriv_session, "request_json", fake_request_json)
    monkeypatch.setattr(deriv_session.time, "sleep", sleeps.append)

    assert deriv_session.get_options_ws_url(
        "demo-account",
        app_id="app",
        token="token",
        attempts=2,
    ) == "wss://example.test/options"
    assert sleeps == [5]


def test_get_options_ws_url_keeps_short_backoff_for_non_rate_limit(monkeypatch):
    """Garante que erros comuns preservam retry curto."""
    calls = []
    sleeps = []

    def fake_request_json(*_args, **_kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise deriv_session.DerivAPIError("temporary", status=500)
        return {"data": {"url": "wss://example.test/options"}}

    monkeypatch.setattr(deriv_session, "request_json", fake_request_json)
    monkeypatch.setattr(deriv_session.time, "sleep", sleeps.append)

    assert deriv_session.get_options_ws_url(
        "demo-account",
        app_id="app",
        token="token",
        attempts=2,
    ) == "wss://example.test/options"
    assert sleeps == [0.5]
