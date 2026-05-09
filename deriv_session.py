import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


API_BASE = "https://api.derivws.com"
BEARER_TOKEN_PREFIXES = ("pat_", "ory_at_")


class DerivAPIError(RuntimeError):
    def __init__(self, message: str, status: int | None = None, code: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.code = code


def is_bearer_token(token: str, auth_mode: str = "auto") -> bool:
    """Detecta tokens do fluxo novo da Deriv (PAT/OAuth Bearer).

    O modo `auto` preserva compatibilidade com tokens legados da WebSocket API.
    Use DERIV_AUTH_MODE=bearer para forçar REST+OTP quando o prefixo do token
    mudar ou não for reconhecido automaticamente.
    """
    mode = str(auth_mode or "auto").strip().lower()
    if mode in ("bearer", "pat", "oauth", "rest"):
        return True
    if mode == "legacy":
        return False
    return str(token).strip().startswith(BEARER_TOKEN_PREFIXES)


def is_pat_token(token: str) -> bool:
    return str(token).strip().startswith("pat_")


def request_json(
    method: str,
    path: str,
    *,
    app_id: str,
    token: str,
    payload: dict | None = None,
    timeout_sec: int = 15,
) -> dict:
    body = None
    headers = {
        "Authorization": f"Bearer {token}",
        "Deriv-App-ID": str(app_id),
        "Content-Type": "application/json",
    }
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")

    request = Request(
        f"{API_BASE}{path}",
        data=body,
        headers=headers,
        method=method.upper(),
    )

    try:
        with urlopen(request, timeout=timeout_sec) as response:
            raw = response.read().decode("utf-8", errors="replace")
            return json.loads(raw) if raw else {}
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace").strip()
        message = raw or f"HTTP {exc.code}"
        code = ""
        try:
            parsed = json.loads(raw)
            errors = parsed.get("errors") or []
            if errors:
                first = errors[0]
                code = str(first.get("code", ""))
                message = str(first.get("message", message))
        except Exception:
            pass
        raise DerivAPIError(message, status=exc.code, code=code) from exc
    except URLError as exc:
        raise DerivAPIError(f"Network error: {exc.reason}") from exc


def _extract_accounts(data: dict) -> list[dict]:
    raw = data.get("data") if isinstance(data, dict) else []
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        raw = data.get("accounts", []) if isinstance(data, dict) else []
    return [item for item in raw if isinstance(item, dict)]


def get_options_accounts(*, app_id: str, token: str, timeout_sec: int = 15) -> list[dict]:
    data = request_json(
        "GET",
        "/trading/v1/options/accounts",
        app_id=app_id,
        token=token,
        timeout_sec=timeout_sec,
    )
    return _extract_accounts(data)


def create_options_account(
    *,
    app_id: str,
    token: str,
    account_type: str,
    timeout_sec: int = 15,
) -> list[dict]:
    data = request_json(
        "POST",
        "/trading/v1/options/accounts",
        app_id=app_id,
        token=token,
        timeout_sec=timeout_sec,
        payload={"currency": "USD", "group": "row", "account_type": account_type},
    )
    return _extract_accounts(data)


def choose_options_account(accounts: list[dict], *, demo: bool) -> dict | None:
    desired_type = "demo" if demo else "real"
    candidates = [
        acc for acc in accounts
        if str(acc.get("account_type", "")).lower() == desired_type
    ]
    active = [
        acc for acc in candidates
        if str(acc.get("status", "")).lower() in ("", "active")
    ]
    pool = active or candidates
    return pool[0] if pool else None


def get_options_ws_url(
    account_id: str,
    *,
    app_id: str,
    token: str,
    timeout_sec: int = 15,
) -> str:
    data = request_json(
        "POST",
        f"/trading/v1/options/accounts/{account_id}/otp",
        app_id=app_id,
        token=token,
        timeout_sec=timeout_sec,
    )
    payload = data.get("data", {}) if isinstance(data, dict) else {}
    url = payload.get("url") if isinstance(payload, dict) else ""
    if not url:
        raise DerivAPIError("OTP response did not include a WebSocket URL")
    return str(url)


def setup_bearer_options_session(
    *,
    app_id: str,
    token: str,
    demo: bool,
    timeout_sec: int = 15,
) -> dict:
    account_type = "demo" if demo else "real"
    accounts = get_options_accounts(app_id=app_id, token=token, timeout_sec=timeout_sec)
    account = choose_options_account(accounts, demo=demo)

    if account is None:
        accounts = create_options_account(
            app_id=app_id,
            token=token,
            account_type=account_type,
            timeout_sec=timeout_sec,
        )
        account = choose_options_account(accounts, demo=demo)

    if account is None:
        raise DerivAPIError(f"No {account_type} Options account available for this token")

    account_id = str(account.get("account_id") or account.get("id") or "")
    if not account_id:
        raise DerivAPIError("Options account response did not include account_id")

    ws_url = get_options_ws_url(
        account_id,
        app_id=app_id,
        token=token,
        timeout_sec=timeout_sec,
    )
    return {"account": account, "account_id": account_id, "ws_url": ws_url}


def setup_pat_options_session(**kwargs) -> dict:
    """Compatibilidade com chamadas antigas; use setup_bearer_options_session."""
    return setup_bearer_options_session(**kwargs)
