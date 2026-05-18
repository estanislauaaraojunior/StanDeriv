import json
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


API_BASE = "https://api.derivws.com"
BEARER_TOKEN_PREFIXES = ("pat_", "ory_at_")

# User-Agent que passa pelo WAF da Cloudflare (Python-urllib é bloqueado).
_USER_AGENT = "Mozilla/5.0 (compatible; DerivBot/2.0; +https://developers.deriv.com)"


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
    if mode in ("legacy", "none", "ws", "websocket"):
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
    method = method.upper()
    body = None
    headers = {
        "Authorization": f"Bearer {token}",
        "Deriv-App-ID": str(app_id),
        "User-Agent": _USER_AGENT,
        "Accept": "application/json",
    }
    # Content-Type só em requisições com corpo; GET sem body não deve enviá-lo
    # pois o WAF da Cloudflare pode rejeitar GETs com Content-Type.
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = Request(
        f"{API_BASE}{path}",
        data=body,
        headers=headers,
        method=method,
    )

    try:
        with urlopen(req, timeout=timeout_sec) as response:
            raw = response.read().decode("utf-8", errors="replace")
            return json.loads(raw) if raw else {}
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace").strip()
        message = raw or f"HTTP {exc.code}"
        code = ""
        try:
            parsed = json.loads(raw)
            # Formato novo Deriv: {"errors": [{"code": "...", "message": "..."}]}
            errors = parsed.get("errors") or []
            if errors:
                first = errors[0]
                code = str(first.get("code", ""))
                message = str(first.get("message", message))
            # Formato alternativo: {"error": "...", "message": "..."}
            elif isinstance(parsed.get("error"), str):
                code = str(parsed["error"])
                message = str(parsed.get("message", code))
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
    last_err: DerivAPIError | None = None
    for attempt in range(3):
        try:
            data = request_json(
                "GET",
                "/trading/v1/options/accounts",
                app_id=app_id,
                token=token,
                timeout_sec=timeout_sec,
            )
            return _extract_accounts(data)
        except DerivAPIError as exc:
            last_err = exc
            # 429 / Cloudflare 1015 (rate limit) — backoff exponencial
            is_rate_limit = (
                exc.status in (429,)
                or "1015" in str(exc)
                or str(exc.code).lower() in ("ratelimit", "rate_limit")
            )
            if is_rate_limit and attempt < 2:
                wait = 2 ** (attempt + 1)  # 2s, 4s
                time.sleep(wait)
                continue
            raise
    raise last_err or DerivAPIError("get_options_accounts: max retries exceeded")


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
    attempts: int = 3,
) -> str:
    last_error: DerivAPIError | None = None
    max_attempts = max(1, int(attempts or 1))

    for attempt in range(max_attempts):
        try:
            data = request_json(
                "POST",
                f"/trading/v1/options/accounts/{account_id}/otp",
                app_id=app_id,
                token=token,
                timeout_sec=timeout_sec,
            )
            break
        except DerivAPIError as exc:
            last_error = exc
            if attempt >= max_attempts - 1:
                raise
            time.sleep(0.5 * (attempt + 1))
    else:
        raise last_error or DerivAPIError("Unable to obtain Options WebSocket URL")

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
