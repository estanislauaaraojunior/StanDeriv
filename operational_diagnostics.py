"""Diagnósticos operacionais sem URLs de sessão ou credenciais."""
import os
import re
import time
from datetime import datetime, timezone


def safe_detail(value):
    """Remove segredos conhecidos e campos sensíveis antes de registrar texto externo."""
    text = str(value or '')
    for key, secret in os.environ.items():
        if any(part in key.upper() for part in ('TOKEN', 'PASSWORD', 'SECRET', 'API_KEY')) and secret:
            text = text.replace(secret, '[REDACTED]')
    text = re.sub(r'(?i)\b(?:https?|wss?)://\S+', '[URL REDACTED]', text)
    text = re.sub(r'(?i)\bBearer\s+\S+', 'Bearer [REDACTED]', text)
    text = re.sub(r'\b(?:CR|VRTC|DOT|MF|MLT)\d+\b', '[ACCOUNT REDACTED]', text)
    text = re.sub(r'''(?ix)(["']?(?:\w*token\w*|authorization|password|secret|otp|loginid)["']?\s*[:=]\s*)("[^"]*"|'[^']*'|[^\s,;]+)''', r'\1[REDACTED]', text)
    return ' '.join(text.split())[:400]


def connection_diagnostic(opened_at, *, error=None, code=None, reason=None):
    """Inclui horário UTC e duração monotônica; distingue conexão ainda não aberta."""
    stamp = datetime.now(timezone.utc).isoformat(timespec='seconds')
    duration = 'não aberta' if opened_at is None else f'{max(0, time.monotonic() - opened_at):.1f}s'
    detail = f'horário={stamp} duração={duration}'
    if error is not None:
        detail += f' tipo={type(error).__name__} causa={safe_detail(error)}'
    else:
        detail += f' código={safe_detail(code)} motivo={safe_detail(reason) or "sem motivo informado"}'
    return detail
