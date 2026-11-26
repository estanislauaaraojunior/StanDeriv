"""Atualização coordenada do status de exibição; não altera estado financeiro."""
import fcntl
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from model_contract import write_json_atomic

SESSION_ID = os.environ.get('STANDERIV_SESSION_ID') or uuid.uuid4().hex
_CANDLE_FIELDS = ('candle_count', 'min_candles', 'model_feature_window',
                  'warmup_remaining_candles', 'candle_source')


def update_status(path, updates, *, owner='', reset=False):
    """Serializa read/merge/write entre dashboard, pipeline e executor."""
    path = Path(path)
    with path.with_suffix(path.suffix + '.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            state = json.loads(path.read_text())
            if not isinstance(state, dict):
                state = {}
        except (OSError, ValueError):
            state = {}
        updates = dict(updates)
        new_session = any(
            key in updates and updates[key] is not None and state.get(key) != updates[key]
            for key in ('session_id', 'pid')
        )
        new_symbol = bool(updates.get('symbol') and state.get('symbol') != updates['symbol'])
        if reset or new_session or new_symbol:
            state.update(last_decision={}, decision_counts={}, last_api_error={},
                         last_error='', no_entry_for_sec=0, last_tick_epoch=0,
                         last_tick_at=None, last_loop_at=None, event='starting',
                         candle_count=0, warmup_remaining_candles=updates.get('min_candles', state.get('min_candles', 150)),
                         candle_source=None, shadow={})
        if new_session:
            state.update(started_at=datetime.now(timezone.utc).isoformat(),
                         stopped_at=None, finished_at=None, returncode=None)
        if owner == 'pipeline' and state.get('candle_source') in ('executor', 'training_private'):
            for key in _CANDLE_FIELDS:
                updates.pop(key, None)
        if (owner == 'dashboard' and updates.get('status') == 'starting'
                and state.get('status') == 'running' and state.get('candle_source') == 'executor'):
            updates.pop('status', None)
        state.update(updates)
        write_json_atomic(path, state)
        return state
