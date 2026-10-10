"""Validação forward de previsões sem compras ou mutação do risco financeiro."""
import math
import json
from pathlib import Path

import config
from model_contract import MAX_TICK_GAP_SEC, write_json_atomic


class ShadowMonitor:
    """Registra previsões direcionais e aprova um bundle após resultados shadow suficientes."""

    SCHEMA_VERSION = 1

    def __init__(self, path, *, symbol, model_id, timeframe_sec, duration_sec):
        if not model_id:
            raise ValueError('shadow exige model_id aprovado')
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.identity = {
            'symbol': str(symbol),
            'model_id': str(model_id),
            'timeframe_sec': int(timeframe_sec),
            'duration_sec': int(duration_sec),
        }
        self.minimum_results = config.SHADOW_MIN_RESULTS
        self.minimum_accuracy = config.SHADOW_MIN_ACCURACY
        self.minimum_net_payout = config.SHADOW_MIN_NET_PAYOUT
        self.state = self._load_state()

    def _empty_state(self, archives=None):
        return {
            'schema_version': self.SCHEMA_VERSION,
            **self.identity,
            'pending': None,
            'results': [],
            'approved_at_epoch': None,
            'rejected_at_epoch': None,
            'archives': list(archives or [])[-20:],
        }

    def _load_state(self):
        if not self.path.exists():
            return self._empty_state()
        try:
            previous = json.loads(self.path.read_text())
        except (OSError, ValueError):
            previous = None
        if (isinstance(previous, dict)
                and previous.get('schema_version') == self.SCHEMA_VERSION
                and all(previous.get(key) == value for key, value in self.identity.items())
                and isinstance(previous.get('results'), list)):
            previous.setdefault('pending', None)
            previous.setdefault('approved_at_epoch', None)
            previous.setdefault('rejected_at_epoch', None)
            previous.setdefault('archives', [])
            return previous
        archives = previous.get('archives', []) if isinstance(previous, dict) else []
        if isinstance(previous, dict) and previous.get('model_id'):
            archives = [*archives, {
                key: previous.get(key) for key in
                ('symbol', 'model_id', 'timeframe_sec', 'duration_sec', 'results', 'approved_at_epoch')
            }]
        state = self._empty_state(archives)
        self._save_state(state)
        return state

    def _save_state(self, state=None):
        write_json_atomic(self.path, self.state if state is None else state)

    @property
    def approved(self):
        return self.state.get('approved_at_epoch') is not None

    @property
    def rejected(self):
        return self.state.get('rejected_at_epoch') is not None

    @property
    def result_count(self):
        return len(self.state['results'])

    @property
    def accuracy(self):
        window = self.state['results'][-self.minimum_results:]
        if not window:
            return 0.0
        return sum(bool(row['correct']) for row in window) / len(window)

    def observe_prediction(self, *, entry_epoch, entry_price, direction, confidence, net_payout):
        """Armazena uma previsão elegível e não sobreposta; não envia proposta nem compra."""
        if self.approved or self.rejected or self.state.get('pending') is not None:
            return 'already_pending_or_approved'
        if direction not in (0, 1):
            return 'invalid_direction'
        try:
            entry_epoch = int(entry_epoch)
            entry_price, confidence, net_payout = map(float, (entry_price, confidence, net_payout))
        except (TypeError, ValueError, OverflowError):
            return 'invalid_observation'
        if (entry_epoch <= 0 or not all(math.isfinite(value) for value in
                (entry_price, confidence, net_payout)) or entry_price <= 0
                or not config.AI_CONFIDENCE_MIN < confidence <= 1):
            return 'invalid_observation'
        if net_payout < self.minimum_net_payout:
            return 'payout_below_floor'
        results = self.state['results']
        if results and entry_epoch < results[-1]['expiry_epoch']:
            return 'overlapping_prediction'
        pending = {
            'entry_epoch': entry_epoch,
            'expiry_epoch': entry_epoch + self.identity['duration_sec'],
            'entry_price': entry_price,
            'direction': int(direction),
            'confidence': confidence,
            'net_payout': net_payout,
        }
        self.state['pending'] = pending
        self._save_state()
        return 'pending'

    def settle_tick(self, *, epoch, price):
        """Liquida pelo primeiro tick no vencimento dentro da tolerância; gaps e empates não contam."""
        pending = self.state.get('pending')
        if pending is None:
            return 'no_pending'
        try:
            epoch, price = int(epoch), float(price)
        except (TypeError, ValueError, OverflowError):
            return 'invalid_tick'
        if not math.isfinite(price) or price <= 0:
            return 'invalid_tick'
        if epoch < pending['expiry_epoch']:
            return 'pending'
        if epoch - pending['expiry_epoch'] > MAX_TICK_GAP_SEC:
            self.state['pending'] = None
            self._save_state()
            return 'expired_with_gap'
        if price == pending['entry_price']:
            self.state['pending'] = None
            self._save_state()
            return 'tie'

        target = int(price > pending['entry_price'])
        result = {
            **pending,
            'exit_epoch': epoch,
            'exit_price': price,
            'correct': target == pending['direction'],
        }
        self.state['results'].append(result)
        self.state['results'] = self.state['results'][-1000:]
        self.state['pending'] = None
        recent = self.state['results'][-self.minimum_results:]
        if (len(recent) >= self.minimum_results
                and sum(bool(row['correct']) for row in recent) / len(recent) >= self.minimum_accuracy):
            self.state['approved_at_epoch'] = epoch
        elif len(recent) >= self.minimum_results:
            self.state['rejected_at_epoch'] = epoch
        self._save_state()
        return 'approved' if self.approved else ('rejected' if self.rejected else 'settled')