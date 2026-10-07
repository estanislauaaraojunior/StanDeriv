"""Propostas amostradas sem compras e consulta histórica causal de payout."""
import bisect
import csv
import fcntl
import json
import hashlib
import io
import math
import time
from pathlib import Path
from model_contract import net_payout_ratio

FIELDS = ['observed_epoch', 'symbol', 'duration', 'duration_unit', 'direction',
          'currency', 'ask_price', 'payout', 'net_payout', 'source']


def market_unavailable(error):
    """Only market restrictions, never an arbitrary validation/authentication error."""
    code = error.get('code', '')
    message = str(error.get('message', '')).lower()
    duration_or_contract_mismatch = (
        'not offered for this duration' in message
        or 'trading is not offered for this duration' in message
        or 'duration' in message and 'not offered' in message
    )
    if duration_or_contract_mismatch:
        return True
    return code in {'MarketIsClosed', 'MarketClosed', 'TradingDisabled', 'InvalidSymbol'} or (
        code == 'ContractBuyValidationError' and
        ('trading is not available' in message or
         'trading on forex contracts' in message and 'is not available' in message)
    )


class ProposalRecorder:
    """Solicita propostas CALL/PUT periodicamente (sem comprar) e registra o payout em CSV."""
    def __init__(self, path, symbol, duration, unit, currency='USD', interval=300, on_unavailable=None):
        """Configura arquivo de saída, contrato amostrado, intervalo entre amostras e callback de mercado indisponível."""
        self.path, self.symbol = Path(path), symbol
        self.duration, self.unit, self.currency = duration, unit, currency
        self.interval, self.last_request = interval, -float('inf')
        self.counter, self.pending = 800_000_000, {}
        self.on_unavailable = on_unavailable

    def sample(self, ws, *, bearer, now=None):
        """Envia pelo websocket propostas CALL e PUT se o intervalo mínimo desde a última amostra já passou."""
        now = time.time() if now is None else now
        self.pending = {k:v for k,v in self.pending.items() if now-v[1] <= 30}
        if now-self.last_request < self.interval:
            return
        self.last_request = now
        for direction in ('CALL', 'PUT'):
            self.counter += 1
            self.pending[self.counter] = (direction, now)
            request = dict(proposal=1, amount=1, basis='stake', contract_type=direction,
                           currency=self.currency, duration=self.duration, duration_unit=self.unit,
                           req_id=self.counter)
            request['underlying_symbol' if bearer else 'symbol'] = self.symbol
            ws.send(json.dumps(request))

    def handle(self, message, now=None):
        """Processa resposta de proposta pendente e grava a linha no CSV; retorna True se a mensagem era dela."""
        key = message.get('req_id')
        if key not in self.pending:
            return False
        direction, sent = self.pending.pop(key)
        now = time.time() if now is None else now
        if 'error' in message:
            if 0 <= now-sent <= 30 and self.on_unavailable and market_unavailable(message['error']):
                self.on_unavailable(self.symbol)
            print(f"[PAYOUT] {direction} indisponível: {message['error'].get('code', 'erro API')}")
            return True
        if not 0 <= now-sent <= 30:
            print(f'[PAYOUT] {direction}: resposta fora do prazo de amostragem')
            return True
        try:
            proposal = message['proposal']
            price, payout = float(proposal['ask_price']), float(proposal['payout'])
            if not all(math.isfinite(v) for v in (price, payout)) or not 0 < price < payout or not proposal.get('id'):
                return True
            row = [now, self.symbol, self.duration, self.unit, direction, self.currency,
                   price, payout, net_payout_ratio(price, payout), 'sampled_proposal']
            with self.path.open('a+', newline='') as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                f.seek(0)
                header = next(csv.reader(f), None)
                if header is not None and header != FIELDS:
                    raise ValueError('esquema de propostas incompatível')
                f.seek(0, 2)
                writer = csv.writer(f)
                if header is None:
                    writer.writerow(FIELDS)
                writer.writerow(row)
        except (KeyError, TypeError, ValueError, OSError) as exc:
            print(f'[PAYOUT] Proposta não registrada: {type(exc).__name__}')
        return True


class QuoteHistory:
    """Histórico de payouts líquidos amostrados, consultado de forma causal por última cotação anterior."""
    def __init__(self, path, *, currency='USD', max_age=900):
        """Carrega o CSV de propostas (lock compartilhado), agrupando por símbolo, duração e direção."""
        self.groups, self.max_age = {}, max_age
        with Path(path).open(newline='') as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            raw = f.read()
            self.source_sha256 = hashlib.sha256(raw.encode()).hexdigest()
            for row in csv.DictReader(io.StringIO(raw)):
                if row['currency'] != currency:
                    continue
                epoch, price, payout = (float(row[k]) for k in ('observed_epoch', 'ask_price', 'payout'))
                if not all(math.isfinite(v) for v in (epoch, price, payout)) or not 0 < price < payout:
                    raise ValueError('cotação histórica inválida')
                key = (row['symbol'], int(row['duration']), row['duration_unit'], row['direction'])
                self.groups.setdefault(key, []).append((epoch, net_payout_ratio(price, payout)))
        self.groups = {key: sorted(values) for key, values in self.groups.items()}
        self.epochs = {key: [v[0] for v in values] for key, values in self.groups.items()}

    def lookup(self, symbol, duration, unit, direction, epoch):
        """Payout líquido da cota mais recente até `epoch` (direction 1=CALL), ou None se não houver ou estiver velha."""
        key = (symbol, int(duration), unit, 'CALL' if direction == 1 else 'PUT')
        times = self.epochs.get(key, [])
        index = bisect.bisect_right(times, epoch)-1
        if index < 0 or epoch-times[index] > self.max_age:
            return None
        return self.groups[key][index][1]
