"""Histórico público exclusivo da IA; não acessa conta ou arquivos da coleta.

Protocolo: https://developers.deriv.com/docs/data/ticks-history/
Endpoint: https://developers.deriv.com/docs/options/websocket/
"""
import argparse
import bisect
import copy
import csv
import fcntl
import hashlib
import json
import math
import os
import re
import tempfile
import time
from pathlib import Path

import websocket

from market_data import FIELDS, canonicalize, read_records
from model_contract import FEATURE_WINDOW, MAX_TICK_GAP_SEC, fingerprint, write_json_atomic
from operational_diagnostics import safe_detail
from training_data import TRAINING_HISTORY_DAYS, TrainingDataError

PUBLIC_URL = 'wss://api.derivws.com/trading/v1/options/ws/public'


class RecoveryBudgetExhausted(RuntimeError):
    """A próxima rodada continua somente os intervalos ainda ausentes."""


class HistoryAPIError(RuntimeError):
    def __init__(self, code, retry_after=None):
        super().__init__(f'histórico público: {safe_detail(code)}')
        self.code, self.retry_after = code, retry_after


class PublicHistoryClient:
    """Socket independente, somente ticks_history, com correlação e prazo total."""

    def __init__(self, *, max_requests=12, seconds=15, interval=1.0,
                 connect=None, clock=time.monotonic, sleep=time.sleep):
        self.max_requests = max_requests
        self.deadline = clock() + seconds
        self.interval, self.clock, self.sleep = interval, clock, sleep
        self.connect = connect or websocket.create_connection
        self.socket = None
        self.next_send = 0
        self.stats = dict(requests=0, timeouts=0, request_errors=0, max_latency_sec=0.0)

    def _remaining(self):
        remaining = self.deadline - self.clock()
        if remaining <= 0:
            raise RecoveryBudgetExhausted('prazo da recuperação atingido')
        return remaining

    def request(self, request):
        allowed = {'ticks_history', 'start', 'end', 'count', 'style', 'granularity'}
        if (set(request) - allowed or not re.fullmatch(r'\w{2,30}', str(request.get('ticks_history', '')))
                or request.get('style') not in ('ticks', 'candles')):
            raise ValueError('cliente da IA permite somente histórico público sem assinatura')
        if self.stats['requests'] >= self.max_requests:
            raise RecoveryBudgetExhausted('limite de consultas da rodada atingido')
        wait = max(0, self.next_send - self.clock())
        if wait >= self._remaining():
            raise RecoveryBudgetExhausted('cadência ultrapassaria o prazo da rodada')
        if wait:
            self.sleep(wait)
        started = self.clock()
        self.stats['requests'] += 1
        req_id = self.stats['requests']
        self.next_send = self.clock() + self.interval
        try:
            if self.socket is None:
                self.socket = self.connect(PUBLIC_URL, timeout=min(5, self._remaining()))
            self.socket.settimeout(min(5, self._remaining()))
            self.socket.send(json.dumps(dict(request, req_id=req_id)))
            expected = 'history' if request['style'] == 'ticks' else 'candles'
            while True:
                self.socket.settimeout(min(5, self._remaining()))
                raw = self.socket.recv()
                if not raw:
                    raise ConnectionError('socket público encerrado')
                data = json.loads(raw)
                if not isinstance(data, dict) or data.get('req_id') != req_id:
                    continue
                echo = data.get('echo_req', {})
                if echo.get('ticks_history', request['ticks_history']) != request['ticks_history']:
                    raise ValueError('resposta de outro símbolo')
                if data.get('error'):
                    error = data['error']
                    raise HistoryAPIError(error.get('code', 'erro API'), error.get('retry_after'))
                if data.get('msg_type') != expected:
                    raise ValueError('tipo de resposta histórica inesperado')
                return data
        except (TimeoutError, websocket.WebSocketTimeoutException):
            self.stats['timeouts'] += 1
            self.stats['request_errors'] += 1
            raise
        except RecoveryBudgetExhausted:
            raise
        except Exception:
            self.stats['request_errors'] += 1
            raise
        finally:
            self.stats['max_latency_sec'] = round(max(self.stats['max_latency_sec'],
                                                    self.clock() - started), 3)

    def close(self):
        if self.socket is not None:
            try:
                self.socket.close()
            finally:
                self.socket = None


def _integer(value):
    number = float(value)
    integer = int(number)
    if isinstance(value, bool) or integer != number or not 0 < integer <= 9999999999:
        raise ValueError('época histórica inválida')
    return integer


def _price(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError('preço histórico inválido')
    return result


def _tick_page(data, symbol, start, end):
    history = data.get('history')
    if not isinstance(history, dict):
        raise ValueError('histórico de ticks ausente')
    times, prices = history.get('times'), history.get('prices')
    if not isinstance(times, list) or not isinstance(prices, list) or len(times) != len(prices):
        raise ValueError('arrays de ticks inválidos')
    rows = []
    for epoch, price in zip(times, prices):
        epoch = _integer(epoch)
        if start <= epoch <= end:
            rows.append(dict(epoch=epoch, datetime='', symbol=symbol, price=_price(price),
                             kind='tick', timeframe=0, open='', high='', low=''))
    return sorted(rows, key=lambda row: row['epoch'])


def _candle_page(data, symbol, start, end, timeframe, boundary):
    if not isinstance(data.get('candles'), list):
        raise ValueError('histórico OHLC ausente')
    rows = []
    for candle in data['candles']:
        epoch = _integer(candle['epoch'])
        if not start <= epoch <= end or epoch + timeframe > boundary:
            continue
        price, opening, high, low = map(_price, [candle['close'], candle['open'], candle['high'], candle['low']])
        if epoch % timeframe or not low <= min(opening, price) <= max(opening, price) <= high:
            raise ValueError('OHLC histórico inválido')
        rows.append(dict(epoch=epoch, datetime='', symbol=symbol, price=price,
                         kind='candle', timeframe=timeframe, open=opening, high=high, low=low))
    return rows


def _ranges(epochs, step):
    ranges = []
    for epoch in epochs:
        if ranges and epoch == ranges[-1][1] + step:
            ranges[-1] = (ranges[-1][0], epoch)
        else:
            ranges.append((epoch, epoch))
    return ranges


def _label_ranges(candles, timeframe, duration, now):
    """Somente contratos já vencidos e com 50 candles consecutivos para features."""
    segment, ranges = 0, []
    for i, epoch in enumerate(candles):
        if i and epoch - candles[i - 1] != timeframe:
            segment = i
        signal, end = epoch + timeframe, epoch + timeframe + duration + 2 * MAX_TICK_GAP_SEC
        if i - segment + 1 < FEATURE_WINDOW or end > now:
            continue
        if ranges and signal <= ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], end)
        else:
            ranges.append((signal, end))
    return ranges


def _missing_ticks(known, start, end):
    """Busca somente lacunas maiores que a tolerância de rotulagem vigente."""
    result, previous = [], None
    lo, hi = bisect.bisect_left(known, start), bisect.bisect_right(known, end)
    for epoch in known[lo:hi]:
        if previous is None:
            if epoch - start > MAX_TICK_GAP_SEC:
                result.append((start, epoch - 1))
        elif epoch - previous > MAX_TICK_GAP_SEC:
            result.append((previous + 1, epoch - 1))
        previous = epoch
    if previous is None:
        return [(start, end)]
    if end - previous > MAX_TICK_GAP_SEC:
        result.append((previous + 1, end))
    return result


class TrainingHistory:
    """Cache incremental da IA com orçamento por rodada e falha fechada por página."""

    def __init__(self, root, *, timeframe=900, duration=900, max_requests=12, seconds=15,
                 now=time.time, client_factory=PublicHistoryClient, cancelled=lambda: False):
        if timeframe <= 0 or duration <= 0 or max_requests < 0 or seconds <= 0:
            raise ValueError('limites de recuperação da IA inválidos')
        self.root = Path(root)
        self.timeframe, self.duration = timeframe, duration
        self.max_requests, self.seconds = max_requests, seconds
        self.now, self.client_factory, self.cancelled = now, client_factory, cancelled
        self._verified_snapshots = {}

    def _folder(self, symbol):
        digest = hashlib.sha256(symbol.encode()).hexdigest()[:12]
        return self.root / f'{digest}-{self.timeframe}'

    def _read(self, folder, symbol):
        manifest = folder / 'manifest.json'
        if not manifest.exists():
            return [], dict(excluded_epochs=[], retries={})
        state = json.loads(manifest.read_text())
        path = folder / 'records.csv'
        if (state.get('symbol') != symbol or state.get('timeframe') != self.timeframe
                or state.get('sha256') != fingerprint(path)):
            raise TrainingDataError('cache histórico da IA incompatível ou alterado')
        rows, rejected = read_records(path)
        if rejected or any(r['symbol'] != symbol for r in rows):
            raise TrainingDataError('cache histórico da IA inválido')
        self._verified_snapshots[symbol] = (rows, copy.deepcopy(state))
        return rows, state

    def _save(self, folder, symbol, rows, state):
        folder.mkdir(parents=True, exist_ok=True)
        fd, temp = tempfile.mkstemp(prefix='records-', dir=folder)
        try:
            with os.fdopen(fd, 'w', newline='') as handle:
                writer = csv.DictWriter(handle, FIELDS)
                writer.writeheader()
                writer.writerows(rows)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, folder / 'records.csv')
        finally:
            if os.path.exists(temp):
                os.unlink(temp)
        state.update(symbol=symbol, timeframe=self.timeframe, sha256=fingerprint(folder / 'records.csv'))
        write_json_atomic(folder / 'manifest.json', state)
        self._verified_snapshots[symbol] = (rows, copy.deepcopy(state))

    def _sibling_ticks(self, symbol, folder):
        """Lê somente ticks de caches verificados, sob o lock de recuperação da IA."""
        if self.timeframe != 300:
            return [], set()
        reader = TrainingHistory(self.root, timeframe=900, duration=self.duration)
        sibling = reader._folder(symbol)
        if sibling == folder or not (sibling / 'manifest.json').exists():
            return [], set()
        cached, state = reader._read(sibling, symbol)
        return [row for row in cached if row['kind'] == 'tick' and row['timeframe'] == 0], set(state['excluded_epochs'])

    def cached_records(self, symbol):
        """Snapshot local verificado sob lock compartilhado, sem consulta pública."""
        if not self.root.exists():
            return [], dict(excluded_epochs=[], retries={})
        with (self.root / '.public-history.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            records, state = self._read(self._folder(symbol), symbol)
            excluded = set(state['excluded_epochs'])
            return [row for row in records if row['epoch'] not in excluded], state

    def enrich(self, records, preparation):
        symbol = preparation['symbol']
        self.root.mkdir(parents=True, exist_ok=True)
        # Serializa apenas os downloads da IA, inclusive entre processos.
        with (self.root / '.public-history.lock').open('a+') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                # Ler records/manifest sem lock poderia misturar duas publicações.
                # Conserve somente um snapshot anteriormente verificado nesta instância.
                previous = self._verified_snapshots.get(symbol)
                if previous is not None:
                    cached, state = previous
                    records, conflicts, _ = canonicalize([*records, *cached])
                    excluded = set(preparation['excluded_epochs']) | set(state['excluded_epochs'])
                    excluded.update(row['epoch'] for row in conflicts)
                    preparation['excluded_epochs'] = sorted(excluded)
                    now = int(self.now())
                    boundary = now // self.timeframe * self.timeframe
                    start = boundary - TRAINING_HISTORY_DAYS * 86400
                    records = [row for row in records if row['epoch'] not in excluded
                               and start <= row['epoch'] <= now
                               and (row['kind'] != 'candle' or
                                    row['timeframe'] == self.timeframe
                                    and row['epoch'] + self.timeframe <= boundary)]
                return records, dict(stage='busy', requests=0, cache_preserved=previous is not None,
                                     reason='outra preparação da IA está recuperando dados')
            return self._enrich_locked(records, preparation, symbol)

    def enrich_incremental(self, records, preparation, *, rounds=1,
                           total_requests=None, total_seconds=None, ready=None,
                           clock=time.monotonic):
        """Continua lotes com ganho, dentro de limites totais e sem esperar dados ao vivo."""
        if rounds < 1 or int(rounds) != rounds:
            raise ValueError('quantidade de rodadas deve ser um inteiro positivo')
        total_requests = self.max_requests * rounds if total_requests is None else total_requests
        total_seconds = self.seconds * rounds if total_seconds is None else total_seconds
        if (total_requests < 0 or int(total_requests) != total_requests
                or not math.isfinite(total_seconds) or total_seconds <= 0):
            raise ValueError('orçamento total da recuperação da IA inválido')
        deadline = clock() + total_seconds
        totals = dict(requests=0, timeouts=0, request_errors=0,
                      candles_added=0, ticks_added=0, max_latency_sec=0.0)
        errors, completed, added = [], 0, 0
        status = dict(stage='partial')
        stop_reason = 'round_limit'
        for _ in range(rounds):
            remaining = deadline - clock()
            if self.cancelled():
                stop_reason = 'cancelled'
                break
            if remaining <= 0:
                stop_reason = 'time_budget'
                break
            remaining_requests = total_requests - totals['requests']
            if completed and remaining_requests <= 0:
                stop_reason = 'request_budget'
                break
            before = {(row['epoch'], row['kind'], row['timeframe']) for row in records}
            worker = TrainingHistory(
                self.root, timeframe=self.timeframe, duration=self.duration,
                max_requests=min(self.max_requests, remaining_requests),
                seconds=min(self.seconds, remaining), now=self.now,
                client_factory=self.client_factory, cancelled=self.cancelled,
            )
            worker._verified_snapshots = self._verified_snapshots
            # Inclui leitura, merge e callbacks no prazo global; o socket recebe
            # somente o tempo que ainda resta quando for criado.
            def bounded_factory(**kwargs):
                kwargs['seconds'] = min(kwargs['seconds'], max(0, deadline - clock()))
                return self.client_factory(**kwargs)
            worker.client_factory = bounded_factory
            worker.cancelled = lambda: self.cancelled() or clock() >= deadline
            records, status = worker.enrich(records, preparation)
            completed += 1
            gain = len({(row['epoch'], row['kind'], row['timeframe']) for row in records} - before)
            added += gain
            errors.extend(status.get('errors', []))
            for key in totals:
                value = status.get(key, 0)
                totals[key] = max(totals[key], value) if key == 'max_latency_sec' else totals[key] + value
            if status['stage'] in ('busy', 'backoff', 'unavailable'):
                stop_reason = status['stage']
                break
            if self.cancelled():
                stop_reason = 'cancelled'
                break
            if clock() >= deadline:
                stop_reason = 'time_budget'
                break
            if ready is not None and ready(records, preparation):
                stop_reason = 'ready'
                break
            if not gain:
                stop_reason = 'no_progress'
                break
        status = dict(status, **totals, errors=errors, rounds_completed=completed,
                      records_added=added, stop_reason=stop_reason)
        return records, status

    def _enrich_locked(self, records, preparation, symbol):
        now = int(self.now())
        boundary = now // self.timeframe * self.timeframe
        start = boundary - TRAINING_HISTORY_DAYS * 86400
        recovery_start = max(start, int(preparation.get('recovery_start', start)))
        recovery_start = min(boundary, recovery_start // self.timeframe * self.timeframe)
        folder = self._folder(symbol)
        cached, state = self._read(folder, symbol)
        sibling_ticks, sibling_excluded = self._sibling_ticks(symbol, folder)
        combined, conflicts, _ = canonicalize([*records, *cached, *sibling_ticks])
        excluded = set(preparation['excluded_epochs']) | set(state['excluded_epochs']) | sibling_excluded
        excluded.update(r['epoch'] for r in conflicts)
        preparation['excluded_epochs'] = sorted(excluded)
        combined = [r for r in combined if start <= r['epoch'] <= now
                    and (r['kind'] != 'candle' or
                         r['timeframe'] == self.timeframe and r['epoch'] + self.timeframe <= boundary)]
        retries = {key: value for key, value in state.get('retries', {}).items() if value > now}
        status = dict(stage='recovering', requests=0, candles_added=0, ticks_added=0, errors=[],
                      excluded_epochs=len(excluded), recovery_start=recovery_start)
        client = self.client_factory(max_requests=self.max_requests, seconds=self.seconds)
        incoming = []
        attempts = []

        def request(style, first, last, count):
            if self.cancelled():
                raise RecoveryBudgetExhausted('encerramento solicitado')
            key = f'{style}:{first}:{last}'
            if retries.get(key, 0) > now:
                return None
            attempts.append((style, first, last))
            return client.request(dict(ticks_history=symbol, start=first, end=str(last),
                                       count=count, style=style,
                                       **({'granularity': self.timeframe} if style == 'candles' else {})))

        try:
            backoff_path = self.root / 'backoff.json'
            global_backoff = json.loads(backoff_path.read_text()).get('retry_after', 0) if backoff_path.exists() else 0
            retry_after = max(state.get('retry_after', 0), global_backoff)
            if retry_after > now:
                status.update(stage='backoff', retry_after=retry_after)
                raise RecoveryBudgetExhausted('aguardando backoff do histórico público')
            known_candles = {r['epoch'] for r in combined if r['kind'] == 'candle'}
            missing = [e for e in range(recovery_start, boundary, self.timeframe) if e not in known_candles]
            pages = []
            for first, last in _ranges(missing, self.timeframe):
                while last >= first:
                    page_start = max(first, last - 999 * self.timeframe)
                    pages.append((page_start, last))
                    last = page_start - self.timeframe
            # Reserva orçamento para ticks mesmo quando houver intervalos OHLC vazios.
            candle_requests = 0
            for first, last in sorted(pages, reverse=True):
                if candle_requests >= max(1, min(4, self.max_requests // 2)):
                    break
                data = request('candles', first, last + self.timeframe - 1,
                               (last - first) // self.timeframe + 1)
                if data is None:
                    continue
                candle_requests += 1
                rows = _candle_page(data, symbol, first, last, self.timeframe, boundary)
                incoming.extend(rows)
                status['candles_added'] += len(rows)
                # Ausência é provisória; nunca se torna cobertura confirmada.
                retries[f'candles:{first}:{last + self.timeframe - 1}'] = now + 300
            candles = sorted({r['epoch'] for r in [*combined, *incoming] if r['kind'] == 'candle'})
            targets = [(max(first, recovery_start), last)
                       for first, last in _label_ranges(candles, self.timeframe, self.duration, now)
                       if last >= recovery_start]
            known_ticks = sorted({r['epoch'] for r in combined if r['kind'] == 'tick'})
            gaps = [gap for first, last in targets for gap in _missing_ticks(known_ticks, first, last)]
            status['missing_tick_ranges_before'] = len(gaps)
            # Alterna faixas entre rodadas: uma lacuna recente indisponível não
            # pode consumir todo o orçamento e impedir recuperar as demais.
            ordered_gaps = list(reversed(gaps))
            offset = int(state.get('tick_gap_offset', 0)) % max(1, len(ordered_gaps))
            ordered_gaps = ordered_gaps[offset:] + ordered_gaps[:offset]
            state['tick_gap_offset'] = offset + 1
            for first, last in ordered_gaps:
                cursor = last
                while cursor >= first:
                    data = request('ticks', first, cursor, 5000)
                    if data is None:
                        break
                    rows = _tick_page(data, symbol, first, cursor)
                    if not rows:
                        retries[f'ticks:{first}:{cursor}'] = now + 300
                        break
                    incoming.extend(rows)
                    status['ticks_added'] += len(rows)
                    earliest = rows[0]['epoch']
                    if earliest <= first + MAX_TICK_GAP_SEC:
                        break
                    cursor = earliest - 1
        except RecoveryBudgetExhausted:
            pass
        except Exception as exc:
            status['errors'].append(dict(type=type(exc).__name__, detail=safe_detail(exc)))
            status['stage'] = 'unavailable'
            if isinstance(exc, HistoryAPIError) and str(exc.code).lower() in ('ratelimit', 'rate_limit'):
                try:
                    delay = float(exc.retry_after)
                    delay = delay if math.isfinite(delay) else 60
                except (ValueError, TypeError):
                    delay = 60
                state['retry_after'] = now + max(60, delay)
                write_json_atomic(self.root / 'backoff.json', dict(retry_after=state['retry_after']))
        finally:
            status.update(client.stats)
            client.close()
        merged, conflicts, _ = canonicalize([*combined, *incoming])
        excluded.update(r['epoch'] for r in conflicts)
        # Não reintroduz um preço conflitante que foi removido em rodada anterior.
        merged = [r for r in merged if r['epoch'] not in excluded]
        # Páginas podem repetir margens ou preços já presentes; métricas contam
        # somente registros novos que continuaram válidos após deduplicação/vetos.
        existing = {(row['epoch'], row['kind'], row['timeframe']) for row in combined}
        recovered = [row for row in merged
                     if (row['epoch'], row['kind'], row['timeframe']) not in existing]
        status['candles_added'] = sum(row['kind'] == 'candle' for row in recovered)
        status['ticks_added'] = sum(row['kind'] == 'tick' for row in recovered)
        preparation['excluded_epochs'] = sorted(excluded)
        state.update(excluded_epochs=sorted(excluded), retries=retries, last_attempt=now)
        self._save(folder, symbol, merged, state)
        status['cache_sha256'] = state['sha256']
        status['cached_records'] = len(merged)
        candle_epochs = sorted({r['epoch'] for r in merged if r['kind'] == 'candle'})
        tick_epochs = sorted({r['epoch'] for r in merged if r['kind'] == 'tick'})
        status['missing_candles'] = len(set(range(recovery_start, boundary, self.timeframe)) - set(candle_epochs))
        status['missing_tick_ranges'] = sum(
            len(_missing_ticks(tick_epochs, max(first, recovery_start), last))
            for first, last in _label_ranges(candle_epochs, self.timeframe, self.duration, now)
            if last >= recovery_start
        )
        if status['stage'] == 'recovering':
            status['stage'] = 'partial' if status['missing_candles'] or status['missing_tick_ranges'] else 'available'
        return merged, status


def feed_diagnostics(log_path, started_at=None):
    """Resume falhas já registradas sem tocar nos callbacks ou timeouts do coletor."""
    from datetime import datetime
    count, last = 0, None
    executor_count, executor_last = 0, None
    try:
        with Path(log_path).open('rb') as handle:
            handle.seek(max(0, handle.seek(0, 2) - 256000))
            lines = handle.read().decode('utf-8', errors='replace').splitlines()
        cutoff = datetime.fromisoformat(started_at).timestamp() if started_at else None
        for line in lines:
            match = re.search(r'horário=(\S+)', line)
            if not match or not any(tag in line for tag in
                                    ('[COLETOR] Erro WebSocket:', '[BOT] Erro WebSocket:')):
                continue
            epoch = datetime.fromisoformat(match.group(1)).timestamp()
            if cutoff is not None and epoch < cutoff:
                continue
            if 'ping/pong timed out' in line:
                if '[BOT] Erro WebSocket:' in line:
                    executor_count += 1
                    executor_last = match.group(1)
                else:
                    count += 1
                    last = match.group(1)
    except (OSError, ValueError, TypeError):
        return dict(available=False)
    return dict(available=True, ping_timeouts=count, last_timeout_at=last,
                executor_ping_timeouts=executor_count, executor_last_timeout_at=executor_last,
                scope='last_256_kb',
                note='Leitura do log; latência de callbacks não foi medida.')


def main():
    """Prepara dados públicos da IA; não treina, promove, autentica ou envia ordens."""
    from training_data import prepare_snapshot
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--symbol', required=True)
    parser.add_argument('--history')
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--max-requests', type=int, default=12)
    parser.add_argument('--seconds', type=float, default=15)
    parser.add_argument('--timeframe', type=int, choices=(300, 900), default=900,
                        help='candles exclusivos da IA em segundos (padrão: 900)')
    parser.add_argument('--duration', type=int, default=900,
                        help='vencimento dos contratos em segundos (padrão: 900)')
    parser.add_argument('--rounds', type=int, default=1,
                        help='máximo de lotes incrementais; continua somente enquanto há ganho')
    parser.add_argument('--total-requests', type=int,
                        help='teto global de consultas (padrão: --max-requests vezes --rounds)')
    parser.add_argument('--total-seconds', type=float,
                        help='prazo global em segundos (padrão: --seconds vezes --rounds)')
    args = parser.parse_args()
    if (args.rounds < 1 or args.duration <= 0 or args.max_requests < 0
            or not math.isfinite(args.seconds) or args.seconds <= 0
            or args.total_requests is not None and args.total_requests < 0
            or args.total_seconds is not None and
            (not math.isfinite(args.total_seconds) or args.total_seconds <= 0)):
        parser.error('duração, rodadas e prazos devem ser positivos; consultas não podem ser negativas')
    provider = TrainingHistory(args.cache_dir, timeframe=args.timeframe, duration=args.duration,
                               max_requests=args.max_requests, seconds=args.seconds)
    enricher = provider.enrich
    if args.rounds != 1 or args.total_requests is not None or args.total_seconds is not None:
        enricher = lambda records, preparation: provider.enrich_incremental(
            records, preparation, rounds=args.rounds,
            total_requests=args.total_requests, total_seconds=args.total_seconds)
    snapshot, report = prepare_snapshot(args.input, args.output_dir, args.symbol,
                                        history_path=args.history, enricher=enricher)
    print(json.dumps(dict(snapshot=str(snapshot), history_recovery=report['history_recovery']), ensure_ascii=False))


if __name__ == '__main__':
    main()
