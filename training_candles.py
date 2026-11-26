"""Candles privados da IA, sem alterar os buffers da coleta ou do Score."""
import bisect
import math
import threading
import time
from pathlib import Path

import indicators
from model_contract import FEATURE_WINDOW, MAX_TICK_GAP_SEC
from training_history import TrainingHistory


class TrainingCandleBuffer:
    """Reutiliza cache verificado e completa OHLC com ticks realmente observados.

    ``refresh`` faz leitura local e deve rodar antes do WebSocket ou em worker.
    ``feed`` e ``window`` trabalham somente em memória dentro dos callbacks.
    """

    def __init__(self, root, symbol, *, timeframe=300, now=time.time):
        self.history = TrainingHistory(Path(root), timeframe=timeframe)
        self.symbol, self.timeframe, self.now = symbol, timeframe, now
        self._lock = threading.RLock()
        self._candles = {}
        # Epochs cujo OHLC veio do histórico público oficial (autoritativo).
        self._official = set()
        self._excluded = []
        self._current = None
        self._last_tick = 0
        self._gap_epoch = 0
        self._last_observed_close = None
        self.needs_refresh = True
        self.last_error = ''

    def _overlaps_exclusion(self, start, end):
        index = bisect.bisect_left(self._excluded, start)
        return index < len(self._excluded) and self._excluded[index] <= end

    def refresh(self, *, now=None):
        """Lê somente cache com hash/manifest válidos; falhas preservam a memória."""
        now = int(self.now() if now is None else now)
        try:
            records, state = self.history.cached_records(self.symbol)
            excluded = sorted(set(map(int, state.get('excluded_epochs', []))))
            boundary = now // self.timeframe * self.timeframe
            candles = {
                row['epoch']: dict(epoch=row['epoch'], open=row['open'], high=row['high'],
                                   low=row['low'], close=row['price'])
                for row in records if row['kind'] == 'candle'
                and row['timeframe'] == self.timeframe
                and row['epoch'] + self.timeframe <= boundary
            }
            official = set(candles)
            ticks = [row for row in records if row['kind'] == 'tick' and row['epoch'] <= now]
            epochs = [row['epoch'] for row in ticks]
            gaps = [0]
            for first, last in zip(epochs, epochs[1:]):
                gaps.append(gaps[-1] + (last - first > MAX_TICK_GAP_SEC))
            for candle in indicators.ticks_to_candles_by_time(ticks, self.timeframe):
                if candle['epoch'] + self.timeframe > boundary:
                    continue
                first = bisect.bisect_left(epochs, candle['epoch'])
                last = bisect.bisect_left(epochs, candle['epoch'] + self.timeframe) - 1
                if (last < first or epochs[first] - candle['epoch'] > MAX_TICK_GAP_SEC
                        or candle['epoch'] + self.timeframe - epochs[last] > MAX_TICK_GAP_SEC
                        or gaps[first] != gaps[last]):
                    continue
                candles.setdefault(candle['epoch'], candle)
        except BlockingIOError:
            self.last_error = 'cache_busy'
            return False
        except (OSError, ValueError, KeyError, TypeError, OverflowError):
            self.last_error = 'invalid_or_unavailable_cache'
            return False

        with self._lock:
            # Uma nova página nunca remove quarentenas já observadas nesta sessão.
            self._excluded = sorted(set(self._excluded) | set(excluded))
            for epoch, candle in self._candles.items():
                if epoch in official:
                    # O OHLC oficial substitui a vela montada ao vivo; divergência
                    # de abertura/extremos por conexão no meio da vela ou tick
                    # filtrado não justifica quarentena.
                    continue
                if epoch in candles and candles[epoch] != candle:
                    self._excluded = sorted(set(self._excluded) | {epoch})
                else:
                    candles[epoch] = candle
            self._official = (self._official | official) & set(candles)
            self._candles = {
                epoch: candles[epoch] for epoch in sorted(candles)[-500:]
                if not self._overlaps_exclusion(epoch, epoch + self.timeframe - 1)
            }
            # Um gap ao vivo só é reparado por OHLC fechado confirmado desse intervalo.
            if self._gap_epoch:
                gap_candle = self._gap_epoch // self.timeframe * self.timeframe
                if gap_candle in self._candles:
                    self._gap_epoch = 0
            self.last_error = ''
            self.needs_refresh = not bool(self.window(now))
        return True

    def feed(self, epoch, price):
        """Consome tick validado; retorna se apareceu um fechamento utilizável novo."""
        epoch, price = int(epoch), float(price)
        if epoch <= 0 or not math.isfinite(price) or price <= 0:
            return False
        with self._lock:
            if epoch <= self._last_tick:
                return False
            if self._last_tick and epoch - self._last_tick > MAX_TICK_GAP_SEC:
                self._current = None
                self._gap_epoch = self._last_tick
                self.needs_refresh = True
            self._last_tick = epoch
            start = epoch // self.timeframe * self.timeframe
            if self._current is not None and start > self._current['epoch']:
                candle = self._current
                if (start == candle['epoch'] + self.timeframe
                        and not self._overlaps_exclusion(candle['epoch'], start)):
                    confirmed = self._candles.get(candle['epoch'])
                    if confirmed is not None and candle['epoch'] in self._official:
                        pass  # Mantém o OHLC oficial já confirmado.
                    elif confirmed is not None and confirmed != candle:
                        self._excluded = sorted(set(self._excluded) | {candle['epoch']})
                        self.needs_refresh = True
                    else:
                        self._candles[candle['epoch']] = dict(candle)
                    self._candles = {key: self._candles[key] for key in sorted(self._candles)[-500:]}
                self._current = None
            if self._current is None:
                # Não transforma a margem parcial da conexão em candle completo.
                if epoch - start <= MAX_TICK_GAP_SEC:
                    self._current = dict(epoch=start, open=price, high=price, low=price, close=price)
            else:
                self._current['high'] = max(self._current['high'], price)
                self._current['low'] = min(self._current['low'], price)
                self._current['close'] = price
            window = self.window(epoch)
            if not window:
                self.needs_refresh = True
                return False
            self.needs_refresh = False
            latest = window[-1]['epoch']
            changed = latest != self._last_observed_close
            self._last_observed_close = latest
            # O shadow segue o mesmo instante de sinal usado nos rótulos do treino.
            return changed and epoch - (latest + self.timeframe) <= MAX_TICK_GAP_SEC

    def window(self, epoch):
        """Retorna 50 OHLC fechados contíguos e atuais, ou nenhuma entrada elegível."""
        with self._lock:
            latest = int(epoch) // self.timeframe * self.timeframe - self.timeframe
            first = latest - (FEATURE_WINDOW - 1) * self.timeframe
            if self._gap_epoch and self._gap_epoch >= first:
                return []
            if self._overlaps_exclusion(first, latest + self.timeframe):
                return []
            expected = range(first, latest + 1, self.timeframe)
            if any(item not in self._candles for item in expected):
                return []
            return [dict(self._candles[item]) for item in expected]

    def candle_count(self, epoch):
        """Quantidade atual contígua, mesmo quando ainda não há janela completa."""
        with self._lock:
            latest = int(epoch) // self.timeframe * self.timeframe - self.timeframe
            count = 0
            while latest in self._candles and count < FEATURE_WINDOW:
                if self._overlaps_exclusion(latest, latest + self.timeframe - 1):
                    break
                count += 1
                latest -= self.timeframe
            return count
