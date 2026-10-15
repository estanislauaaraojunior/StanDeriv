"""Calendário por data UTC da Deriv; ausência de informação nunca prova fechamento."""
from datetime import datetime, timezone
import threading
import time

from operational_diagnostics import safe_detail


class MarketCalendar:
    def __init__(self, fetch, ttl=3600):
        self.fetch = fetch
        self.ttl = ttl
        self._days = {}
        self._lock = threading.Lock()

    def _day(self, epoch):
        date = datetime.fromtimestamp(epoch, timezone.utc).date()
        key = date.isoformat()
        with self._lock:
            saved = self._days.get(key)
            if saved and time.monotonic() - saved[0] < saved[2]:
                return date, saved[1]
            try:
                response = self.fetch({'trading_times': key})
                entries = {}
                for market in response['trading_times']['markets']:
                    for submarket in market['submarkets']:
                        for item in submarket['symbols']:
                            symbol = item.get('underlying_symbol') or item.get('symbol')
                            if symbol:
                                entries[symbol] = item
            except Exception as exc:
                print(f'[CALENDÁRIO] Data={key} indisponível tipo={type(exc).__name__} causa={safe_detail(exc)}; lacunas permanecem não confirmadas.')
                entries = {}
            # Cache negativo curto evita repetir falhas para cada candle/ativo.
            self._days[key] = (time.monotonic(), entries, self.ttl if entries else min(self.ttl, 60))
            if len(self._days) > 32:
                del self._days[next(iter(self._days))]
            return date, entries

    @staticmethod
    def _seconds(value):
        h, m, s = map(int, value.split(':'))
        if not (0 <= h < 24 and 0 <= m < 60 and 0 <= s < 60):
            raise ValueError('horário inválido')
        return h * 3600 + m * 60 + s

    def _sessions(self, epoch, symbol):
        date, entries = self._day(epoch)
        item = entries.get(symbol)
        if item is None:
            return None
        # Descrições de eventos não são um calendário estruturado. Não deduzir
        # feriados/horários especiais a partir de texto livre.
        events = item.get('events') or []
        if not isinstance(events, list) or any(not isinstance(event, dict) for event in events):
            return None
        for event in events:
            dates = str(event.get('dates', ''))
            names = ('Mondays', 'Tuesdays', 'Wednesdays', 'Thursdays', 'Fridays', 'Saturdays', 'Sundays')
            if dates == date.isoformat() or dates == names[date.weekday()]:
                return None
            if dates not in names:
                try:
                    datetime.strptime(dates, '%Y-%m-%d')
                except ValueError:
                    return None
        days = item.get('trading_days')
        weekdays = ('Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun')
        if days is not None:
            if not isinstance(days, list) or any(day not in weekdays for day in days):
                return None
            if weekdays[date.weekday()] not in days:
                return []
        try:
            opens, closes = item['times']['open'], item['times']['close']
            if not isinstance(opens, list) or not isinstance(closes, list) or len(opens) != len(closes):
                return None
            sessions = []
            for opening, closing in zip(opens, closes):
                start, end = self._seconds(opening), self._seconds(closing)
                # O fechamento é o último segundo negociável, inclusive.
                if end < start:
                    return None  # sessão atravessando a meia-noite: não inferir
                sessions.append((start, end + 1))
            return sessions
        except (AttributeError, KeyError, TypeError, ValueError):
            return None

    def closed(self, symbol, start, end):
        """True somente quando todo [start,end) tem fechamento comprovado."""
        if end <= start:
            return False
        while start < end:
            boundary = int(start) // 86400 * 86400
            stop = min(end, boundary + 86400)
            sessions = self._sessions(start, symbol)
            if sessions is None or any(a < stop - boundary and b > start - boundary for a, b in sessions):
                return False
            start = stop
        return True

    def closed_candle(self, symbol, epoch, timeframe):
        return self.closed(symbol, epoch, epoch + timeframe)

    def consecutive(self, symbol, earlier, later, timeframe):
        return (later - earlier == timeframe or
                (later > earlier + timeframe and
                 self.closed(symbol, earlier + timeframe, later)))
