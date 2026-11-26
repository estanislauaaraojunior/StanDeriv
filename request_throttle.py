"""Cadência compartilhada para consultas de leitura do scanner."""
import math
import random
import threading
import time


class RequestThrottle:
    def __init__(self, interval=1.0):
        self._lock = threading.Lock()
        self._interval = interval
        self._next = 0.0
        self._backoffs = 0

    def wait(self):
        # Revalida após dormir: outro worker pode ter imposto uma pausa global.
        while True:
            with self._lock:
                now = time.monotonic()
                delay = self._next - now
                if delay <= 0:
                    self._next = now + self._interval
                    return
            time.sleep(delay)

    def backoff(self, retry_after=None):
        with self._lock:
            self._backoffs = min(self._backoffs + 1, 4)
            delay = min(60.0, 5.0 * 2 ** (self._backoffs - 1))
            try:
                supplied = float(retry_after)
                if math.isfinite(supplied) and supplied > 0:
                    delay = max(delay, supplied)
            except (TypeError, ValueError):
                pass
            delay += random.uniform(0.0, 1.0)
            self._interval = min(5.0, self._interval * 2)
            self._next = max(self._next, time.monotonic() + delay)
            return delay
