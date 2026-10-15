"""Ping JSON da Deriv, independente dos frames ping/pong do WebSocket."""
import json
import threading
import time

from operational_diagnostics import safe_detail


class ApiKeepalive:
    def __init__(self, ws, label, interval=30):
        self.ws, self.label, self.interval = ws, label, interval
        self._stop = threading.Event()
        self.sent = self.received = 0
        self.last_latency = None
        self._pending = {}
        self._lock = threading.Lock()

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True, name=f'{self.label}-api-ping')
        self._thread.start()

    def _run(self):
        while not self._stop.wait(self.interval):
            self.send_ping()

    def send_ping(self):
        if self._stop.is_set():
            return
        with self._lock:
            self.sent += 1
            req_id = 1_900_000_000 + self.sent
            self._pending[req_id] = time.monotonic()
            self._pending = dict(list(self._pending.items())[-10:])
        try:
            self.ws.send(json.dumps({'ping': 1, 'req_id': req_id}))
        except Exception as exc:
            if not self._stop.is_set():
                print(f'[{self.label}] Ping API falhou tipo={type(exc).__name__} causa={safe_detail(exc)}')

    def acknowledge(self, data):
        with self._lock:
            started = self._pending.pop(data.get('req_id'), None)
            if started is None:
                return
            self.received += 1
            self.last_latency = time.monotonic() - started

    def stop(self):
        if not self._stop.is_set():
            print(f'[{self.label}] Keepalive API encerrado enviados={self.sent} respostas={self.received}')
        self._stop.set()
