"""Histórico OHLC incremental por ativo, separado dos ticks usados no treino."""
from __future__ import annotations

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

from market_data import FIELDS, canonicalize, read_records


class AssetHistoryStore:
    """Persiste candles fechados por ativo e sincroniza somente epochs ainda ausentes."""

    def __init__(self, root, timeframe_sec: int, *, max_days: int = 7,
                 max_candles: int | None = None, page_limit: int = 1000):
        if timeframe_sec <= 0 or max_days <= 0 or page_limit <= 0:
            raise ValueError("timeframe, janela e limite de página devem ser positivos")
        if max_candles is not None and max_candles <= 0:
            raise ValueError("limite de candles deve ser positivo")
        self.root = Path(root)
        self.timeframe_sec = int(timeframe_sec)
        self.max_days = int(max_days)
        self.max_candles = int(max_candles) if max_candles is not None else None
        self.page_limit = int(page_limit)

    def path_for(self, symbol: str) -> Path:
        normalized = str(symbol).strip()
        slug = re.sub(r"[^A-Za-z0-9_-]+", "_", normalized).strip("_") or "symbol"
        suffix = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:10]
        return self.root / f"{slug}-{suffix}.csv"

    def read(self, symbol: str) -> list[dict]:
        path = self.path_for(symbol)
        if not path.exists():
            return []
        rows, rejected = read_records(path)
        if rejected:
            raise ValueError("histórico local contém registros inválidos")
        clean, conflicts, _duplicates = canonicalize(rows)
        if conflicts:
            raise ValueError("histórico local contém candles conflitantes")
        if any(not self._is_asset_candle(row, symbol) for row in clean):
            raise ValueError("histórico local contém ativo ou tipo inesperado")
        return clean

    def synchronize(self, symbol: str, fetcher, *, now_epoch: int | None = None) -> dict:
        """Busca gaps em páginas; falhas mantêm dados válidos e deixam intervalos para retry."""
        now = int(time.time() if now_epoch is None else now_epoch)
        empty_epoch_ttl = max(self.timeframe_sec, 3600)
        current_boundary = now // self.timeframe_sec * self.timeframe_sec
        end_epoch = current_boundary - self.timeframe_sec
        candle_count = self.max_candles or self.max_days * 86400 // self.timeframe_sec
        start_epoch = end_epoch - (candle_count - 1) * self.timeframe_sec
        if end_epoch < start_epoch:
            return {"candles": [], "complete": False, "requested": 0, "added": 0}

        path = self.path_for(symbol)
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.with_suffix(path.suffix + ".lock")
        with lock_path.open("a+") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            existing = self._read_locked(path, symbol)
            metadata_path = path.with_suffix(".metadata.json")
            metadata = self._read_metadata(metadata_path, symbol)

            existing_by_epoch = {row["epoch"]: row for row in existing}
            present = {epoch for epoch in existing_by_epoch
                       if start_epoch <= epoch <= end_epoch}
            raw_no_data = metadata.get("no_data", {})
            no_data_checks = {}
            if isinstance(raw_no_data, dict):
                for epoch, checked_at in raw_no_data.items():
                    try:
                        epoch_value, checked_value = int(epoch), int(checked_at)
                    except (TypeError, ValueError):
                        continue
                    if (start_epoch <= epoch_value <= end_epoch
                            and 0 <= now - checked_value < empty_epoch_ttl):
                        no_data_checks[str(epoch_value)] = checked_value
            no_data = {
                int(epoch) for epoch in no_data_checks
            }
            present.update(existing_by_epoch)
            no_data.difference_update(existing_by_epoch)

            requested_ranges = self._missing_ranges(
                start_epoch, end_epoch, present, no_data
            )
            incoming: dict[int, dict] = {}
            complete = True
            failed_ranges = []
            for range_start, range_end in requested_ranges:
                try:
                    fetched = list(fetcher(range_start, range_end) or [])
                    normalized = {}
                    for candle in fetched:
                        row = self._normalize_candle(symbol, candle)
                        if range_start <= row["epoch"] <= range_end:
                            normalized.setdefault(row["epoch"], row)
                    for epoch in range(range_start, range_end + 1, self.timeframe_sec):
                        if epoch in normalized:
                            present.add(epoch)
                            no_data_checks.pop(str(epoch), None)
                            incoming.setdefault(epoch, normalized[epoch])
                        elif epoch not in present:
                            no_data.add(epoch)
                            no_data_checks[str(epoch)] = now
                except Exception:
                    complete = False
                    failed_ranges.append((range_start, range_end))

            merged = dict(existing_by_epoch)
            conflicts = 0
            for epoch, row in incoming.items():
                current = merged.get(epoch)
                if current is None:
                    merged[epoch] = row
                elif not self._same_candle(current, row):
                    conflicts += 1

            retained = [row for epoch, row in sorted(merged.items())
                        if start_epoch <= epoch <= end_epoch]
            self._write_csv_atomic(path, retained)
            self._write_metadata_atomic(
                metadata_path,
                symbol,
                self.timeframe_sec,
                {"present": sorted(present), "no_data": no_data_checks},
            )
            return {
                "candles": retained,
                "complete": complete,
                "requested": len(requested_ranges),
                "failed_ranges": failed_ranges,
                "added": sum(row["epoch"] not in existing_by_epoch for row in incoming.values()),
                "conflicts": conflicts,
                "window_start": start_epoch,
                "window_end": end_epoch,
            }

    def _read_locked(self, path: Path, symbol: str) -> list[dict]:
        if not path.exists():
            return []
        rows, rejected = read_records(path)
        if rejected:
            raise ValueError("histórico local contém registros inválidos; arquivo preservado")
        clean, conflicts, _duplicates = canonicalize(rows)
        if conflicts:
            raise ValueError("histórico local contém candles conflitantes; arquivo preservado")
        if any(not self._is_asset_candle(row, symbol) for row in clean):
            raise ValueError("histórico local contém ativo ou tipo inesperado; arquivo preservado")
        return clean

    def _read_metadata(self, path: Path, symbol: str) -> dict:
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
            if (not isinstance(metadata, dict)
                    or metadata.get("symbol") != symbol
                    or metadata.get("timeframe_sec") != self.timeframe_sec
                    or not isinstance(metadata.get("present"), list)
                    or not isinstance(metadata.get("no_data"), dict)
                    or any(not str(epoch).isdigit() or not isinstance(checked, int)
                           for epoch, checked in metadata.get("no_data", {}).items())):
                return {"present": [], "no_data": {}}
            return metadata
        except (OSError, TypeError, ValueError):
            return {"present": [], "no_data": {}}

    def _missing_ranges(self, start: int, end: int, present: set, no_data: set) -> list[tuple[int, int]]:
        missing = []
        range_start = None
        last_missing = None
        for epoch in range(start, end + 1, self.timeframe_sec):
            if epoch not in present and epoch not in no_data:
                if range_start is None:
                    range_start = epoch
                last_missing = epoch
            elif range_start is not None:
                missing.append((range_start, last_missing))
                range_start = last_missing = None
        if range_start is not None:
            missing.append((range_start, last_missing))

        pages = []
        max_span = self.page_limit * self.timeframe_sec
        for range_start, range_end in missing:
            page_start = range_start
            while page_start <= range_end:
                page_end = min(range_end, page_start + max_span - self.timeframe_sec)
                pages.append((page_start, page_end))
                page_start = page_end + self.timeframe_sec
        return pages

    def _normalize_candle(self, symbol: str, candle: dict) -> dict:
        epoch = int(candle["epoch"])
        values = {key: float(candle[key]) for key in ("open", "high", "low", "close")}
        if (epoch <= 0 or epoch % self.timeframe_sec
                or any(not math.isfinite(value) or value <= 0 for value in values.values())
                or not values["low"] <= min(values["open"], values["close"])
                <= max(values["open"], values["close"]) <= values["high"]):
            raise ValueError("candle inválido recebido da fonte")
        return {
            "epoch": epoch, "datetime": "", "symbol": symbol,
            "price": values["close"], "kind": "candle",
            "timeframe": self.timeframe_sec, "open": values["open"],
            "high": values["high"], "low": values["low"],
        }

    def _is_asset_candle(self, row: dict, symbol: str) -> bool:
        return (row["symbol"] == symbol and row["kind"] == "candle"
                and row["timeframe"] == self.timeframe_sec)

    @staticmethod
    def _same_candle(left: dict, right: dict) -> bool:
        return all(left[key] == right[key] for key in ("price", "open", "high", "low"))

    @staticmethod
    def _write_csv_atomic(path: Path, rows: list[dict]) -> None:
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @staticmethod
    def _write_metadata_atomic(path: Path, symbol: str, timeframe_sec: int, metadata: dict) -> None:
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump({"symbol": symbol, "timeframe_sec": timeframe_sec,
                           "present": metadata["present"], "no_data": metadata["no_data"]},
                          handle, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
