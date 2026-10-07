"""Leitura explícita, recuperação não destrutiva e persistência de preços."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

FIELDS = ['epoch', 'datetime', 'symbol', 'price', 'kind', 'timeframe', 'open', 'high', 'low']


def read_records(path):
    """Legado mantém proveniência desconhecida; nunca inferir ticks pelo intervalo."""
    records, rejected = [], []
    with Path(path).open(newline = '', encoding = 'utf-8-sig') as handle:
        reader = csv.reader(handle)
        header = None
        for line, values in enumerate(reader, 1):
            if not values:
                continue
            if values[0].strip().lower() == 'epoch':
                header = values
                continue
            try:
                if len(values) != len(header or FIELDS[: 4]):
                    raise ValueError('quantidade de colunas inválida')
                row = dict(zip(header or FIELDS[: 4], values))
                epoch_float = float(row['epoch'])
                epoch = int(epoch_float)
                price = float(row['price'])
                symbol = row['symbol'].strip()
                kind = row.get('kind', 'legacy_unknown') or 'legacy_unknown'
                if epoch != epoch_float or epoch <= 0 or not symbol or not math.isfinite(price) or price <= 0:
                    raise ValueError('preço, símbolo ou horário inválido')
                if kind not in {'tick', 'candle', 'legacy_unknown'}:
                    raise ValueError('proveniência inválida')
                item = dict(epoch = epoch, datetime = row.get('datetime', ''), symbol = symbol,
                            price = price, kind = kind, timeframe = int(row.get('timeframe') or 0),
                            open = '', high = '', low = '')
                if kind == 'candle':
                    for key in ('open', 'high', 'low'):
                        item[key] = float(row[key])
                    if item['timeframe'] <= 0 or any(not math.isfinite(item[k]) or item[k] <= 0 for k in ('open', 'high', 'low')):
                        raise ValueError('OHLC inválido')
                    if not item['low'] <= min(item['open'], price) <= max(item['open'], price) <= item['high']:
                        raise ValueError('OHLC inconsistente')
                records.append(item)
            except (KeyError, ValueError, OverflowError, TypeError):
                rejected.append({'line': line, 'reason': 'registro inválido', 'fields': values})
    return records, rejected


def canonicalize(records):
    """Duplicatas idênticas são removidas; conflitos ficam fora da série utilizável."""
    unique, conflicts, blocked = {}, [], set()
    duplicates = 0
    for row in records:
        key = (row['symbol'], row['epoch'], row['kind'], row['timeframe'])
        if key in blocked:
            conflicts.append(row)
            continue
        if key in unique:
            old = unique[key]
            if all(old[k] == row[k] for k in ('price', 'open', 'high', 'low')):
                duplicates += 1
            else:
                conflicts.extend([unique.pop(key), row])
                blocked.add(key)
        else:
            unique[key] = row
    return sorted(unique.values(), key = lambda r: (r['symbol'], r['epoch'], r['kind'])), conflicts, duplicates


def ensure_header(path):
    """Garante o cabeçalho v2 no CSV (criando-o se vazio); falha em esquema legado."""
    import fcntl
    with Path(path).open('a+', newline = '') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        header = next(csv.reader(f), None)
        if header is None:
            csv.writer(f).writerow(FIELDS)
        elif header != FIELDS:
            raise ValueError('esquema legado; recupere em outro arquivo antes de gravar')
        f.flush()


def append_record(path, epoch, symbol, price, * , kind = 'tick', timeframe = 0, candle = None):
    """Anexa um registro (tick ou candle) ao CSV v2 sob lock exclusivo, criando o cabeçalho se necessário."""
    import fcntl
    candle = candle or {}
    with Path(path).open('a+', newline = '') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        header = next(csv.reader(f), None)
        if header is None:
            csv.writer(f).writerow(FIELDS)
        elif header != FIELDS:
            raise ValueError('esquema legado; gravar somente em um arquivo v2')
        f.seek(0, 2)
        csv.writer(f).writerow([epoch, '', symbol, price, kind, timeframe,
                               candle.get('open', ''), candle.get('high', ''), candle.get('low', '')])
        f.flush()



def recover(source, output_dir):
    """Escreve uma nova recuperação; não substitui arquivos existentes."""
    source, dest = Path(source), Path(output_dir)
    if dest.exists():
        raise ValueError('diretório de recuperação já existe')
    # Snapshot em memória: checksum e leitura descrevem exatamente os mesmos bytes.
    raw = source.read_bytes()
    dest.mkdir(parents = True)
    snapshot = dest / 'original.csv'
    snapshot.write_bytes(raw)
    records, rejected = read_records(snapshot)
    clean, conflicts, duplicates = canonicalize(records)
    with (dest / 'prices.csv').open('w', newline = '') as f:
        writer = csv.DictWriter(f, FIELDS)
        writer.writeheader()
        writer.writerows(clean)
    report = {'source': str(source), 'sha256': hashlib.sha256(raw).hexdigest(),
              'valid_input': len(records), 'recovered': len(clean), 'duplicates': duplicates,
              'symbols': dict(Counter(r['symbol'] for r in clean)),
              'provenance': dict(Counter(r['kind'] for r in clean)),
              'rejected': len(rejected), 'conflicting_records': len(conflicts),
              'training_eligible': sum(r['kind'] == 'tick' for r in clean)}
    (dest / 'quarantine.json').write_text(json.dumps({'invalid': rejected, 'conflicts': conflicts}, indent = 2))
    (dest / 'report.json').write_text(json.dumps(report, indent = 2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Recupera CSV sem modificar a origem')
    parser.add_argument('--input', required = True)
    parser.add_argument('--output-dir', required = True)
    args = parser.parse_args()
    print(json.dumps(recover(args.input, args.output_dir), indent = 2))
