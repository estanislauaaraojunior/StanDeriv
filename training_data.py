"""Preparação offline de dados da IA, sem escrever nas fontes operacionais."""
import argparse
import csv
import fcntl
import json
import shutil
import tempfile
from pathlib import Path

from market_data import FIELDS, canonicalize, read_records
from model_contract import fingerprint, write_json_atomic

# Independente dos sete dias e 336 candles da coleta homologada.
TRAINING_HISTORY_DAYS = 28
PREPARATION_VERSION = 1


class TrainingDataError(ValueError):
    """Dados cujo intervalo afetado não pode ser determinado com segurança."""


def load_records(path, symbol):
    """Isola o ativo antes de validar; preserva os horários excluídos para veto temporal.

    Registros inválidos com época conhecida são colocados em quarentena. Nenhuma
    amostra que alcance essa época pode ser usada. Corrupção sem época atribuível
    ao ativo bloqueia a preparação inteira. O parser de preços permanece canônico.
    """
    path = Path(path)
    selected_lines = []
    quarantine = []
    excluded = []
    other_symbols = 0
    with tempfile.TemporaryDirectory(prefix='standeriv-training-read-') as temp:
        filtered = Path(temp) / 'selected.csv'
        with path.open(newline='', encoding='utf-8-sig') as source, filtered.open('w', newline='') as target:
            reader, writer = csv.reader(source), csv.writer(target)
            header = next(reader, None)
            if header != FIELDS:
                raise TrainingDataError('esquema inválido ou dados legados sem proveniência')
            writer.writerow(FIELDS)
            for line, values in enumerate(reader, 2):
                if not values or values == FIELDS:
                    continue
                primary_symbol = values[2].strip() if len(values) > 2 else ''
                relevant = primary_symbol == symbol or symbol in values
                if not relevant:
                    other_symbols += 1
                    continue
                try:
                    epoch_float = float(values[0])
                    epoch = int(epoch_float)
                    if (primary_symbol != symbol or epoch != epoch_float
                            or not 0 < epoch <= 253402300799):
                        raise ValueError()
                except (ValueError, OverflowError, IndexError):
                    raise TrainingDataError(
                        f'registro inválido de {symbol} sem época segura na linha {line}'
                    ) from None
                if len(values) != len(FIELDS):
                    quarantine.append(dict(line=line, reason='quantidade de colunas inválida', fields=values))
                    excluded.append(epoch)
                    continue
                writer.writerow(values)
                selected_lines.append((line, epoch))
        records, rejected = read_records(filtered)
        for item in rejected:
            original_line, epoch = selected_lines[item['line'] - 2]
            quarantine.append(dict(item, line=original_line))
            excluded.append(epoch)
    clean, conflicts, duplicates = canonicalize(records)
    excluded.extend(record['epoch'] for record in conflicts)
    report = dict(version=PREPARATION_VERSION, symbol=symbol,
                  history_days=TRAINING_HISTORY_DAYS, other_symbol_records=other_symbols,
                  rejected_records=len(quarantine), conflicting_records=len(conflicts),
                  duplicates=duplicates, excluded_epochs=sorted(set(excluded)),
                  quarantine=quarantine, conflicts=conflicts)
    # Um snapshot recuperado continua carregando os vetos temporais da origem.
    sidecar = Path(str(path) + '.preparation.json')
    if sidecar.exists():
        previous = json.loads(sidecar.read_text())
        if previous.get('snapshot_sha256') != fingerprint(path) or previous.get('symbol') != symbol:
            raise TrainingDataError('snapshot de treino difere do relatório de preparação')
        report['excluded_epochs'] = sorted(set(report['excluded_epochs'] + previous['excluded_epochs']))
        report['source_preparation'] = previous
    return clean, report


def prepare_snapshot(source, destination, symbol, *, history_path=None, enricher=None):
    """Cria diretório novo com snapshot estável por ativo, quarentena e hashes."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix='standeriv-training-source-') as temp:
        raw = Path(temp) / 'source.csv'
        # O lock é mantido somente durante a cópia, nunca durante o parsing.
        with Path(source).open('rb') as handle, raw.open('wb') as target:
            fcntl.flock(handle, fcntl.LOCK_SH)
            shutil.copyfileobj(handle, target)
        records, report = load_records(raw, symbol)
        source_sidecar = Path(str(source) + '.preparation.json')
        if source_sidecar.exists():
            prior = json.loads(source_sidecar.read_text())
            if prior.get('snapshot_sha256') != fingerprint(raw) or prior.get('symbol') != symbol:
                raise TrainingDataError('origem recuperada difere do relatório')
            report['excluded_epochs'] = sorted(set(report['excluded_epochs'] + prior['excluded_epochs']))
            report['source_preparation'] = prior
        report['source_sha256'] = fingerprint(raw)
    if history_path is not None and Path(history_path).exists():
        with tempfile.TemporaryDirectory(prefix='standeriv-training-ohlc-') as temp:
            raw = Path(temp) / 'ohlc.csv'
            # O histórico é substituído atomicamente pelo escritor existente.
            shutil.copyfile(history_path, raw)
            history, rejected = read_records(raw)
            if rejected or any(r['symbol'] != symbol or r['kind'] != 'candle' for r in history):
                raise TrainingDataError('histórico OHLC inválido para o ativo')
            history, conflicts, _ = canonicalize(history)
            if conflicts:
                raise TrainingDataError('histórico OHLC contém conflitos')
            report['history_sha256'] = fingerprint(raw)
        records, conflicts, duplicates = canonicalize([*records, *history])
        report['excluded_epochs'] = sorted(set(report['excluded_epochs'] + [r['epoch'] for r in conflicts]))
        report['conflicts'].extend(conflicts)
        report['conflicting_records'] += len(conflicts)
        report['duplicates'] += duplicates
    if enricher is not None:
        records, report['history_recovery'] = enricher(records, report)
    snapshot = destination / 'prices.csv'
    with snapshot.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, FIELDS)
        writer.writeheader()
        writer.writerows(records)
    report.update(snapshot_sha256=fingerprint(snapshot), records=len(records))
    write_json_atomic(str(snapshot) + '.preparation.json', report)
    return snapshot, report


def main():
    """CLI exclusivamente offline; nunca substitui a fonte ou publica modelos."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--symbol', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--history')
    args = parser.parse_args()
    snapshot, report = prepare_snapshot(args.input, args.output_dir, args.symbol, history_path=args.history)
    print(json.dumps(dict(snapshot=str(snapshot), records=report['records'],
                          rejected_records=report['rejected_records'],
                          conflicting_records=report['conflicting_records']), ensure_ascii=False))


if __name__ == '__main__':
    main()
