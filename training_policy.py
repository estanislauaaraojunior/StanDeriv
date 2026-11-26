"""Política exclusiva do bootstrap da IA; não altera critérios de aprovação."""
from copy import deepcopy
import json
import math
from pathlib import Path
import threading

from model_contract import write_json_atomic


_PROGRESS_VERSION = 1
# Mais de três janelas de 28 dias/15 minutos; o descarte mantém um watermark.
_MAX_RETAINED_LABELS = 8192


def _label_ids(labels):
    """Normaliza identidades reais, sem inferir contratos a partir de contagens."""
    result = set()
    for label in labels:
        if not isinstance(label, (tuple, list)) or len(label) != 4:
            raise ValueError('identidade de rótulo deve conter sinal, entrada, vencimento e target')
        normalized = []
        for value in label:
            if isinstance(value, bool):
                raise ValueError('identidade de rótulo inválida')
            integer = int(value)
            if integer != value:
                raise ValueError('identidade de rótulo inválida')
            normalized.append(integer)
        signal, entry, expiry, target = normalized
        if not signal <= entry < expiry or target not in (0, 1, 2):
            raise ValueError('cronologia ou target do rótulo inválidos')
        result.add(tuple(normalized))
    return result


class TrainingProgress:
    """Rótulos já usados em tentativas, isolados por símbolo e política.

    Observar não consome dados. Uma tentativa consome exatamente seu snapshot,
    inclusive se a recuperação tiver encontrado outros rótulos durante o treino.
    IDs antigos são compactados com um limite de época: não voltam a contar
    como novidade se aparecerem novamente depois de sair da janela móvel.
    """

    def __init__(self, path, identity=None):
        self.path = Path(path)
        self.identity = json.loads(json.dumps(identity, sort_keys=True, allow_nan=False))
        self._lock = threading.RLock()
        self._state = dict(version=_PROGRESS_VERSION, identity=self.identity, symbols={})
        if self.path.exists():
            try:
                state = json.loads(self.path.read_text())
                if not isinstance(state, dict):
                    raise ValueError('estado deve ser um objeto')
                if state.get('version') != _PROGRESS_VERSION or state.get('identity') != self.identity:
                    return
                if not isinstance(state.get('symbols'), dict):
                    raise ValueError('símbolos inválidos')
                for symbol, saved in state['symbols'].items():
                    if (not isinstance(symbol, str) or not symbol or not isinstance(saved, dict)
                            or not isinstance(saved.get('metrics'), dict)
                            or not isinstance(saved.get('previous_samples'), int)
                            or saved['previous_samples'] < 0
                            or not isinstance(saved.get('retired_through'), int)):
                        raise ValueError('progresso de símbolo inválido')
                    consumed = _label_ids(saved['consumed'])
                    if (len(consumed) > _MAX_RETAINED_LABELS
                            or any(label[0] <= saved['retired_through'] for label in consumed)):
                        raise ValueError('retenção de rótulos inválida')
                self._state = state
            except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
                raise ValueError('progresso de treino inválido; não é seguro inferir tentativas anteriores') from exc

    def observe(self, symbol, label_ids):
        """Conta IDs disponíveis ainda não usados, mesmo com total de linhas estável."""
        labels = _label_ids(label_ids)
        with self._lock:
            saved = self._state['symbols'].get(symbol)
            if saved is None:
                return dict(new_samples=len(labels), attempted=False, previous_samples=0, metrics={}, revision=0)
            consumed = {tuple(label) for label in saved['consumed']}
            fresh = {label for label in labels if label[0] > saved['retired_through']} - consumed
            return dict(new_samples=len(fresh), attempted=True,
                        previous_samples=saved['previous_samples'], metrics=deepcopy(saved['metrics']),
                        revision=saved.get('revision', 0))

    def mark_attempt(self, symbol, label_ids, metrics):
        """Grava uma tentativa concluída sem consumir observações posteriores."""
        if not isinstance(symbol, str) or not symbol or not isinstance(metrics, dict):
            raise ValueError('símbolo e métricas são obrigatórios')
        labels = _label_ids(label_ids)
        with self._lock:
            saved = self._state['symbols'].get(symbol, {})
            retired = saved.get('retired_through', -1)
            consumed = {tuple(label) for label in saved.get('consumed', [])} | labels
            consumed = sorted(label for label in consumed if label[0] > retired)
            if len(consumed) > _MAX_RETAINED_LABELS:
                retired = max(retired, consumed[-_MAX_RETAINED_LABELS - 1][0])
                consumed = [label for label in consumed if label[0] > retired]
            state = deepcopy(self._state)
            state['symbols'][symbol] = dict(consumed=consumed, retired_through=retired,
                                           previous_samples=len(labels), metrics=deepcopy(metrics),
                                           revision=saved.get('revision', 0) + 1)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            write_json_atomic(self.path, state)
            self._state = state
            return self.observe(symbol, labels)


def evaluation_capacity(frame):
    """Limite superior de contratos sem sobreposição, sem assumir sinais ou lucro."""
    count, available = 0, -math.inf
    for row in frame.sort_values('expiry_epoch').itertuples():
        if row.entry_epoch >= available:
            count += 1
            available = row.expiry_epoch
    return count


def retry_plan(labels, previous, metrics, default=120, *, new_samples=None):
    """Problemas só de amostra permitem lote menor; desempenho fraco exige renovação."""
    reasons = metrics.get('rejection_reasons') or []
    sample_only = bool(reasons) and all('amostra' in r.lower() for r in reasons)
    batch = max(30, min(default, math.ceil(previous * .05))) if sample_only else default
    target = previous + batch
    fresh = max(0, labels-previous) if new_samples is None else max(0, int(new_samples))
    return dict(next_training_at=target if new_samples is None else None,
                remaining_samples=max(0, batch-fresh),
                new_samples=fresh, retry_batch=batch)


def readiness_capacity(frame, required=579, test_ratio=.2):
    """Preflight temporal leve; prontidão não implica sinais nem aprovação do modelo."""
    result = dict(stage='insufficient_samples', labelable_contracts=len(frame),
                  required_contracts=required, evaluation_capacity={})
    if len(frame) < required:
        result['reason'] = f'amostra insuficiente: {len(frame)}/{required} contratos rotuláveis'
        return result
    from train_model import temporal_partitions, walk_forward_partitions, fit_calibration_partition
    try:
        _, _, validation, test = temporal_partitions(frame, test_ratio)
    except ValueError as exc:
        result['reason'] = str(exc)
        return result
    capacities = dict(validation=evaluation_capacity(validation), test=evaluation_capacity(test))
    result['evaluation_capacity'] = capacities
    if min(capacities.values()) < 30:
        result.update(stage='evaluation_capacity',
                      reason='amostra temporal insuficiente: menos de 30 oportunidades independentes na validação/teste')
    else:
        development = frame[frame.expiry_epoch < int(test.feature_start_epoch.iloc[0])]
        try:
            folds = 1 if len(development) < 1000 else 3
            list(walk_forward_partitions(development, folds=folds))
            fit_calibration_partition(development)
        except ValueError as exc:
            result.update(stage='evaluation_capacity', reason=str(exc))
        else:
            result['stage'] = 'ready'
    return result
