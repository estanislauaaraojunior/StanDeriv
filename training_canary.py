"""Canary fixo da IA isolada; preserva o ledger original de 30 resultados."""
import hashlib
import json
import math
import time
from datetime import datetime
from pathlib import Path

import config
from model_contract import MAX_TICK_GAP_SEC
from shadow_monitor import ShadowMonitor


def policy():
    return dict(version='isolated_ai_canary_v1', results=8, minimum_wins=7,
                positive_unit_profit=True, minimum_historical_trades=30,
                minimum_confidence=config.AI_CONFIDENCE_MIN,
                minimum_net_payout=config.SHADOW_MIN_NET_PAYOUT)


def _path(path, identity):
    encoded = json.dumps(dict(identity, policy=policy()), sort_keys=True).encode()
    return Path(str(path) + '.canary') / (hashlib.sha256(encoded).hexdigest() + '.json')


def historical_evidence(metadata, identity):
    """Extrai prova do holdout; não reavalia nem transforma histórico em forward."""
    try:
        metrics = metadata['metrics']
        digest = metadata['dataset_sha256']
        model_id = metadata.get('model_id') or metadata.get('artifact_sha256')
        trained = datetime.fromisoformat(metadata['trained_at']).timestamp()
        proof = {key: metrics[key] for key in (
            'evaluation_start', 'evaluation_end', 'test_first_feature',
            'fit_last_expiry', 'calibration_first_feature', 'calibration_last_expiry')}
        if (metadata.get('approved') is not True or metrics.get('eligible') is not True
                or metrics.get('holdout_used_for_selection') is not False
                or model_id != identity['model_id'] or metadata['symbol'] != identity['symbol']
                or metadata['timeframe_sec'] != 300 or identity['timeframe_sec'] != 300
                or identity['duration_sec'] != 900
                or metadata['duration'] != 15 or metadata['duration_unit'] != 'm'
                or not isinstance(digest, str) or len(digest) != 64
                or any(c not in '0123456789abcdef' for c in digest)
                or not isinstance(metrics['trades'], int) or isinstance(metrics['trades'], bool)
                or not isinstance(metrics['wins'], int) or isinstance(metrics['wins'], bool)
                or metrics['trades'] < 30 or not 0 <= metrics['wins'] <= metrics['trades']
                or not all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0
                           for v in proof.values())
                or not proof['fit_last_expiry'] < proof['calibration_first_feature']
                or not proof['calibration_last_expiry'] < proof['test_first_feature']
                or not proof['test_first_feature'] <= proof['evaluation_start']
                or not proof['evaluation_start'] < proof['evaluation_end'] <= trained):
            return None
        return dict(proof, dataset_sha256=digest, historical_trades=metrics['trades'],
                    historical_wins=metrics['wins'], trained_at=metadata['trained_at'])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def _unreachable_epoch(results):
    """Primeiro vencimento que tornou impossível atingir os sete acertos."""
    wins = 0
    for count, row in enumerate(results, start=1):
        wins += int(row['correct'])
        if wins + policy()['results'] - count < policy()['minimum_wins']:
            return row['exit_epoch']
    return None


def _reject_unreachable(state):
    """Normaliza o término sem apagar resultados/pending ou rever decisões finais."""
    if state.get('approved_at_epoch') is not None or state.get('rejected_at_epoch') is not None:
        return False
    epoch = _unreachable_epoch(state['results'])
    if epoch is None:
        return False
    state.update(rejected_at_epoch=epoch, rejection_reason='minimum_wins_unreachable')
    return True


class CanaryMonitor(ShadowMonitor):
    """Uma sequência de oito, persistida por bundle/política, sem janela móvel."""
    SCHEMA_VERSION = 2

    def __init__(self, path, *, evidence, now=time.time, **identity):
        self.identity = dict(identity, policy=policy())
        self.path = _path(path, identity)
        self.minimum_results, self.minimum_accuracy = 8, 7 / 8
        self.minimum_net_payout = policy()['minimum_net_payout']
        self.invalid = False
        if self.path.exists():
            state = self._read_state(self.path)
            if self._valid_state(state, evidence):
                self.state = state
                if _reject_unreachable(self.state):
                    self._save_state()
                return
            # Não apaga nem reinicia uma tentativa danificada até encontrar aprovação.
            self.invalid = True
        self.state = dict(schema_version=2, **self.identity, evidence=evidence,
                          frozen_at_epoch=int(now()), pending=None, results=[],
                          approved_at_epoch=None, rejected_at_epoch=0 if self.invalid else None)
        if not self.invalid:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._save_state()

    def _valid_state(self, state, evidence):
        try:
            if (not isinstance(state, dict) or state.get('schema_version') != 2
                    or any(state.get(k) != v for k, v in self.identity.items())
                    or evidence is None or state['evidence'] != evidence
                    or not isinstance(state['frozen_at_epoch'], int)
                    or state['frozen_at_epoch'] <= 0 or not isinstance(state['results'], list)
                    or len(state['results']) > 8):
                return False
            reconstructed = dict(
                **{k: self.identity[k] for k in ('symbol', 'model_id', 'timeframe_sec')},
                duration=15, duration_unit='m', approved=True,
                trained_at=evidence['trained_at'], dataset_sha256=evidence['dataset_sha256'],
                metrics=dict(evidence, eligible=True, holdout_used_for_selection=False,
                             trades=evidence['historical_trades'], wins=evidence['historical_wins']))
            if (historical_evidence(reconstructed, self.identity) != evidence
                    or datetime.fromisoformat(evidence['trained_at']).timestamp() > state['frozen_at_epoch']):
                return False
            previous_expiry = state['frozen_at_epoch']
            for row in [*state['results'], *([state['pending']] if state.get('pending') else [])]:
                if (row['direction'] not in (0, 1)
                        or not isinstance(row['entry_epoch'], int)
                        or row['entry_epoch'] < previous_expiry
                        or row['expiry_epoch'] != row['entry_epoch'] + 900
                        or not all(math.isfinite(row[k]) for k in
                                   ('entry_price', 'confidence', 'net_payout'))
                        or row['entry_price'] <= 0
                        or not policy()['minimum_confidence'] < row['confidence'] <= 1
                        or row['net_payout'] < self.minimum_net_payout):
                    return False
                if row is not state.get('pending'):
                    if (not 0 <= row['exit_epoch'] - row['expiry_epoch'] <= MAX_TICK_GAP_SEC
                            or not math.isfinite(row['exit_price']) or row['exit_price'] <= 0
                            or row['exit_price'] == row['entry_price']
                            or row['correct'] is not
                            ((row['exit_price'] > row['entry_price']) == bool(row['direction']))):
                        return False
                previous_expiry = row['expiry_epoch']
            complete = len(state['results']) == 8
            passed = complete and sum(r['correct'] for r in state['results']) >= 7
            profit = sum(r['net_payout'] if r['correct'] else -1 for r in state['results'])
            passed = passed and profit > 0
            terminal_epoch = state['results'][-1]['exit_epoch'] if complete else None
            if not complete and state.get('rejected_at_epoch') is not None:
                unreachable = _unreachable_epoch(state['results'])
                return (state.get('approved_at_epoch') is None and unreachable is not None
                        and state.get('rejected_at_epoch') == unreachable
                        and state.get('rejection_reason') == 'minimum_wins_unreachable')
            return (state.get('approved_at_epoch') == (terminal_epoch if passed else None)
                    and state.get('rejected_at_epoch') == (terminal_epoch if complete and not passed else None)
                    and (not complete or state.get('pending') is None))
        except (KeyError, TypeError, ValueError, OverflowError):
            return False

    @property
    def unit_profit(self):
        return sum(row['net_payout'] if row['correct'] else -1 for row in self.state['results'])

    def observe_prediction(self, **observation):
        if observation['entry_epoch'] < self.state['frozen_at_epoch']:
            return 'before_policy_freeze'
        return super().observe_prediction(**observation)

    def settle_tick(self, **tick):
        if self.invalid:
            return 'rejected'
        if self.rejected:
            # Pendências simuladas preservadas no término não reabrem o ledger.
            return 'no_pending'
        result = super().settle_tick(**tick)
        if _reject_unreachable(self.state):
            self._save_state()
            return 'rejected'
        if self.result_count == 8 and self.approved and self.unit_profit <= 0:
            self.state['approved_at_epoch'] = None
            self.state['rejected_at_epoch'] = self.state['results'][-1]['exit_epoch']
            self._save_state()
            return 'rejected'
        return result


class ParallelShadow:
    """Mantém a validação original em andamento sem importar seus resultados no canary."""
    def __init__(self, legacy, canary):
        self.legacy, self.canary = legacy, canary
        self._rejection_reported = False

    @property
    def _display_monitor(self):
        return self.legacy if self.approved and self.legacy.approved and not self.canary.approved else self.canary

    @property
    def minimum_results(self):
        return self._display_monitor.minimum_results

    @property
    def minimum_accuracy(self):
        return self._display_monitor.minimum_accuracy

    @property
    def minimum_net_payout(self):
        return self.canary.minimum_net_payout

    @property
    def state(self):
        return self.canary.state

    @property
    def approved(self):
        return not self.rejected and (self.canary.approved or self.legacy.approved)

    @property
    def rejected(self):
        return self.canary.rejected or self.legacy.rejected

    @property
    def result_count(self):
        return self._display_monitor.result_count

    @property
    def accuracy(self):
        return self._display_monitor.accuracy

    def diagnostics(self):
        wins = sum(row['correct'] for row in self.canary.state['results'])
        remaining = self.canary.minimum_results - self.canary.result_count
        status = dict(policy=policy(), unit_profit=self.canary.unit_profit,
                    approval_path=('legacy_30' if self.approved and self.legacy.approved and not self.canary.approved
                                   else 'canary_8'), canary_result_count=self.canary.result_count,
                    legacy_result_count=self.legacy.result_count,
                    legacy_pending=bool(self.legacy.state.get('pending')),
                    frozen_at_epoch=self.canary.state['frozen_at_epoch'],
                    historical_trades=(self.canary.state.get('evidence') or {}).get('historical_trades'),
                    wins=wins, remaining_results=remaining,
                    maximum_possible_wins=wins + remaining,
                    rejection_reason=self.canary.state.get('rejection_reason'),
                    invalid_canary_state=self.canary.invalid)
        if self.rejected:
            # Uma captura no mesmo fechamento não transforma uma reprovação
            # já publicada em "aguardando" novamente.
            status['reason'] = 'rejected'
        return status

    def observe_prediction(self, **observation):
        if self.approved or self.rejected or self.legacy.state.get('pending'):
            return 'already_pending_or_approved'
        result = self.canary.observe_prediction(**observation)
        if result == 'pending':
            self.legacy.observe_prediction(**observation)
        return result

    def settle_tick(self, **tick):
        legacy_result = result = 'no_pending'
        if not self.rejected:
            legacy_result = self.legacy.settle_tick(**tick)
            result = self.canary.settle_tick(**tick)
        if self.rejected:
            if not self._rejection_reported:
                self._rejection_reported = True
                return 'rejected'
            return 'no_pending'
        if self.approved:
            return 'approved' if result == 'approved' or legacy_result == 'approved' else 'no_pending'
        return legacy_result if result == 'no_pending' else result


def saved_canary(path, **identity):
    """Consulta sem criar arquivo; estado danificado não autoriza recomeçar o bundle."""
    target = _path(path, identity)
    if not target.exists():
        return None
    state = ShadowMonitor._read_state(target)
    if not isinstance(state, dict):
        return dict(rejected_at_epoch=0)
    validator = object.__new__(CanaryMonitor)
    validator.identity = dict(identity, policy=policy())
    validator.minimum_net_payout = policy()['minimum_net_payout']
    if not validator._valid_state(state, state.get('evidence')):
        return dict(rejected_at_epoch=0)
    # Consulta do pipeline permanece somente leitura. Um estado antigo
    # inviável já bloqueia retomada antes de o executor carregá-lo e persistir.
    _reject_unreachable(state)
    return state


def create_monitor(path, *, evidence=None, now=time.time, **identity):
    legacy = ShadowMonitor(path, **identity)
    if identity['timeframe_sec'] != 300 or identity['duration_sec'] != 900:
        return legacy
    target = _path(path, identity)
    proof = historical_evidence(evidence or {}, identity)
    if proof and datetime.fromisoformat(proof['trained_at']).timestamp() > now():
        proof = None
    if target.exists():
        if proof is None:
            saved = saved_canary(path, **identity)
            if saved is None or saved.get('rejected_at_epoch') is None:
                # Prova histórica momentaneamente não verificável (relógio,
                # metadados incompletos): não é reprovação. O arquivo canary fica
                # intacto e é retomado quando a prova voltar a conferir; até lá
                # vale somente a validação original.
                legacy.policy_reason = 'historical_holdout_not_verified'
                return legacy
            # Reprovação persistida ou arquivo danificado continuam terminais.
        return ParallelShadow(legacy, CanaryMonitor(path, evidence=proof, now=now, **identity))
    if legacy.approved or legacy.rejected:
        return legacy
    if proof is None:
        legacy.policy_reason = 'historical_holdout_not_verified'
        return legacy
    # Ledger de admissão: um mesmo período não é reutilizado para tentar outros
    # bundles até encontrar um aprovado. Outra identidade retoma seu próprio estado.
    for previous in target.parent.glob('*.json'):
        state = ShadowMonitor._read_state(previous)
        if not isinstance(state, dict):
            legacy.policy_reason = 'historical_admission_ledger_invalid'
            return legacy
        if (state.get('symbol') == identity['symbol'] and state.get('policy') == policy()
                and state.get('model_id') != identity['model_id']):
            old = state.get('evidence')
            if not isinstance(old, dict) or not all(
                    isinstance(old.get(k), (int, float)) and math.isfinite(old[k])
                    for k in ('test_first_feature', 'evaluation_end')):
                legacy.policy_reason = 'historical_admission_ledger_invalid'
                return legacy
            if (max(old['test_first_feature'], proof['test_first_feature'])
                    <= min(old['evaluation_end'], proof['evaluation_end'])):
                legacy.policy_reason = 'historical_holdout_already_consumed'
                return legacy
    return ParallelShadow(legacy, CanaryMonitor(path, evidence=proof, now=now, **identity))
