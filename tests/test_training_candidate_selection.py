"""Decisões do Modelo de Treino sem sockets, compras ou alterações de coleta."""
import json
from unittest.mock import Mock

import pytest

import pipeline


def test_ready_candidate_precedes_larger_but_overlapping_dataset(monkeypatch):
    diagnostics = {
        'LARGE': dict(ready_for_training=False, labelable_contracts=1000,
                      evaluation_capacity={'validation': 20, 'test': 100}),
        'READY': dict(ready_for_training=True, labelable_contracts=800,
                      evaluation_capacity={'validation': 31, 'test': 80}),
        'EMPTY': dict(ready_for_training=False, labelable_contracts=0),
    }
    monkeypatch.setattr(pipeline, '_diagnose_training_symbol', lambda symbol: diagnostics[symbol])
    pipeline._shutdown.clear()
    assert pipeline._prioritize_training_candidates(['LARGE', 'EMPTY', 'READY']) == ['READY', 'LARGE', 'EMPTY']


@pytest.mark.parametrize('model_available,rotation_allowed', [(True, True), (False, False)])
def test_shadow_or_pending_reconciliation_prevents_data_rotation(tmp_path, monkeypatch, model_available, rotation_allowed):
    monkeypatch.setattr(pipeline, 'TICKS_CSV', str(tmp_path / 'ticks.csv'))
    monkeypatch.setattr(pipeline, '_load_latest_final_metrics', lambda: {'eligible': True} if model_available else None)
    monkeypatch.setattr(pipeline, '_training_rotation_allowed', rotation_allowed)
    scanner = Mock(side_effect=AssertionError('não deve interromper shadow/reconciliação'))
    monkeypatch.setattr(pipeline, '_detect_top_trending_symbols', scanner)
    pipeline._market_switch.clear()
    pipeline._maybe_request_training_candidate()
    assert not pipeline._market_switch.is_set()
    scanner.assert_not_called()


def test_data_rotation_resumes_only_after_persisted_reconciliation(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, 'TICKS_CSV', str(tmp_path / 'ticks.csv'))
    monkeypatch.setattr(pipeline, '_runtime_symbol', lambda: 'FIRST')
    monkeypatch.setattr(pipeline, '_training_rotation_allowed', False)
    monkeypatch.setattr(pipeline, '_load_latest_final_metrics', lambda: None)
    scanner = Mock(return_value=[])
    monkeypatch.setattr(pipeline, '_detect_top_trending_symbols', scanner)
    pipeline._market_switch.clear()
    state = tmp_path / 'state.json'
    state.write_text(json.dumps(dict(symbol='FIRST', active_contract=dict(phase='buy_uncertain'))))
    pipeline._maybe_request_training_candidate()
    scanner.assert_not_called()
    state.write_text(json.dumps(dict(symbol='FIRST', active_contract=None)))
    pipeline._maybe_request_training_candidate()
    assert pipeline._training_rotation_allowed
    scanner.assert_called_once()


@pytest.mark.parametrize('saved', [[], {'symbol': 'FIRST'}, {'symbol': 'OTHER', 'active_contract': None}])
def test_missing_or_unrelated_contract_state_does_not_authorize_rotation(tmp_path, monkeypatch, saved):
    monkeypatch.setattr(pipeline, 'TICKS_CSV', str(tmp_path / 'ticks.csv'))
    monkeypatch.setattr(pipeline, '_runtime_symbol', lambda: 'FIRST')
    monkeypatch.setattr(pipeline, '_training_rotation_allowed', False)
    (tmp_path / 'state.json').write_text(json.dumps(saved))
    scanner = Mock(side_effect=AssertionError('reconciliação não foi confirmada'))
    monkeypatch.setattr(pipeline, '_detect_top_trending_symbols', scanner)
    pipeline._maybe_request_training_candidate()
    assert not pipeline._training_rotation_allowed
    scanner.assert_not_called()


def test_waiting_symbol_yields_only_to_ready_candidate_and_preserves_runtime(monkeypatch):
    monkeypatch.setattr(pipeline, '_runtime_symbol', lambda: 'WAITING')
    monkeypatch.setattr(pipeline, '_load_latest_final_metrics', lambda: None)
    monkeypatch.setattr(pipeline, '_detect_top_trending_symbols', lambda **kwargs: ['OTHER', 'READY'])
    monkeypatch.setattr(pipeline, '_first_live_symbol_with_compatible_model', lambda candidates: '')
    monkeypatch.setattr(pipeline, '_diagnose_training_symbol', lambda symbol:
                        dict(ready_for_training=symbol == 'READY', labelable_contracts=800 if symbol == 'READY' else 10))
    setter = Mock(side_effect=AssertionError('a troca precisa encerrar a sessão anterior'))
    monkeypatch.setattr(pipeline, '_set_runtime_symbol', setter)
    pipeline._shutdown.clear()
    pipeline._market_switch.clear()
    try:
        pipeline._maybe_request_training_candidate()
        assert pipeline._market_switch.is_set()
        assert pipeline._market_switch_reason == 'training_candidate_ready'
        assert pipeline._training_next_symbol == 'READY'
        assert pipeline._runtime_symbol() == 'WAITING'
    finally:
        pipeline._market_switch.clear()


def test_no_ready_candidate_keeps_collecting_and_surveys_are_bounded(monkeypatch):
    monkeypatch.setattr(pipeline, '_runtime_symbol', lambda: 'WAITING')
    monkeypatch.setattr(pipeline, '_load_latest_final_metrics', lambda: None)
    monkeypatch.setattr(pipeline, '_detect_top_trending_symbols', lambda **kwargs: ['ONE', 'TWO', 'THREE'])
    monkeypatch.setattr(pipeline, '_first_live_symbol_with_compatible_model', lambda candidates: '')
    visited = []
    monkeypatch.setattr(pipeline, '_diagnose_training_symbol', lambda symbol:
                        visited.append(symbol) or dict(ready_for_training=False, labelable_contracts=10))
    clock = [100.]
    monkeypatch.setattr(pipeline.time, 'monotonic', lambda: clock[0])
    pipeline._shutdown.clear()
    pipeline._market_switch.clear()
    pipeline._maybe_request_training_candidate()
    assert set(visited) == {'ONE', 'TWO'}
    visited.clear()
    pipeline._maybe_request_training_candidate()
    assert not visited
    clock[0] += 61
    pipeline._maybe_request_training_candidate()
    assert 'THREE' in visited
    assert not pipeline._market_switch.is_set()


def test_actual_new_ids_trigger_retrain_even_when_total_is_unchanged(monkeypatch):
    monkeypatch.setattr(pipeline, '_runtime_symbol', lambda: 'FIRST')
    monkeypatch.setattr(pipeline, '_retrain_state', dict(last_epoch=0, last_label_count=800))
    monkeypatch.setattr(pipeline, '_count_labelable_contracts_for_training', lambda: 800)
    monkeypatch.setattr(pipeline, '_training_readiness_cache', {
        pipeline._training_cache_key('FIRST'): (0, dict(attempted=True, new_samples=120, readiness_stage='ready'))})
    assert pipeline._should_retrain_now()[0]


def test_previously_evaluated_bundle_requires_new_information_before_retraining(monkeypatch):
    monkeypatch.setattr(pipeline, '_training_outcomes', {
        'FIRST': dict(samples=800, metrics=dict(eligible=True))})
    diagnostic = dict(stage='ready', readiness_stage='ready', symbol='FIRST',
                      labelable_contracts=800, attempted=True, new_samples=0)
    waiting = pipeline._training_status(diagnostic)
    assert not waiting['ready_for_training']
    assert waiting['stage'] == 'awaiting_new_labels'
    renewed = pipeline._training_status(dict(diagnostic, new_samples=120))
    assert renewed['ready_for_training']
    assert renewed['stage'] == 'ready'


def test_diagnostic_keeps_each_symbols_cache_and_hides_label_ids(tmp_path, monkeypatch):
    import csv
    from market_data import FIELDS
    source = tmp_path / 'ticks.csv'
    with source.open('w') as handle:
        csv.writer(handle).writerow(FIELDS)
    monkeypatch.setattr(pipeline, 'TICKS_CSV', str(source))
    monkeypatch.setattr(pipeline.dataset_builder, 'training_diagnostics', lambda path, symbol:
                        dict(stage='insufficient_samples', symbol=symbol, labelable_contracts=1,
                             required_contracts=579, _label_ids=[(100, 100, 1000, 1)]))
    first = pipeline._diagnose_training_symbol('FIRST')
    pipeline._diagnose_training_symbol('SECOND')
    assert len(pipeline._training_readiness_cache) == 2
    assert '_label_ids' not in first
    assert not (tmp_path / '.training_inputs' / 'progress.json').exists()


def test_model_policy_identifies_enabled_generators_and_invalidates_old_policy(monkeypatch):
    import config
    from model_contract import compatible, specification
    metadata = specification('FIRST')
    assert metadata['approval_policy']['version'] == 4
    assert compatible(metadata, 'FIRST')
    old = json.loads(json.dumps(metadata))
    old['approval_policy']['version'] = 3
    assert not compatible(old, 'FIRST')
    monkeypatch.setattr(config, 'PRICE_ACTION_FILTER', not config.PRICE_ACTION_FILTER)
    assert not compatible(metadata, 'FIRST')


def test_training_handoff_uses_selected_candidate_after_cleanup_without_market_penalty(monkeypatch):
    lock = Mock()
    events = []
    monkeypatch.setattr(pipeline, 'acquire_execution_lock', lambda path: lock)
    monkeypatch.setattr(pipeline, '_write_pipeline_status', lambda **kwargs: None)
    monkeypatch.setattr(pipeline, '_finish_market_session', lambda resources: events.append('cleanup'))
    failure = Mock(side_effect=AssertionError('handoff de dados não é falha de mercado'))
    monkeypatch.setattr(pipeline, '_mark_market_failure', failure)

    def session(resources, forced):
        events.append(forced)
        if forced is None:
            monkeypatch.setattr(pipeline, '_training_next_symbol', 'READY')
            monkeypatch.setattr(pipeline, '_market_switch_reason', 'training_candidate_ready')
            raise pipeline.MarketUnavailable('WAITING')

    monkeypatch.setattr(pipeline, '_main_session', session)
    monkeypatch.setattr(pipeline, '_reset_ai_predictor', lambda: None)
    pipeline.main()
    assert events == [None, 'cleanup', 'READY', 'cleanup']
    failure.assert_not_called()
    lock.close.assert_called_once()


def test_completed_training_consumes_exact_dataset_ids(tmp_path, monkeypatch):
    import csv
    import pandas as pd
    from market_data import FIELDS
    source = tmp_path / 'ticks.csv'
    with source.open('w') as handle:
        csv.writer(handle).writerow(FIELDS)
    monkeypatch.setattr(pipeline, 'TICKS_CSV', str(source))
    monkeypatch.setattr(pipeline, '_runtime_symbol', lambda: 'FIRST')
    monkeypatch.delattr(pipeline._run_training, '_last_dataset_key', raising=False)
    rows = [dict(signal_epoch=1000 + i * 900, entry_epoch=1000 + i * 900,
                 expiry_epoch=1900 + i * 900, target=i % 2) for i in range(800)]

    def build(snapshot, output, **kwargs):
        pd.DataFrame(rows).to_csv(output, index=False)
        return len(rows)

    monkeypatch.setattr(pipeline.dataset_builder, 'build_dataset', build)
    monkeypatch.setattr(pipeline.train_model, 'train', lambda *args, **kwargs:
                        dict(eligible=False, best_model='offline', rejection_reasons=['taxa de acerto baixa']))
    assert pipeline._run_training(tmp_path / 'dataset.csv', str(tmp_path / 'model.pkl'))[0]
    current = rows[120:] + [dict(signal_epoch=1000 + i * 900, entry_epoch=1000 + i * 900,
                                expiry_epoch=1900 + i * 900, target=i % 2) for i in range(800, 920)]
    ids = [tuple(row[key] for key in ('signal_epoch', 'entry_epoch', 'expiry_epoch', 'target')) for row in current]
    assert pipeline._training_progress().observe('FIRST', ids)['new_samples'] == 120
