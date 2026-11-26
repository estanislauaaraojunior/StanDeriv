"""Recuperação isolada e continuidade de features/rótulos da IA."""
import csv
import json
from pathlib import Path

import pytest

import config
import dataset_builder
from market_data import FIELDS, read_records
from model_contract import FEATURE_WINDOW, compatible, minimum_history_candles, specification
from training_data import TrainingDataError, load_records, prepare_snapshot

BASE = 1_800_000_000
TF = 900


def write_source(path, *, gap=None, history=True, corrupt_epoch=None):
    """Escreve OHLC confirmado e ticks de entrada/vencimento em diretório temporário."""
    with path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        if history:
            for i in range(FEATURE_WINDOW + 3):
                writer.writerow([BASE + i * TF, '', 'R_100', 100 + i * .01,
                                 'candle', TF, 100, 102, 99])
        for offset in range((FEATURE_WINDOW - 1) * TF, (FEATURE_WINDOW + 4) * TF, 2):
            if gap and gap[0] < offset < gap[1]:
                continue
            writer.writerow([BASE + offset, '', 'R_100', 100 + offset * .00001,
                             'tick', 0, '', '', ''])
        if corrupt_epoch is not None:
            writer.writerow([corrupt_epoch, '', 'R_100', 'nan', 'tick', 0, '', '', ''])


def test_unrelated_corruption_does_not_block_selected_symbol(tmp_path):
    source = tmp_path / 'ticks.csv'
    write_source(source)
    before = dataset_builder.count_labelable_contracts(source, 'R_100')
    assert before > 0
    with source.open('a', newline='') as handle:
        csv.writer(handle).writerow([BASE + 123, '', 'other', 'nan', 'candle', TF, '', '', ''])
    assert dataset_builder.count_labelable_contracts(source, 'R_100') == before


def test_quarantine_blocks_overlapping_samples_and_survives_snapshot(tmp_path):
    source = tmp_path / 'ticks.csv'
    write_source(source)
    healthy = dataset_builder.count_labelable_contracts(source, 'R_100')
    with source.open('a', newline='') as handle:
        csv.writer(handle).writerow([BASE + 50 * TF + 100, '', 'R_100', 'nan', 'tick', 0, '', '', ''])
    original = source.read_bytes()
    direct = dataset_builder.count_labelable_contracts(source, 'R_100')
    assert direct < healthy
    snapshot, report = prepare_snapshot(source, tmp_path / 'prepared', 'R_100')
    assert report['rejected_records'] == 1
    assert report['quarantine'][0]['reason'] == 'registro inválido'
    assert dataset_builder.count_labelable_contracts(snapshot, 'R_100') == direct
    assert source.read_bytes() == original
    assert read_records(snapshot)[1] == []
    with pytest.raises(FileExistsError):
        prepare_snapshot(source, tmp_path / 'prepared', 'R_100')


def test_unattributable_epoch_fails_closed(tmp_path):
    source = tmp_path / 'ticks.csv'
    write_source(source)
    with source.open('a', newline='') as handle:
        csv.writer(handle).writerow(['broken', '', 'R_100', 100, 'tick', 0, '', '', ''])
    with pytest.raises(TrainingDataError, match='sem época segura'):
        load_records(source, 'R_100')


def test_conflicting_tick_excludes_window_even_with_other_good_ticks(tmp_path):
    source = tmp_path / 'ticks.csv'
    write_source(source)
    baseline = dataset_builder.count_labelable_contracts(source, 'R_100')
    with source.open('a', newline='') as handle:
        csv.writer(handle).writerow([BASE + 50 * TF + 100, '', 'R_100', 999,
                                     'tick', 0, '', '', ''])
    assert dataset_builder.count_labelable_contracts(source, 'R_100') < baseline


def test_confirmed_ohlc_survives_old_tick_gap_but_label_interval_does_not(tmp_path):
    source = tmp_path / 'ticks.csv'
    write_source(source, gap=(49 * TF + 100, 49 * TF + 300))
    count = dataset_builder.count_labelable_contracts(source, 'R_100')
    assert count > 0
    out = tmp_path / 'dataset.csv'
    assert dataset_builder.build_dataset(source, out, symbol='R_100') == count
    with out.open() as handle:
        rows = list(csv.DictReader(handle))
    assert int(rows[0]['signal_epoch']) == BASE + 50 * TF
    write_source(source, gap=(50 * TF + 100, 50 * TF + 300))
    out2 = tmp_path / 'dataset2.csv'
    dataset_builder.build_dataset(source, out2, symbol='R_100')
    assert all(int(row['signal_epoch']) != BASE + 50 * TF
               for row in csv.DictReader(out2.open()))


def test_incomplete_derived_ohlc_cannot_replace_confirmed_history(tmp_path):
    source = tmp_path / 'ticks.csv'
    write_source(source, history=False, gap=(49 * TF + 100, 49 * TF + 300))
    assert dataset_builder.count_labelable_contracts(source, 'R_100') == 0


def test_training_window_changes_do_not_change_collection_budget_or_old_model_gate():
    current = specification('R_100')
    assert minimum_history_candles() == 672
    assert config.ASSET_HISTORY_CANDLES == 336
    assert current['training_history_days'] == 28
    previous = dict(current, training_history_days=7)
    assert not compatible(previous, 'R_100')


def test_separate_history_warms_features_and_source_is_untouched(tmp_path):
    source = tmp_path / 'ticks.csv'
    write_source(source, history=False)
    history = tmp_path / 'ohlc.csv'
    with history.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        for i in range(FEATURE_WINDOW):
            writer.writerow([BASE + i * TF, '', 'R_100', 100, 'candle', TF, 100, 101, 99])
    before = source.read_bytes(), history.read_bytes()
    snapshot, _report = prepare_snapshot(source, tmp_path / 'prepared', 'R_100', history_path=history)
    assert dataset_builder.count_labelable_contracts(snapshot, 'R_100') > 0
    assert (source.read_bytes(), history.read_bytes()) == before


def test_snapshot_tampering_is_rejected(tmp_path):
    source = tmp_path / 'ticks.csv'
    write_source(source)
    snapshot, _ = prepare_snapshot(source, tmp_path / 'prepared', 'R_100')
    with snapshot.open('a') as handle:
        handle.write('\n')
    with pytest.raises(TrainingDataError, match='difere'):
        load_records(snapshot, 'R_100')


def test_future_ohlc_cannot_move_cutoff_or_supply_features(tmp_path):
    source = tmp_path / 'ticks.csv'
    write_source(source)
    baseline = dataset_builder.count_labelable_contracts(source, 'R_100')
    with source.open('a', newline='') as handle:
        csv.writer(handle).writerow([BASE + 100 * 86400, '', 'R_100', 100,
                                     'candle', TF, 100, 101, 99])
    assert dataset_builder.count_labelable_contracts(source, 'R_100') == baseline


def test_28_days_can_reach_structural_floor_with_weekly_closures(tmp_path, monkeypatch):
    source = tmp_path / 'market.csv'
    with source.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        for day in range(28):
            if day % 7 >= 5:
                continue
            start = BASE + day * 86400
            for slot in range(96):
                writer.writerow([start + slot * TF, '', 'R_100', 100,
                                 'candle', TF, 100, 101, 99])
            for offset in range(0, 86400, 10):
                writer.writerow([start + offset, '', 'R_100', 100,
                                 'tick', 0, '', '', ''])
    assert dataset_builder.count_labelable_contracts(source, 'R_100') >= 579
    monkeypatch.setattr(dataset_builder, 'TRAINING_HISTORY_DAYS', 7)
    assert dataset_builder.count_labelable_contracts(source, 'R_100') < 579


def test_readiness_cache_is_per_symbol_and_reports_real_progress(tmp_path, monkeypatch):
    import pipeline
    from asset_history import AssetHistoryStore
    source = tmp_path / 'ticks.csv'
    write_source(source)
    monkeypatch.setattr(pipeline, 'TICKS_CSV', str(source))
    monkeypatch.setattr(pipeline, '_asset_history_store', AssetHistoryStore(tmp_path / 'history', TF))
    monkeypatch.setattr(pipeline, '_runtime_symbol', lambda: 'R_100')
    monkeypatch.setattr(pipeline, '_training_readiness_cache', {})
    count = pipeline._count_labelable_contracts_for_training()
    assert count > 0
    status = json.loads(pipeline._BOT_STATUS_JSON.read_text())
    assert status['training']['labelable_contracts'] == count
    assert status['training']['required_contracts'] == 579
    original = dataset_builder.training_diagnostics
    monkeypatch.setattr(dataset_builder, 'training_diagnostics', lambda *args: pytest.fail('releitura no cache'))
    assert pipeline._count_labelable_contracts_for_training() == count
    monkeypatch.setattr(dataset_builder, 'training_diagnostics', original)
    monkeypatch.setattr(pipeline, '_runtime_symbol', lambda: 'other')
    assert pipeline._count_labelable_contracts_for_training() == 0
    status = json.loads(pipeline._BOT_STATUS_JSON.read_text())
    assert status['training']['stage'] == 'invalid_data'
