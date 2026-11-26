"""Snapshots de treino não acumulam cópias de preços no disco."""
import json
from pathlib import Path

import pytest

import pipeline


def _fake_snapshot(*, independent=False):
    def prepare(source, destination, symbol, *, history_path, enricher):
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        target = destination
        if independent:
            # Simula a falha na origem seguida da base independente.
            target = destination.with_name(destination.name + '-independent')
            target.mkdir(parents=True, exist_ok=True)
        snapshot = target / 'prices.csv'
        snapshot.write_text('epoch,price\n' + '1,1\n' * 1000)
        (target / 'prices.csv.preparation.json').write_text(json.dumps({'sha256': 'x'}))
        return snapshot, {}
    return prepare


@pytest.mark.parametrize('independent', [False, True])
@pytest.mark.parametrize('rows', [0, 10_000])
def test_run_training_removes_price_copy_and_keeps_provenance(tmp_path, monkeypatch, independent, rows):
    monkeypatch.setattr(pipeline, '_runtime_symbol', lambda: 'R_100')
    monkeypatch.setattr(pipeline, '_write_pipeline_status', lambda **_kwargs: None)
    monkeypatch.setattr(pipeline, '_training_history_provider', lambda: None)
    monkeypatch.setattr(pipeline, '_prepare_model_snapshot', _fake_snapshot(independent=independent))
    monkeypatch.setattr(pipeline.dataset_builder, 'build_dataset', lambda *_a, **_k: rows)
    # Treino propriamente dito fora do escopo: falha controlada após o dataset.
    monkeypatch.setattr(pipeline.train_model, 'train',
                        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError('fim do teste')))
    dataset = tmp_path / 'dataset_v2.csv'
    dataset.write_text('signal_epoch\n')

    pipeline._run_training(dataset_path=str(dataset), model_path=str(tmp_path / 'model.pkl.new'))

    inputs = tmp_path / '.training_inputs'
    assert not list(inputs.rglob('prices.csv'))
    assert len(list(inputs.rglob('prices.csv.preparation.json'))) == 1
    # A pasta vazia deixada pela origem bloqueada também não permanece.
    assert all(any(folder.iterdir()) for folder in inputs.iterdir())


def test_cleanup_failure_does_not_abort(tmp_path, monkeypatch, capsys):
    folder = tmp_path / 'snap'
    folder.mkdir()
    (folder / 'prices.csv').write_text('x')
    monkeypatch.setattr(Path, 'unlink', lambda self: (_ for _ in ()).throw(PermissionError('ro')))
    pipeline._discard_snapshot_bulk(folder)
    assert 'Não foi possível limpar' in capsys.readouterr().out
