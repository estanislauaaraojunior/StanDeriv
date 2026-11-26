"""Reserva de modelos validada com artefatos fictícios, sem carregar pickle."""
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import Mock

import pytest

import training_registry
from training_registry import TrainingModelRegistry


def bundle(folder, symbol='FIRST', content=b'offline-artifact', approved=True):
    folder.mkdir(parents=True, exist_ok=True)
    model = folder / 'model.pkl'
    model.write_bytes(content)
    metadata = dict(symbol=symbol, approved=approved,
                    artifact_sha256=hashlib.sha256(content).hexdigest())
    Path(str(model) + '.metadata.json').write_text(json.dumps(metadata))
    return model


@pytest.fixture(autouse=True)
def validated_sidecars(monkeypatch):
    monkeypatch.setattr(training_registry, 'usable',
                        lambda metadata, symbol: metadata.get('symbol') == symbol and metadata.get('approved') is True)


def test_reserves_models_separately_and_preserves_originals(tmp_path, monkeypatch):
    registry = TrainingModelRegistry(tmp_path / 'registry')
    first = bundle(tmp_path / 'first')
    second = bundle(tmp_path / 'second', 'SECOND', b'second-artifact')
    loader = Mock(side_effect=AssertionError('consulta não pode abrir pickle'))
    monkeypatch.setattr(training_registry, 'load_validated_bundle', loader)
    reserved = registry.save(first, 'FIRST')
    registry.save(second, 'SECOND')
    assert reserved != registry.find('SECOND')
    assert registry.find('FIRST') == reserved
    assert first.read_bytes() == reserved.read_bytes()
    assert second.exists()
    assert registry.find('UNKNOWN') is None
    loader.assert_not_called()


def test_rejected_incompatible_and_corrupt_models_are_not_reserved(tmp_path):
    registry = TrainingModelRegistry(tmp_path / 'registry')
    source = bundle(tmp_path / 'source', approved=False)
    assert registry.save(source, 'FIRST') is None
    source = bundle(tmp_path / 'source')
    assert registry.save(source, 'OTHER') is None
    source.write_bytes(b'corrupted')
    assert registry.save(source, 'FIRST') is None
    assert not registry.root.exists()


def test_corrupt_recent_reserve_falls_back_to_older_valid_model(tmp_path):
    registry = TrainingModelRegistry(tmp_path / 'registry')
    older = registry.save(bundle(tmp_path / 'first'), 'FIRST')
    newer = registry.save(bundle(tmp_path / 'second', content=b'new-artifact'), 'FIRST')
    os.utime(older, ns=(1000, 1000))
    assert registry.find('FIRST') == newer
    newer.write_bytes(b'corrupt')
    assert registry.find('FIRST') == older


def test_activation_validates_payload_before_replacing_active_files(tmp_path, monkeypatch):
    registry = TrainingModelRegistry(tmp_path / 'registry')
    source = bundle(tmp_path / 'source')
    reserved = registry.save(source, 'FIRST')
    destination = bundle(tmp_path / 'active', 'SECOND', b'active-artifact')
    before = destination.read_bytes()
    loader = Mock(side_effect=ValueError('payload difere do sidecar'))
    monkeypatch.setattr(training_registry, 'load_validated_bundle', loader)
    assert not registry.activate('FIRST', destination)
    assert destination.read_bytes() == before
    loader.side_effect = None
    loader.return_value = ({}, {})
    assert registry.activate('FIRST', destination)
    loader.assert_called_with(reserved, 'FIRST')
    assert destination.read_bytes() == source.read_bytes()
    assert Path(str(destination) + '.metadata.json').read_bytes() == Path(str(source) + '.metadata.json').read_bytes()


def test_rejected_shadow_identity_is_skipped_without_deleting_bundle(tmp_path, monkeypatch):
    registry = TrainingModelRegistry(tmp_path / 'registry')
    source = bundle(tmp_path / 'source')
    reserved = registry.save(source, 'FIRST')
    assert registry.find('FIRST', accept=lambda metadata: False) is None
    loader = Mock(side_effect=AssertionError('não deve ativar identidade rejeitada'))
    monkeypatch.setattr(training_registry, 'load_validated_bundle', loader)
    assert not registry.activate('FIRST', tmp_path / 'active.pkl', accept=lambda metadata: False)
    assert reserved.exists()


def test_unexpected_io_failure_is_not_reported_as_success(tmp_path, monkeypatch):
    registry = TrainingModelRegistry(tmp_path / 'registry')
    source = bundle(tmp_path / 'source')
    monkeypatch.setattr(training_registry.shutil, 'copyfile', Mock(side_effect=PermissionError('offline')))
    with pytest.raises(PermissionError):
        registry.save(source, 'FIRST')
