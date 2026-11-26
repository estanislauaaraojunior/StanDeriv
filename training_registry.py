"""Reserva de bundles aprovados por símbolo, exclusiva do Modelo de Treino."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from model_contract import fingerprint, load_validated_bundle, usable


class TrainingModelRegistry:
    """Preserva candidatos aprovados sem publicar ou treinar modelos ao consultar."""

    def __init__(self, root):
        self.root = Path(root)

    def _folder(self, symbol):
        return self.root / hashlib.sha256(symbol.encode()).hexdigest()

    @staticmethod
    def _metadata(path, symbol):
        path = Path(path)
        try:
            metadata = json.loads(Path(str(path) + '.metadata.json').read_text())
            if (not usable(metadata, symbol)
                    or metadata.get('artifact_sha256') != fingerprint(path)):
                return None
            return metadata
        except (FileNotFoundError, ValueError, TypeError, AttributeError):
            return None

    @staticmethod
    def _copy_pair(source, destination):
        """Publica arquivos completos; intervalo entre replaces falha fechado pelo hash."""
        source, destination = Path(source), Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.bundle-', dir=destination.parent) as temp:
            model = Path(temp) / 'model.pkl'
            sidecar = Path(str(model) + '.metadata.json')
            shutil.copyfile(source, model)
            shutil.copyfile(str(source) + '.metadata.json', sidecar)
            metadata = json.loads(sidecar.read_text())
            if fingerprint(model) != metadata.get('artifact_sha256'):
                raise ValueError('bundle mudou durante a cópia')
            os.replace(model, destination)
            os.replace(sidecar, str(destination) + '.metadata.json')

    def save(self, model_path, symbol):
        """Guarda somente bundles com sidecar aprovado, compatível e íntegro."""
        metadata = self._metadata(model_path, symbol)
        if metadata is None:
            return None
        destination = self._folder(symbol) / (metadata['artifact_sha256'] + '.pkl')
        if self._metadata(destination, symbol) is None:
            self._copy_pair(model_path, destination)
        return destination

    def find(self, symbol, *, accept=None):
        """Localiza reserva compatível sem desserializar pickle na seleção de dados."""
        paths = self._folder(symbol).glob('*.pkl')
        for path in sorted(paths, key=lambda p: p.stat().st_mtime_ns, reverse=True):
            metadata = self._metadata(path, symbol)
            if metadata is not None and (accept is None or accept(metadata)):
                return path
        return None

    def activate(self, symbol, destination, *, accept=None):
        """Valida o payload e copia a reserva antes de iniciar o executor do símbolo."""
        source = self.find(symbol, accept=accept)
        if source is None:
            return False
        try:
            load_validated_bundle(source, symbol)
        except (FileNotFoundError, ValueError, TypeError):
            return False
        self._copy_pair(source, destination)
        return True
