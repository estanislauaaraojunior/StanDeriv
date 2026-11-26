"""Cartão operacional do painel exibe todos os estágios publicados pelo pipeline."""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1] / 'dashboard' / 'app.js'
NODE = shutil.which('node')


def _function(source, name):
    start = source.index(f'function {name}(')
    # O corpo começa depois da lista de parâmetros (que pode ter `= {}`).
    depth, index = 0, source.index(') {', start) + 2
    for index in range(index, len(source)):
        depth += {'{': 1, '}': -1}.get(source[index], 0)
        if depth == 0:
            return source[start:index + 1]
    raise AssertionError(name)


def card(training):
    source = APP.read_text(encoding='utf-8')
    script = '\n'.join(_function(source, n) for n in ('_withRetryProgress', '_botOperationalCard'))
    bot = dict(symbol='R_100', use_trained_model=True, status='waiting_model',
               training=dict(training, symbol='R_100'))
    script += f'\nconsole.log(JSON.stringify(_botOperationalCard({json.dumps(bot)}, {{ready: false}})));'
    out = subprocess.run([NODE, '-e', script], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


pytestmark = pytest.mark.skipif(NODE is None, reason='node indisponível')


@pytest.mark.parametrize('stage,label', [
    ('awaiting_new_labels', 'AGUARDANDO NOVOS RÓTULOS'),
    ('evaluation_capacity', 'CAPACIDADE TEMPORAL INSUFICIENTE'),
    ('approved_offline', 'APROVADO OFFLINE'),
])
def test_new_pipeline_stages_have_their_own_card(stage, label):
    assert card(dict(stage=stage, reason='x', evaluation_capacity={'validation': 12, 'test': 40}))[0] == label


def test_rejected_model_shows_retry_progress_from_new_samples():
    detail = card(dict(stage='model_rejected', reason='acerto abaixo de 45%', next_training_at=None,
                       new_samples=40, retry_batch=120, remaining_samples=80))[2]
    assert 'novos rótulos: 40/120; faltam 80' in detail


def test_progress_is_not_repeated_when_pipeline_reason_already_has_it():
    reason = 'acerto abaixo de 45% | novos rótulos: 40/120; faltam 80'
    detail = card(dict(stage='model_rejected', reason=reason, new_samples=40,
                       retry_batch=120, remaining_samples=80))[2]
    assert detail.count('novos rótulos') == 1


def test_capacity_card_shows_validation_and_test_counts():
    detail = card(dict(stage='evaluation_capacity', reason='validação curta',
                       evaluation_capacity={'validation': 12, 'test': 40}))[2]
    assert 'validação 12, teste 40' in detail


def test_app_js_syntax_is_valid():
    subprocess.run([NODE, '--check', str(APP)], check=True)
