"""
tests/conftest.py — configuração global de testes.

Este arquivo é carregado pelo pytest antes de qualquer módulo de teste,
garantindo que as variáveis de ambiente necessárias estejam definidas
antes que config.py tente lê-las.
"""

import os
import sys
from pathlib import Path

# Garante que o root do projeto está no sys.path (evita sys.path.insert em cada teste)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Define variáveis de ambiente de teste antes que config.py seja importado
os.environ.setdefault("DERIV_TOKEN", "test_token_placeholder")
os.environ.setdefault("DERIV_APP_ID", "1089")
os.environ.setdefault("DASHBOARD_TOKEN", "")
