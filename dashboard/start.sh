#!/bin/bash
# Inicia o dashboard do bot Deriv.
# Uso: bash dashboard/start.sh
# Acesso: http://localhost:5055

set -e
# Vai para a raiz do projeto independente de onde o script é chamado
cd "$(cd "$(dirname "$0")/.." && pwd)"

# Detecta o Python correto (venv do usuário, venv local ou sistema)
PYTHON=""
for PY in "$HOME/.venv/bin/python3" ".venv/bin/python3" "$(which python3 2>/dev/null)"; do
    if [ -x "$PY" ]; then
        PYTHON="$PY"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "[ERRO] Nenhum python3 encontrado."
    exit 1
fi

echo "Python  : $PYTHON"
echo "Diretório: $(pwd)"
echo "Iniciando Deriv Bot Dashboard em http://localhost:5055 ..."
exec "$PYTHON" dashboard/server.py
