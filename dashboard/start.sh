#!/bin/bash
# Inicia o dashboard do bot Deriv.
# Uso: bash dashboard/start.sh
# Acesso: http://localhost:5055

set -e
cd "$(dirname "$0")/.."   # vai para /home/stanis/Repositorios/Binary

echo "Iniciando Deriv Bot Dashboard em http://localhost:5055 ..."
exec .venv/bin/python3 dashboard/server.py
