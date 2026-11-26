#!/usr/bin/env bash
# Executa pytest do StanDeriv numa cópia temporária, sem .env, dados, modelos
# ou estado operacional, com rede bloqueada por namespace (unshare -rn).
# Uso: run_isolated.sh [--keep] [--allow-network] [argumentos do pytest...]
#   Sem argumentos do pytest roda a suíte completa (tests -q).
set -euo pipefail

SRC="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
KEEP=0
NET=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --keep) KEEP=1; shift ;;
    --allow-network) NET=1; shift ;;
    *) break ;;
  esac
done
ARGS=("$@")
[[ ${#ARGS[@]} -eq 0 ]] && ARGS=(tests -q)

PY="${STANDERIV_PYTHON:-}"
if [[ -z "$PY" ]]; then
  # Caminho do venv sem resolver o symlink, para manter seus site-packages.
  for c in "$SRC/.venv/bin/python" "$SRC/../StanDeriv/.venv/bin/python"; do
    [[ -x "$c" ]] && { PY="$(cd "$(dirname "$c")" && pwd)/python"; break; }
  done
fi
[[ -z "$PY" ]] && { echo "ERRO: .venv não encontrado; defina STANDERIV_PYTHON" >&2; exit 2; }

DEST="$(mktemp -d /tmp/standeriv-isolado-XXXXXXXX)"
cleanup() { [[ $KEEP -eq 1 ]] && echo "Cópia preservada: $DEST" || rm -rf "$DEST"; }
trap cleanup EXIT

# Copia apenas código e testes. Dados, modelos, estado, logs e credenciais
# ficam de fora; arquivos dentro de tests/ (fixtures) são preservados.
rsync -a \
  --exclude='.git' --exclude='.venv' --exclude='__pycache__' --exclude='.pytest_cache' \
  --exclude='.env' --exclude='.env.*' \
  --exclude='/*.csv' --exclude='/*.csv.*' --exclude='/*.tmp' \
  --exclude='/*.pkl' --exclude='/*.pkl.*' --exclude='*.joblib' \
  --exclude='/*.json' --exclude='/*.log' --exclude='/*.pid' --exclude='/*.lock' \
  --exclude='/shadow_state.json*' \
  --exclude='/Histórico Ativos' --exclude='/.training_inputs' --exclude='/.training_candidates' \
  --exclude='/training_reports' --exclude='/recovered_data' --exclude='/backtest_reports' \
  --exclude='/docs' --exclude='/.claude' \
  "$SRC/" "$DEST/"
# pyrightconfig e similares não são necessários aos testes; nada de estado entra.

LEFT=$(cd "$DEST" && find . -maxdepth 1 \( -name '*.csv' -o -name '*.pkl*' -o -name '.env*' -o -name '*state*.json' \) | wc -l)
[[ "$LEFT" -ne 0 ]] && { echo "ERRO: artefato operacional copiado" >&2; exit 3; }

ENV=(env -i HOME="$DEST/.home" PATH=/usr/bin:/bin LANG=C.UTF-8
     DERIV_TOKEN=offline_test_token DERIV_APP_ID=1089
     PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1)
mkdir -p "$DEST/.home"

echo "Cópia: $DEST | Python: $("$PY" --version 2>&1) | rede: $([[ $NET -eq 1 ]] && echo LIBERADA || echo bloqueada)"
cd "$DEST"
START=$(date +%s)
set +e
if [[ $NET -eq 1 ]]; then
  "${ENV[@]}" "$PY" -m pytest -p no:cacheprovider "${ARGS[@]}"
else
  unshare -rn "${ENV[@]}" "$PY" -m pytest -p no:cacheprovider "${ARGS[@]}"
fi
RC=$?
set -e
echo "pytest exit=$RC em $(( $(date +%s) - START ))s"
exit $RC
