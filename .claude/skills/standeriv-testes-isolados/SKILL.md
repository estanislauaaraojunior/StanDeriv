---
name: standeriv-testes-isolados
description: Executa a suíte pytest do StanDeriv (ou testes direcionados) numa cópia temporária sem .env, credenciais, CSVs, modelos .pkl, estados shadow/risco ou logs, com rede bloqueada. Use sempre que for validar qualquer alteração de código no StanDeriv, rodar "os testes", "a suíte", pytest, verificar regressões, ou antes de concluir/commitar uma tarefa — mesmo que o usuário só diga "testa isso" ou "roda os testes". Não rode pytest direto no checkout.
---

# Testes isolados do StanDeriv

O AGENTS.md e os relatórios do projeto exigem que testes rodem em cópia
temporária: alguns testes treinam modelos sintéticos, escrevem arquivos e
importam módulos que leem `.env` ou caminhos operacionais. Rodar no checkout
arrisca sobrescrever `state.json`, `shadow_state.json`, modelos ou CSVs em uso,
e uma conexão acidental à Deriv seria inaceitável. Este script padroniza o
procedimento usado nas entregas anteriores.

## Como rodar

```bash
S=.claude/skills/standeriv-testes-isolados/scripts/run_isolated.sh
bash $S                                   # suíte completa (tests -q)
bash $S tests/test_strategy.py -q         # direcionado
bash $S tests -q -k canary                # filtro
bash $S --keep tests/test_x.py -q         # preserva a cópia para inspeção
```

O script:
- copia só código e `tests/` para `/tmp/standeriv-isolado-*` (exclui `.env`,
  CSVs/JSON/PKL/logs da raiz, `Histórico Ativos/`, `.training_*`, relatórios,
  `.git`, `docs/`) e aborta se algum artefato operacional escapar;
- usa o `.venv` do worktree ou do checkout principal (`STANDERIV_PYTHON` sobrescreve);
- roda com `env -i`, token fictício e limites de threads (evita timeouts nos
  ajustes de HistGradientBoosting);
- bloqueia a rede com `unshare -rn`; `--allow-network` só com motivo explícito;
- apaga a cópia ao final, salvo `--keep`.

## Escolha dos testes

Comece pelos testes da área alterada (mapa em `docs/GUIA_TECNICO.md`). Rode a
suíte completa quando a mudança for transversal ou tocar `executor.py`,
`pipeline.py`, `model_contract.py`, `risk_manager.py` ou autenticação. A suíte
completa leva ~1,5 min. Para mudanças só em Markdown, não rode a suíte; use
`git diff --check`.

## Como relatar

Informe o comando, o número de testes aprovados/falhos/desmarcados, o tempo e a
versão do Python impressos. Diferencie "executado e aprovado", "executado e
falhou" e "não executado". Em falha, mostre o trecho relevante do traceback e
verifique se a falha também ocorre sem sua alteração antes de atribuí-la a ela.
Nunca afirme que testes passaram sem esta saída.
