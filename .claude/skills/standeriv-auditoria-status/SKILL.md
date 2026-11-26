---
name: standeriv-auditoria-status
description: Auditoria somente leitura do estado operacional do StanDeriv — status da sessão, ticks, contrato ativo, modelo ativo (hash sem abrir pickle), shadow original de 30 resultados, canary 8/7, progresso da IA e log sanitizado. Use sempre que o usuário perguntar "por que o bot não abre contratos", "como está o bot", "o que está acontecendo", "o modelo foi aprovado?", "como está o shadow/canary", pedir análise de status/logs ou um novo relatório em docs/historico, mesmo sem citar arquivos.
---

# Auditoria de status do StanDeriv

As auditorias anteriores (docs/historico/STATUS_*, ANALISE_CANARY_*,
REAVALIACAO_*) seguiram sempre o mesmo protocolo: ler arquivos persistidos sem
alterar nada, conferir o modelo por hash, nunca desserializar pickle, nunca
expor credenciais ou identificadores de conta, e separar observação de hipótese.
Este script faz a coleta mecânica; a análise continua sendo sua.

## 1. Coletar

```bash
python3 .claude/skills/standeriv-auditoria-status/scripts/audit_status.py            # raiz deste checkout
python3 .claude/skills/standeriv-auditoria-status/scripts/audit_status.py --root ../StanDeriv --log-lines 60
python3 .claude/skills/standeriv-auditoria-status/scripts/audit_status.py --json
```

Confirme primeiro **qual diretório o processo ativo usa** (`pid`/`session_id`
do status, `started_at`, mtime dos arquivos): o worktree de refatoração e o
checkout principal têm estados diferentes. Se `running=true`, o `pid` pode não
ser visível no namespace da ferramenta; use heartbeat, idade do último tick e
log como evidência de atividade.

## 2. Aprofundar, se necessário

- Log: filtre por sessão e palavra-chave (`grep -n 'shadow\|canary\|reprov\|timeout' pipeline.log | tail`),
  sempre sanitizando antes de citar (sem URLs OTP, tokens, IDs de conta).
- Relatórios de treino: `training_reports/*-evaluation.json` e
  `model.pkl.metadata.json` com `jq` em campos selecionados (o metadata tem
  centenas de KB; nunca despeje inteiro).
- CSVs grandes (`ticks_v2.csv`, `operacoes_v2.csv`): leia só a cauda.
- **Não** abra `.env`, não carregue pickle/joblib, não importe `config`/`pipeline`,
  não consulte a API/conta, não reinicie nada, não edite estado.

## 3. Pontos que costumam explicar "sem contratos"

1. Processo parado ou heartbeat/tick antigos.
2. Modo efetivo: flags `use_*` reais, não o texto `execution_signal_mode`.
3. Aquecimento/recuperação (`warmup_remaining_candles`, `history_recovery`).
4. Sem modelo compatível / treino reprovado (`training.stage`, último motivo).
5. Shadow/canary pendente (`shadow_validation_pending`); confira
   `meta_ainda_possivel` — canary inviável sem `reprovado_em` indica processo
   rodando código anterior à reprovação antecipada.
6. Payout abaixo do piso na direção prevista; gates de retorno esperado.
7. Contrato ativo aguardando reconciliação.
Lembre que `ENFORCE_RISK_GATE=False`: pausa no painel não bloqueia ordens.

## 4. Relatar

Horários em America/Sao_Paulo. Estrutura: Conclusão → Evidência (com horários)
→ Causa vs hipóteses → Recomendações (sem forçar compras nem baixar critérios)
→ Escopo e verificações (o que foi lido, o que não foi feito). Se o usuário pedir
registro, salve em `docs/historico/<TEMA>_<AAAA-MM-DD>_<HHMM>.md` com o aviso
"Consulta histórica" no topo. Nunca invente métricas ou afirme rentabilidade.
