# Índice dos registros históricos

Registros datados de análises, correções e implementações. Eles preservam o
estado observado na data e não redefinem o comportamento atual: para as regras
vigentes, consulte o [README](../../README.md), a [arquitetura](../../ARCHITECTURE.md),
o [guia técnico](../GUIA_TECNICO.md) e a
[homologação de Score e coleta](../HOMOLOGACAO_SCORE_COLETA_2026-10-15.md).

## Setembro de 2026 — contratos v2 e treinamento

| Data | Registro | Assunto |
| --- | --- | --- |
| 08/09 | [ANALISE_LOSSES_2026-09-08.md](ANALISE_LOSSES_2026-09-08.md) | Auditoria das perdas e defeitos que motivaram o contrato v2 |
| 10/09 | [MANUTENCAO_TREINO_2026-09-10.md](MANUTENCAO_TREINO_2026-09-10.md) | Treino sem aprovação, orçamento histórico e sincronização do ativo |
| 10/09 | [MELHORIAS_AVALIACAO_2026-09-10.md](MELHORIAS_AVALIACAO_2026-09-10.md) | Avaliação walk-forward, cotações de payout e limites de aprovação |
| 11/09 | [pesquisa_treino_2026-09-11/RELATORIO.md](pesquisa_treino_2026-09-11/RELATORIO.md) | Pesquisa de métodos de treino, calibração e validação |
| 11/09 | [APLICACAO_TREINO_2026-09-11.md](APLICACAO_TREINO_2026-09-11.md) | Aplicação das recomendações da pesquisa |
| 11/09 | [AUDITORIA_CONFLITOS_2026-09-11.md](AUDITORIA_CONFLITOS_2026-09-11.md) | Conflitos entre backtest, avaliação e executor |
| 27/08 | [ARQUITETURA_ANTIGA.md](ARQUITETURA_ANTIGA.md) | Arquitetura anterior ao v2 (TFT/Stacking), apenas referência |

## Outubro de 2026 — Modelo de Treino isolado

| Data | Registro | Assunto |
| --- | --- | --- |
| 30/10 | [ANALISE_MODELO_TREINO_2026-10-30.md](ANALISE_MODELO_TREINO_2026-10-30.md) | Demora para abrir contratos e seleção de símbolos (antes das correções) |
| 30/10 | [REAVALIACAO_MODELO_TREINO_POS_ALTERACOES_2026-10-30.md](REAVALIACAO_MODELO_TREINO_POS_ALTERACOES_2026-10-30.md) | Reavaliação após progresso por rótulo, retomada do shadow e seleção |

## Novembro de 2026 — IA de 5 minutos, shadow e canary

| Data | Registro | Assunto |
| --- | --- | --- |
| 03/11 | [IMPLEMENTACAO_IA_5M_15M_2026-11-03.md](IMPLEMENTACAO_IA_5M_15M_2026-11-03.md) | Candles de 5 minutos para contratos de 15 e complemento incremental |
| 03/11 | [STATUS_EXECUCAO_IA_2026-11-03.md](STATUS_EXECUCAO_IA_2026-11-03.md) | Sessão com candidato reprovado e falha no complemento após reprovação |
| 03/11 | [REAVALIACAO_EXECUCAO_IA_POS_CORRECAO_2026-11-03.md](REAVALIACAO_EXECUCAO_IA_POS_CORRECAO_2026-11-03.md) | Primeiro modelo aprovado offline e início do shadow |
| 03/11 | [STATUS_SEM_CONTRATOS_2026-11-03_1346.md](STATUS_SEM_CONTRATOS_2026-11-03_1346.md) | Shadow pendente, payout abaixo do piso e retreinos não promovidos |
| 03/11 | [AVALIACAO_VALIDACAO_CURTA_2026-11-03.md](AVALIACAO_VALIDACAO_CURTA_2026-11-03.md) | Estudo da validação forward curta (8 resultados) |
| 03/11 | [IMPLEMENTACAO_CANARY_8_RESULTADOS_2026-11-03.md](IMPLEMENTACAO_CANARY_8_RESULTADOS_2026-11-03.md) | Implementação do canary 8/7 e do cache de inferência |
| 03/11 | [ANALISE_CANARY_SEM_ENTRADAS_2026-11-03_1540.md](ANALISE_CANARY_SEM_ENTRADAS_2026-11-03_1540.md) | Canary sem chance de aprovação aguardando oito resultados |
| 03/11 | [CORRECAO_REPROVACAO_ANTECIPADA_CANARY_2026-11-03.md](CORRECAO_REPROVACAO_ANTECIPADA_CANARY_2026-11-03.md) | Reprovação antecipada quando a meta fica impossível |

Novos registros devem receber o aviso "Consulta histórica" no topo e uma linha
neste índice.
