> **Consulta histórica:** este registro mantém os achados e resultados da data
> indicada, sem redefinir o comportamento atual. Consulte o [README](../../README.md)
> e a [homologação de Score e coleta](../HOMOLOGACAO_SCORE_COLETA_2026-10-15.md).

# Ausência de contratos — nova leitura em 03/11/2026

Auditoria entre 13:46 e 13:50, America/Sao_Paulo, da mesma sessão iniciada às
12:49:39. Somente leitura operacional; este relatório é a única adição.
Complementa a [leitura anterior](REAVALIACAO_EXECUCAO_IA_POS_CORRECAO_2026-11-03.md).

## Motivo confirmado

Às 13:46, o executor recebia ticks atuais, tinha **50/50 candles privados de
5 minutos** e gerava previsão SELL com confiança de aproximadamente **64,3%**. Contudo,
`executor._evaluate_entry` transforma esse sinal em abstenção enquanto o
modelo não tiver shadow aprovado. O motivo publicado é
`shadow_validation_pending`. Assim, dados suficientes e modelo aprovado
offline não autorizam, sozinhos, a proposta de entrada ou a compra.

O shadow avançou de um para **dois resultados**, sem apagar a identidade
do modelo ativo. Ambos foram incorretos:

| Entrada SP | Vencimento SP | Direção | Payout líquido observado | Resultado |
| --- | --- | --- | --- | --- |
| 13:05 | 13:20 | SELL | 82% | Incorreto |
| 13:25 | 13:40 | SELL | 74% | Incorreto |

Na leitura das 13:47:51 não havia previsão pendente nem contrato ativo.
O mínimo continua em **30 resultados**, com pelo menos 55% de acerto.
Dois resultados não permitem concluir a qualidade futura; também não
constituem aprovação. Faltam 28 resultados: **pelo menos sete horas de
contratos válidos de 15 minutos**, além dos intervalos de espera. O estudo
de validação curta ainda não foi implementado.

## Bloqueio adicional: payout da direção prevista

No fechamento das 13:45, o resumo do shadow passou a
`payout_below_floor`. A cotação PUT anterior e válida, observada às
13:43:04, tinha payout líquido de **69%**, abaixo do piso de **70%**.
O CALL da mesma rodada tinha 92%, mas não corresponde à direção SELL
prevista pelo modelo. Não se deve inverter o sinal para aproveitar essa cota.

Às 13:48:05 foi registrada nova cotação PUT de **73%**. Nesse instante o
status ainda conservava o motivo anterior. O executor tenta capturar uma
nova previsão shadow apenas no evento de fechamento elegível da janela;
não tenta novamente a cada tick quando o payout melhora no meio do candle.
Portanto, um resumo `payout_below_floor` não informa necessariamente o payout
mais recente: ele descreve a última tentativa de captura. A consulta de
`QuoteHistory.lookup` é causal, sem usar cotação posterior ao sinal.

### Confirmação no fechamento das 13:50

A auditoria acompanhou o próximo fechamento. O arquivo persistido confirmou
**uma terceira previsão shadow iniciada às 13:50:00**, SELL, com confiança
de aproximadamente **56,0%** e payout líquido de **73%**, vencendo às
**14:05:00**. O status passou a `shadow.reason=pending`, mantendo os dois
resultados anteriores e a mesma identidade do modelo. Na confirmação das
13:50:12, a janela continuava em 50/50 candles.

Isso confirma que o fluxo retomou a captura quando o payout voltou a atender
o piso, sem reinício ou alteração. O terceiro resultado ainda não era
conhecido. A captura é simulada: `shadow_validation_pending` continua
impedindo a compra enquanto não houver aprovação forward. Os 28 resultados
restantes incluem esse contrato simulado em curso.

## Conexão do executor e coleta são diferentes

Foram encontrados **cinco timeouts de ping/pong do executor** na sessão:
13:19:45, 13:26:05, 13:33:27, 13:37:46 e 13:38:31, com reconexão.
O diagnóstico de feed que mostra zero considera somente erros do coletor.

No trecho de CSV lido sob lock compartilhado às 13:47:45, a coleta possuía
1.801 ticks únicos nos últimos 30 minutos, com espaçamento máximo de um
segundo. Havia ticks nos instantes exatos dos fechamentos examinados de
13:15 a 13:45. Isso comprova continuidade da coleta nesse trecho; não prova
que o WebSocket separado do executor recebeu todos esses eventos a tempo.

O timestamp do executor estava atual e a janela estava completa nas leituras
desta auditoria. Não foi medida a duração dos callbacks nem demonstrada a
causa dos timeouts. Eles podem reduzir as oportunidades de captura, mas
não é possível atribuir cada fechamento sem nova previsão a esses erros.

## Retreino está funcionando, mas não libera compras

| Conclusão SP | Receita | Contratos históricos | Elegível offline | Promoção |
| --- | --- | --- | --- | --- |
| 13:03:53 | XGBoost regularizado | 36/85, 42,4% | Não | Reprovado no teste. |
| 13:25:38 | BalancedLogisticRegression | 35/60, 58,3% | Sim | Comparação temporal com o ativo indisponível. |
| 13:43:40 | HistGradientBoosting regularizado | 37/77, 48,1% | Sim | Comparação temporal com o ativo indisponível. |

O ativo continua sendo o RandomForest aprovado às 12:54:42; SHA-256 do
arquivo, metadados e identidade do shadow coincidem. Logo, houve treino e
recuperação de dados novos, mas os últimos treinos não alteraram a inferência.

`train_model.py` só calcula `champion_metrics` se a última época vista pelo
ativo for anterior ao início das features do teste do candidato. O ativo
tem `last_seen_epoch=1793720400`; os inícios das features dos testes dos dois
candidatos aprovados são `1793125200` e `1793158200`. A condição falha porque
essas janelas históricas se sobrepõem ao período já visto pelo ativo.
`pipeline._should_promote_model` recusa a promoção sem essa comparação.
Não se deve remover essa proteção contra contaminação temporal.

Mesmo uma comparação aprovada encontra outra proteção: não trocar o bundle
enquanto seu shadow ainda não foi aprovado. Isso preserva o progresso
solicitado pelo usuário. Na implementação atual, a limpeza do retreino
remove o arquivo temporário do estimador não promovido; nas duas pastas
aprovadas restaram métricas, metadados e relatórios, sem `model.pkl`.
Esses resultados não são bundles prontos para ativação posterior.

O candidato de 58,3% teve expectativa positiva apenas no cenário histórico
de payout fixo de 85%. Com payouts observados, teve 4/10 acertos e expectativa
negativa; no cenário de 70%, 3/7. Seu número principal não demonstra que
trocá-lo agora resolveria as entradas ou a rentabilidade.

## Limitações de observabilidade confirmadas no código

- Liquidações intermediárias do shadow são persistidas, mas o resumo do status
  só é publicado de imediato para aprovação/reprovação final. Outra captura
  acaba atualizando o número, explicando o atraso observado antes.
- `training.stage=approved_offline` nesta leitura se refere ao último treino,
  não comprova ativação do novo candidato nem aprovação forward do ativo.
- `runtime_status.update_status` protege os campos de candles do pipeline
  somente quando `candle_source == 'executor'`. A IA isolada publica
  `training_private`; portanto, o heartbeat do pipeline pode sobrescrever
  essa contagem com o aquecimento técnico. Esse caminho foi identificado
  no código, sem reproduzir a corrida no processo ativo.
- As 120 decisões recentes eram todas `shadow_validation_pending`.
  São uma janela de eventos, não 120 contratos, candles ou falhas da API.

## Próximos ajustes indicados

1. Separar claramente modelo ativo, último candidato, progresso shadow e
   motivo da última captura; proteger também os candles `training_private`
   contra sobrescrita do heartbeat e distinguir timeouts de cada conexão.
2. Medir os callbacks e as perdas da janela privada do executor para explicar
   os cinco timeouts e os fechamentos perdidos, sem alterar a coleta homologada.
3. Planejar comparação em período futuro comum e preservação de candidatos
   aprovados em artefatos isolados. Recuperar mais passado pode renovar
   rótulos, mas não cria automaticamente um teste futuro para o modelo ativo.
4. Definir uma política forward curta como mudança explícita e versionada,
   validada offline. O código atual não oferece uma liberação rápida: mesmo
   com dados completos, seu mínimo de 30 contratos permanece obrigatório.

Não há evidência nesta leitura de que uma nova troca de ativo, mais dados
históricos ou outro retreino isoladamente eliminem esse bloqueio. A seleção
atual cede candidatos no bootstrap sem modelo compatível; ela preserva um
modelo já em shadow, conforme solicitado.

## Verificações e escopo

Foram lidos status, metadados, relatórios, logs filtrados, cotações e ticks
sob lock compartilhado não bloqueante. O modelo foi conferido por hash,
sem desserialização. O estado persistido não tinha contrato ativo.
O registro operacional examinado tinha suas últimas linhas em 19/10;
nenhuma compra desta sessão foi identificada nos arquivos consultados.

Não houve conexão iniciada pela auditoria, consulta à conta, alteração de
código, parâmetros, modelos, dados, Score ou coleta; tampouco compra,
reinício ou retreino. Verificação documental restrita ao novo relatório;
sem necessidade de executar a suíte ou o bot para uma análise de status.
