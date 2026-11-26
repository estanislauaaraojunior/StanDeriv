> **Consulta histórica:** este registro mantém os achados e resultados da data
> indicada, sem redefinir o comportamento atual. Consulte o [README](../../README.md)
> e a [homologação de Score e coleta](../HOMOLOGACAO_SCORE_COLETA_2026-10-15.md).

# Reavaliação da execução da IA após a correção — 03/11/2026

Leituras entre 13:20 e 13:24, America/Sao_Paulo. Este registro descreve uma
nova sessão e complementa o diagnóstico histórico em
[STATUS_EXECUCAO_IA_2026-11-03.md](STATUS_EXECUCAO_IA_2026-11-03.md).

## Resultado observado

A sessão iniciada às **12:49:39**, depois da correção de `pipeline.py` às
12:21:33, avançou além do bloqueio anterior. Há um **modelo aprovado offline
e ativo**, usando candles privados de **300 segundos** para contratos de
**15 minutos**, e uma previsão shadow já foi liquidada. O bloqueio atual
é a validação forward, não a falta de um primeiro modelo treinável.

Na confirmação das 13:22:29, o último tick tinha aproximadamente 1,1 segundo,
a janela estava em **50/50 candles**, sem aquecimento restante, e não havia
contrato ativo no estado persistido. Essas observações comprovam atividade
e prontidão da janela naquele instante; não garantem continuidade futura.

## Avanço após a correção

| Etapa | Evidência da sessão |
| --- | --- |
| Bootstrap | 1.598 rótulos disponíveis, acima do piso estrutural de 593. |
| Primeiro modelo | `RandomForest_regularized_1/temperature`, aprovado às 12:54:42, cerca de cinco minutos após o início da sessão. |
| Complemento posterior | Rodada com 24 consultas, 2.776 candles e 16.000 ticks adicionados; retreino disparado com 336 IDs novos. |
| Challenger | Treinado com 1.934 rótulos; reprovado às 13:03:53 e mantido em diretório isolado. |
| Diagnóstico mais recente | 2.002 rótulos, 68 IDs novos após a última tentativa, faltando 52 para o lote de 120. |

A recuperação voltou a fazer consultas quando faltavam IDs novos. O último
diagnóstico informa 12 consultas sem adições e `stop_reason=no_progress`.
Isso difere do encerramento incorreto por `ready` com zero consultas observado
antes da correção. Não demonstra que todas as lacunas existentes sejam
recuperáveis nem que o complemento será sempre suficiente para outro treino.

O cache privado de GBPJPY foi lido sob lock compartilhado não bloqueante. O
hash do CSV coincidiu com o manifesto. Havia 5.614 candles e 626.915 ticks,
sem ausência de OHLC nas últimas 50 posições fechadas e sem quarentena nesse
intervalo. As duas exclusões persistidas estavam fora da janela recente.

## Modelo ativo e challenger são diferentes

O SHA-256 do arquivo ativo foi conferido sem desserialização e coincide com
o sidecar e com a identidade usada nas decisões e no shadow:
`9471f70c4b8c3e99697dcb4b0cd8cc6d91dfcd5bb7e8f146fba3a6e69eb87257`.

O modelo ativo obteve **35 acertos em 74 contratos históricos, 47,3%**, e foi
aceito pela política atual. Seu teste tem 320 linhas; linhas e contratos
selecionados não são a mesma contagem. A AUC foi **0,4352**.

O challenger posterior obteve **36/85, 42,4%**, abaixo do piso de 45%, e não
substituiu o ativo. Seus artefatos permanecem em
`.training_candidates/retrain-1793721739647733381/`.

O `training.stage=model_rejected` publicado refere-se à última tentativa.
Apresentá-lo como situação única da IA esconde que o modelo anterior continua
ativo e avançando no shadow. Os 68/120 rótulos controlam outro retreino;
**não são requisito para o modelo ativo continuar seu shadow**.

A aprovação offline merece revisão de qualidade antes de reduzir a validação:
no cenário hipotético de payout líquido fixo de 85%, a expectativa histórica
do ativo foi **−0,125 unidade por contrato**. O equilíbrio aritmético nesse
cenário é aproximadamente 54,05% de acertos. O replay com payouts observados
teve somente **5/14, 35,7%**, expectativa de −0,357 e amostra insuficiente.
O cenário fixo de 70% também foi reprovado, com 14/44. Esses dados são
avaliações históricas, não resultados de compras desta sessão nem previsão
de desempenho futuro. Os critérios não foram alterados nesta auditoria.

## Shadow e demora restante

O arquivo `shadow_state.json` registra uma previsão SELL às **13:05:00**,
liquidada às **13:20:00**, com resultado incorreto. Há **1/30 resultado**,
nenhuma previsão pendente no instante consultado e nenhuma aprovação ou
reprovação final. Essa previsão não representa uma compra.

A política continua exigindo 30 resultados sem sobreposição, acerto mínimo
de 55% e payout líquido mínimo de 70%. A avaliação de validação curta em
[AVALIACAO_VALIDACAO_CURTA_2026-11-03.md](AVALIACAO_VALIDACAO_CURTA_2026-11-03.md)
permanece uma proposta; não foi ativada.

Candles de cinco minutos permitem atualizar a análise a cada fechamento,
mas não reduzem o vencimento de 15 minutos nem o número exigido de contratos
forward. Com 29 resultados restantes, seriam necessárias **pelo menos 7h15
de contratos válidos**, contadas a partir de uma próxima entrada elegível,
sem gaps, empates ou intervalos entre entradas. O tempo real pode ser maior.

O arquivo de estado é mais atual que o resumo no painel: o status ainda
publicava zero resultados e uma previsão pendente. Em `executor._handle_tick`,
`settle_tick` publica o resumo somente quando retorna `approved` ou `rejected`.
Os retornos intermediários `settled`, `tie` e `expired_with_gap` não atualizam
esse resumo imediatamente. O progresso foi persistido; a apresentação ficou
atrasada até outra captura publicar status.

## Recuperação e conexão

Na primeira leitura, o executor publicava `training_history_recovery` e
0/50 candles. Pouco depois voltou a 50/50 e `shadow_validation_pending`.
Portanto, houve indisponibilidade transitória da janela, seguida de recuperação;
não foi comprovado um travamento permanente ou perda dos resultados shadow.

O log da nova sessão contém um timeout de ping/pong do **executor** às
13:19:45, seguido de reconexão. As respostas do keepalive JSON haviam sido
48/48; isso não impede timeout do ping/pong do transporte. O diagnóstico
`training.feed.ping_timeouts=0` contabiliza apenas linhas de erro do coletor,
conforme `training_history.feed_diagnostics`. Esse campo não demonstra que
o executor esteja livre de timeouts.

O timeout precede a recuperação observada, mas não foram medidas latências
dos callbacks, perfil de CPU ou estado privado em memória. Não é possível
atribuir a causa à rede, ao treinamento ou a conflito de candles somente
com esses arquivos. As contagens de decisões publicadas representam uma
janela recente, não totais acumulados de bloqueios na sessão.

## Melhorias prioritárias decorrentes desta leitura

1. Corrigir a observabilidade da IA: publicar liquidações intermediárias do
   shadow e distinguir modelo ativo, progresso forward e último challenger.
   Separar também a saúde da conexão do executor da saúde do coletor.
2. Medir os motivos e a duração das perdas da janela privada: lacuna recente,
   fechamento ainda não disponível, quarentena ou cache ocupado. Priorizar
   reparo recente sem descartar o histórico ou reiniciar o shadow.
3. Rever a política de aprovação offline da IA antes de encurtar o shadow:
   a elegibilidade atual aceita modelos com expectativa histórica negativa.
   Validar qualquer nova política em avaliação temporal isolada.
4. Depois dessa revisão, decidir a política forward curta estudada, com
   versão e identidade próprias. Recuperação histórica não deve ser contada
   como resultado futuro. Candidatos sem dados podem ceder lugar; este modelo
   já em shadow deve preservar seu progresso.

## Escopo e verificações

Foram confrontados status, logs filtrados, metadados, avaliações, código dos
consumidores e estado persistido. Foram conferidos hashes do modelo e do cache,
sem carregar pickle. Não foram feitas consultas à conta ou à API, treinamento,
compra, promoção, reinício ou alteração de código, parâmetros e dados
operacionais. Score Ponderado e coleta permanecem fora das alterações.

Somente este relatório foi acrescentado. A validação documental confere os
links locais e o whitespace; a suíte não foi reexecutada para esta análise.
A conferência global de whitespace encontrou ocorrências nos CSVs operacionais
preexistentes, com terminações CRLF; esses arquivos não foram modificados pela
auditoria. A conferência específica do novo relatório passou.
