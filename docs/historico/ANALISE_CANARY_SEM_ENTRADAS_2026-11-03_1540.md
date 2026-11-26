> **Consulta histórica:** este registro mantém os achados e resultados da data
> indicada, sem redefinir o comportamento atual. Consulte o [README](../../README.md)
> e a [homologação de Score e coleta](../HOMOLOGACAO_SCORE_COLETA_2026-10-15.md).

# Canary ativo sem entradas — 03/11/2026, 15:40–15:42 SP

## Conclusão

A nova sessão, iniciada às **14:37:51**, carregou o ajuste. O status publica
`isolated_ai_canary_v1`, mínimo de oito resultados e 87,5% de acerto. Não se
trata de processo ainda executando a regra antiga.

O bot não comprou porque o modelo ativo **não passou no forward**:
**zero acertos em três resultados concluídos**, saldo simulado de −3 unidades.
Mais importante: **já não pode atingir os sete acertos exigidos**, mas a
implementação só registra reprovação quando completa os oito resultados.
Enquanto isso, mantém `shadow_validation_pending` e não aciona a seleção
de outro mercado. O ajuste anterior encurtou a amostra, mas deixou essa
espera por uma reprovação matematicamente certa.

## Evidência de funcionamento e bloqueio

Às 15:40:20 o tick tinha cerca de 0,3 segundo, a janela privada estava em
**50/50 candles**, sem aquecimento restante, e o diagnóstico tinha **2.749
rótulos para piso de 593**, com capacidade temporal de 111 contratos na
validação e 185 no teste. Os três resultados canary pertencem ao mesmo
modelo e não houve perda do progresso original.

O executor publicava SELL, com confiança de aproximadamente 61,6%, mas
`executor._evaluate_entry` removia o sinal antes da proposta de entrada por
falta de aprovação shadow. As 120 decisões recentes tinham esse mesmo
motivo. O estado não tinha contrato ativo e o CSV de operações não continha
fechamento de contrato com entrada nessa sessão. Propostas amostradas para
cotar payout não constituem compras.

O hash do modelo ativo foi conferido sem abrir pickle e coincidiu com os
metadados e o canary. Continua sendo o RandomForest aprovado offline às
**12:54:42**, com 35/74 acertos históricos, 47,3%, AUC 0,4352 e expectativa
negativa no cenário histórico de payout fixo de 85%.

## Sequência forward observada

Política curta ativada às **14:42:02**, com seus próprios resultados futuros:

| Entrada SP | Vencimento SP | Previsão | Payout líquido | Resultado |
| --- | --- | --- | --- | --- |
| 14:50 | 15:05 | SELL | 91% | Incorreto |
| 15:05 | 15:20 | SELL | 76% | Incorreto |
| 15:20 | 15:35 | SELL | 72% | Incorreto |
| 15:35 | 15:50 | SELL | 71% | Pendente na leitura |

Nos três casos concluídos, o preço de saída foi maior que o de entrada.
Isso explica os erros dessas previsões SELL; não justifica inverter o modelo
ou escolher BUY retrospectivamente. Confiança prevista não é taxa de acerto
comprovada. A validação original preservava sete resultados, com um acerto,
e a mesma previsão pendente; esses sete não são sete resultados do canary.

Com três erros, restam cinco resultados e o máximo possível é **5/8**.
A impossibilidade já existia às **15:20**, após o segundo erro: seriam no
máximo seis acertos. Sete acertos em oito permitem no máximo uma perda.

Sem gaps, empates ou esperas adicionais, o oitavo vencimento dessa sequência
seria às **16:50**. O processo pode, portanto, continuar esperando mais de
uma hora após esta leitura para registrar algo que já era certo desde 15:20.
O tempo adicional não pode levar esse canary à aprovação.

## Reprodução offline do fluxo

Foram usados os três registros concluídos reais como entrada para os métodos
atuais de `ShadowMonitor` e `CanaryMonitor`, extraídos por AST. O contexto
usou somente biblioteca padrão, configuração mínima explícita e JSONs em
diretório temporário. Não houve import de configuração operacional, leitura
de `.env`, modelo desserializado ou acesso à rede.

Nos estados de 1/8, 2/8 e 3/8, todos com zero acertos, o retorno foi `settled`,
com `approved=false` e `rejected=false`. A reprodução confirmou a espera
descrita nos arquivos, sem modificar os originais.

`CanaryMonitor.settle_tick` delega a liquidação ao monitor original. Este só
decide aprovação/reprovação com `len(recent) >= minimum_results`. A validação
de persistência do canary também prevê marcadores finais apenas aos oito.
Não existe a condição de impossibilidade `acertos + restantes < mínimo`.

`pipeline._training_model_resumable` só descarta o bundle quando existe
`rejected_at_epoch`. Como o marcador continua vazio, o modelo permanece
resumível. A seleção por dados preserva o shadow de um modelo compatível e
o callback de troca por reprovação ainda não é chamado. Essas proteções
acabam prolongando a permanência de um candidato que já não pode passar.

## Retreinos da sessão

| Conclusão SP | Modelo candidato | Contratos históricos | Resultado |
| --- | --- | --- | --- |
| 14:49:07 | HistGradientBoosting | 39/92, 42,4% | Reprovado: abaixo de 45%. |
| 15:10:35 | LogisticRegression | 24/56, 42,9% | Reprovado: abaixo de 45%. |

Esses candidatos não foram promovidos. Os arquivos temporários dos estimadores
foram removidos pela limpeza existente, preservando relatórios/metadados.
Logo, recuperar histórico e retreinar não significou melhorar o modelo que
está emitindo as previsões ao vivo.

O diagnóstico tinha 119 IDs novos após a última tentativa e faltava um para
o lote de 120. Essa espera controla outro retreino; não é a causa da falta
de aprovação do modelo ativo e não deve ser confundida com progresso forward.

## Payout, conexão e latência

Os três resultados e a previsão pendente entraram com payout acima do piso.
A cotação PUT observada às 15:40:10 estava em 73%. Portanto, payout baixo
não explica o bloqueio principal desta leitura.

O executor registrou dois timeouts de ping/pong, às **14:44:17 e 15:11:02**.
O coletor teve um encerramento remoto às **15:06:41**, seguido de reconexão.
Na leitura das 15:41:42 havia 901 ticks únicos nos últimos 15 minutos, com
gap máximo de um segundo e tick final de aproximadamente um segundo de idade.

O contador às 15:40:20 media **310 de 3.583 trechos de análise acima de um
segundo**, aproximadamente 8,65%, com máximo de **8,07 segundos**. As últimas
amostras publicadas estavam na ordem de milissegundos. O cache de previsão
não eliminou toda a lentidão do trecho medido.

Esse tempo inclui janela privada, possível captura shadow, análise, escrita
de status e contenção de execução; não mede só o estimador nem o callback
inteiro. Os contadores não permitem identificar qual etapa causou os picos
ou provar que eles causaram os timeouts. São um problema secundário a medir
por etapa, não uma explicação alternativa para o canary sem chance de passar.

## Correção indicada

1. Rejeitar antecipadamente um canary quando **acertos + resultados restantes
   forem inferiores ao mínimo exigido**, sem permitir aprovação antecipada,
   baixar limiares ou apagar o histórico. Aqui isso teria ocorrido no segundo
   erro, às 15:20. Persistência, retomada e status precisam reconhecer esse
   término antecipado, respeitando eventual previsão simulada pendente.
2. Acionar a busca de outro candidato apto ao confirmar a inviabilidade,
   preservando progresso de candidatos ainda viáveis e a identidade dos
   reprovados. Não reusar o mesmo bundle para repetir a sequência até passar.
3. Tornar visível o limite de aprovação ainda possível e distinguir espera,
   reprovação inevitável e falta de dados. Oito resultados com sete acertos
   são uma exigência de qualidade; encurtar o tempo não garante mais entradas.
4. Medir a latência das etapas privadas da IA para localizar os picos antes
   de alterar coleta ou parâmetros de conexão homologados.

Nenhuma dessas correções foi aplicada nesta auditoria: o pedido foi uma nova
análise. A recomendação não é forçar uma compra com o modelo atual.

## Escopo e verificações

Leitura de status, código, metadados, logs filtrados e ledger; hash do modelo;
ticks, cotações e operações sob lock compartilhado não bloqueante; reprodução
determinística isolada. Não foram executados treinos ou a suíte completa,
pois não houve alteração de código. Somente este relatório foi acrescentado.
Nenhuma API, conta, compra, reinício, modelo, estado financeiro, Score ou
coleta foi alterado pela auditoria.
