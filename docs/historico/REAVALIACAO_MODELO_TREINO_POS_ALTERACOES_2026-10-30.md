> **Consulta histórica:** este registro mantém os achados e resultados da data
> indicada, sem redefinir o comportamento atual. Consulte o [README](../../README.md)
> e a [homologação de Score e coleta](../HOMOLOGACAO_SCORE_COLETA_2026-10-15.md).

# Reavaliação do Modelo de Treino após as alterações — 30/10/2026

## Resultado e limites da análise

As correções da avaliação da IA isolada, da contagem de rótulos novos e da
retomada do shadow passaram nas verificações direcionadas. Entretanto, a busca
por candidatos ainda pode permanecer nos mesmos cinco ativos e deixar de
examinar um ativo com dados suficientes fora desse conjunto. Também foram
reproduzidas duas lacunas no diagnóstico temporal e no uso do cache durante
concorrência.

No histórico examinado, a espera de `OTC_NDX` tem uma causa estrutural: suas
sessões contínuas são curtas para produzir 579 amostras com a janela atual.
`frxGBPJPY`, por sua vez, dispõe de 612 rótulos e passa nas partições temporais
externas e internas verificadas. Isso permite **tentar um treino**, sem
demonstrar aprovação, qualidade da previsão ou disponibilidade operacional
atual do ativo.

Esta entrega é uma análise. Somente este relatório foi acrescentado. Não houve
alteração de código, correção de CSV operacional, treinamento de estimadores,
promoção de modelos, reinício de serviço, consulta conectada ou compra de
contratos. Score Ponderado, coleta e demais módulos foram preservados.

O [relatório anterior](ANALISE_MODELO_TREINO_2026-10-30.md) descreve o diagnóstico
anterior à implementação. Seus achados sobre contagem e perda de shadow não
devem ser tomados como descrição integral do código atual.

## Estado observado: duas sessões diferentes

Horários abaixo no fuso `America/Sao_Paulo`, em 30/10/2026.

| Observação | Sessão e configuração efetiva | Evidência e interpretação |
| --- | --- | --- |
| Aproximadamente 13:07 | Sessão iniciada às 12:24:54, IA isolada em `OTC_NDX` | Status indicava execução, `insufficient_samples`, 113/579 rótulos, revisão 0 e nenhuma tentativa concluída. As informações novas de progresso evidenciam o caminho atualizado; não houve treino ou shadow observado. |
| 17:12–17:15 | Outra sessão, iniciada às 13:57:06, somente Score Ponderado em `frxXAUUSD` | `use_trained_model=false`, `use_weighted_signal=true`, `training={}` e `shadow={}`. Status `crashed`, `running=false`, encerramento registrado às 15:52:44 e último tick às 15:52:41. |

Na última sessão, `last_error` era `processo ausente`, com `returncode=null`.
O supervisor pode registrar esse estado quando não encontra o processo; a
mensagem, isoladamente, não comprova uma exceção Python nem identifica a causa
do encerramento. Ver
[dashboard/server.py](../../dashboard/server.py), funções
`_mark_stale_bot_status` e `_mark_bot_stopped`.

O campo textual `execution_signal_mode` ainda continha `trained_model`, mas
os flags efetivos e o log da última sessão indicavam Score Ponderado. Por isso,
essa sessão não serve para validar a IA após as alterações. O campo textual
isolado não deve determinar a conclusão sobre o módulo utilizado.

Não havia `model.pkl`, seu sidecar padrão, `shadow_state.json` nem
`.training_inputs/progress.json` na leitura mais recente. A ausência de progresso
persistido é esperada antes de uma tentativa concluída: observações não gravam
nem consomem rótulos. Não foi comprovado o ciclo completo
treino → aprovação offline → shadow → liberação de entrada.

## Dados examinados offline

Foi criada uma cópia dos fontes Python e dos dados necessários em diretório
temporário. Ticks foram copiados sob lock de leitura; o cache privado da IA,
sob seu lock próprio. A preparação usou somente registros já disponíveis,
incluindo hashes, conflitos e vetos temporais. Rede, DNS, leitura de modelos e
carregamento de `.env` ficaram bloqueados.

As contagens abaixo respeitam o corte de 28 dias em relação ao relógio da
análise, candles fechados de 900 segundos e os vetos existentes. Não são
contagens de sinais de compra ou estimativas de rentabilidade.

| Ativo | Rótulos válidos / mínimo | Candles de features | Diagnóstico |
| --- | ---: | ---: | --- |
| `OTC_NDX` | 113 / 579 | 1.126 | Amostra insuficiente; sessões curtas e duas épocas em quarentena. |
| `frxXAUUSD` | Preparação bloqueada | — | Registro malformado atribuível ao símbolo sem associação temporal segura. |
| `frxXAGUSD` | 213 / 579 | 902 | Amostra insuficiente. |
| `frxEURCAD` | 182 / 579 | 1.853 | Amostra insuficiente. |
| `OTC_SPC` | 35 / 579 | 513 | Amostra insuficiente. |
| `frxGBPCAD` | 11 / 579 | 1.855 | Amostra insuficiente. |
| `frxGBPJPY` | **612 / 579** | 1.680 | Passa nos requisitos temporais leves externos e internos examinados. |

O número de candles de `OTC_NDX` antes do corte pelo relógio era 1.139.
Treze candles antigos saem da janela atual; o total de rótulos permanece 113.
Isso explica a diferença em relação ao diagnóstico anterior sem representar
perda de rótulos novos.

Para `frxGBPJPY`, as partições externas continham 255/96/35/123 linhas. A
capacidade independente da validação era **35**, e a do teste, **122**. O
development tinha 438 linhas; a seleção interna com um fold produziu
212/38/88 linhas, todas com pelo menos duas classes. O ajuste/calibração final
produziu 300/88 linhas, também com classes suficientes.

Os rótulos desse candidato iam de um sinal em 16/10 às 11:45 UTC a um
vencimento em 28/10 às 18:30 UTC. Não se supôs que esses dados provassem
atualidade do mercado. Abertura, suspensão, ADX e operabilidade precisam ser
confirmados pelo scanner no momento de uma eventual seleção.

## Achados ordenados por impacto na espera

### 1. A pesquisa de dados ainda pode ignorar candidatos fora dos cinco primeiros

**Falha confirmada no agendamento da IA.**

Em [pipeline.py](../../pipeline.py), `_maybe_request_training_candidate`
solicita `_detect_top_trending_symbols(limit=_TOP_TRENDING_SYMBOLS, ...)`,
com limite 5. O deslocamento usado para escolher dois candidatos por rodada
percorre apenas essa lista já cortada. Se o ranking permanecer estável, os
ativos nas posições seguintes nunca recebem diagnóstico nessa espera.

Reprodução com seis candidatos elegíveis e somente o sexto pronto: seis
rodadas examinaram os pares A/B, C/D, E/A, B/C, D/E e A/B. O sexto nunca foi
consultado e não houve pedido de troca. No controle, oferecendo seis candidatos
ao mesmo agendador, o sexto foi encontrado.

O scanner pode analisar dezenas de ativos para tendência sem que a IA examine
os dados de todos eles. A priorização por capacidade funciona dentro do lote
recebido, mas não recupera candidatos descartados antes desse diagnóstico.
`frxGBPJPY` ilustra por que essa distinção importa: há dados para tentar treino,
mas isso não assegura que entre no conjunto dos cinco considerados.

**Melhoria prioritária:** fazer o agendador da IA percorrer progressivamente os
elegíveis além do primeiro grupo de cinco, com orçamento limitado de preparação
por rodada, mantendo os critérios existentes de mercado, ADX, sondagem e
preferência de categorias. O candidato precisa continuar elegível no momento
da troca. Não basta aumentar globalmente `_TOP_TRENDING_SYMBOLS`, pois ele
também atende outros fluxos e a restrição permanece exclusiva da IA.

Referências: `pipeline.py:501`, `pipeline.py:2556`, `pipeline.py:2598` e
`pipeline.py:2602`.

### 2. Em OTC_NDX, continuar aguardando não resolve a insuficiência observada

**Limitação estrutural do histórico, sem evidência de erro na nova contagem.**

Antes do corte pelo relógio, foram encontrados 21 segmentos de candles
contínuos: um com 13, 19 com 57 e um com 43 candles. As 19 sessões completas
observadas iam de 03:00 a 17:00 SP. Com 50 candles por amostra, uma sessão de
57 oferece oito janelas possíveis; no histórico, sete tinham entrada e seis
tinham vencimento observável. Dezoito sessões produziram seis rótulos cada,
e a sessão de 19/10 produziu cinco: **113 rótulos**.

Mantido esse padrão, cerca de 20 sessões em 28 dias produziriam aproximadamente
120 rótulos. Mesmo uma hipótese mais generosa, com 57 candles em todos os
28 dias e todas as oito janelas aproveitadas, chegaria a somente **224**,
abaixo de 579. Essa conclusão é condicional ao padrão observado; não constitui
consulta ao calendário oficial nem previsão de sessões futuras.

Na sessão de IA observada às 13:07, a recuperação repetia rodadas parciais,
inclusive com 12 consultas e nenhum tick acrescentado, sem sair de 113 rótulos.
Recuperação sem ganho não deve ser interpretada como uma previsão de que
mais alguns minutos de coleta completarão a amostra.

**Melhoria:** reconhecer no diagnóstico da IA a baixa capacidade observada
das sessões e permitir que outro candidato apto assuma o bootstrap. Não
reduzir a janela de 50, o embargo ou o piso de amostras para acomodar esse ativo.

Referências: `dataset_builder.py:99`, `dataset_builder.py:107` e
`dataset_builder.py:113`.

### 3. frxXAUUSD está bloqueado por um registro malformado

**Problema confirmado de integridade da origem, preservada sem edição.**

A linha lógica 3.935.581 de `ticks_v2.csv` tinha 13 colunas, enquanto o esquema
espera nove. Sua primeira coluna continha época numérica válida, mas
`frxXAUUSD` aparecia na coluna de índice 6, e não no campo de símbolo de índice
2. O conteúdo bruto não foi exposto.

A mensagem `sem época segura` significa que o parser não consegue atribuir
com segurança aquela época ao símbolo deslocado. Não significa que a primeira
coluna seja necessariamente não numérica. A preparação aborta antes de chamar
o recuperador histórico, o que explica a falta de um cache útil para esse
candidato na inspeção.

**Melhoria:** identificar esse candidato como bloqueado por integridade,
expor o motivo na lista de candidatos e seguir para outro elegível. Uma
eventual recuperação deve produzir snapshot exclusivo da IA com proveniência
e vetos demonstráveis; não se deve descartar a linha silenciosamente ou
reescrever o CSV homologado como efeito desta análise.

Referências: `training_data.py:45`, `training_data.py:53`,
`training_data.py:57` e `training_data.py:99`.

### 4. O diagnóstico de prontidão não verifica todas as partições do treinador

**Falha reproduzida offline; não observada na sessão operacional.**

`readiness_capacity` verifica as partições externas e a capacidade de validação
e teste. O treinador também exige classes e tamanho mínimos nas divisões
internas do walk-forward, que podem falhar mesmo quando as externas passam.

Reprodução: 579 linhas espaçadas em 900 segundos, janela de features de
45.000 segundos e targets alternados. Tornando os targets dos índices
247 a 278 iguais a zero, as partições externas continuam com pelo menos
duas classes e capacidade 30/116; o diagnóstico retorna `ready`. Entretanto,
a calibração interna fica com classe única e `walk_forward_partitions`
lança `seleção semanal insuficiente após embargo`.

`_run_training` registra a chave do dataset e chama `mark_attempt` somente
depois que o treinador retorna. Essa exceção ocorre antes desse registro;
o bootstrap pode voltar a preparar o mesmo dataset em ciclos posteriores.
Não se deve consumir rótulos como se um treino tivesse terminado normalmente.

**Melhoria:** verificar também as partições internas sem ajustar estimadores,
representando insuficiência estrutural no diagnóstico. Distinguir essa condição
de uma reprovação de desempenho e evitar repetir trabalho caro sem mudança
relevante do snapshot. O caso real de `frxGBPJPY` passou nessa verificação;
portanto, esta lacuna não invalida sua aptidão temporal observada.

Referências: `training_policy.py:140`, `train_model.py:134`,
`train_model.py:323`, `pipeline.py:2760`, `pipeline.py:2775` e
`pipeline.py:2789`.

### 5. Cache válido pode desaparecer do diagnóstico quando o lock está ocupado

**Falha reproduzida offline; incidência operacional não medida.**

Quando o lock de recuperação pública já está ocupado, `TrainingHistory.enrich`
retorna somente os registros da origem, sem incorporar o cache da IA. O
diagnóstico fica temporariamente baseado em um conjunto menor.

Reprodução: cache validado com um registro, origem vazia e lock ocupado.
O retorno continha zero registros, `stage=busy` e zero consultas de rede.
No pipeline, o diagnóstico resultante pode ser mantido por até 60 segundos.
Isso pode adiar ou distorcer a escolha de um candidato pronto.

**Melhoria:** preservar o último diagnóstico estável ou reutilizar uma versão
validada do cache sob coordenação segura. Ler CSV e manifesto sem lock não é
uma solução, pois o par de arquivos não é atualizado em uma única transação.

Referências: `training_history.py:260`, `training_history.py:264`,
`training_history.py:272`, `pipeline.py:2499` e `pipeline.py:2534`.

## Quarentena temporal de OTC_NDX

O cache registrava vetos em 30/10 às 12:00 e 13:30 SP. Foi confirmada uma
divergência entre o OHLC da origem e o do cache para 13:30: a origem tinha
abertura, máxima, mínima e fechamento iguais, enquanto o cache divergia em
fechamento, máxima e mínima. A hipótese de candle parcial versus fechado é
compatível com a evidência, mas sua origem não foi comprovada.

Não se trata de conflito direto entre tick e candle: a chave canônica inclui
`kind` e `timeframe`. Após o conflito entre candles, o veto por época exclui
também outros registros daquele instante e qualquer amostra cuja janela
atravesse a época vetada. Todas as possíveis janelas de 50 candles da sessão
observada de 57 atravessariam esses horários. Completar a sessão de 30/10,
por si só, não produziria rótulos desse dia enquanto o veto persistisse.

Essa proteção deve permanecer. Uma futura revisão de recuperação exclusiva da
IA precisa demonstrar qual registro é autoritativo, preservar proveniência e
não alterar a coleta homologada nem simplesmente apagar a quarentena.

Referências: `market_data.py:63`, `training_history.py:371`,
`training_history.py:374` e `dataset_builder.py:79`.

## O que funcionou nas alterações verificadas

- **Avaliação isolada:** com os geradores técnicos desligados, score técnico
  zero deixa de bloquear a previsão da IA pelo score combinado. Probabilidades
  de 60% e 65% com payout líquido de 85% passaram nos casos controlados;
  56% com payout de 70% continuou bloqueado pelo retorno esperado. Com módulo
  técnico habilitado, o filtro combinado continuou exigido.
- **Dados novos:** identidades de sinal, entrada, vencimento e target distinguem
  rótulos novos mesmo quando o total da janela fica estável. Observações não
  consomem rótulos; a tentativa usa os IDs do snapshot entregue ao treinador.
  Retomada, isolamento por ativo/política, compactação e falha de escrita
  atômica foram exercitados pelos testes.
- **Preservação de shadow:** troca e retorno recuperam estado completo somente
  da mesma identidade de símbolo, modelo, timeframe e duração, incluindo
  previsão pendente, resultados, aprovação e reprovação. Estado parcial
  legado não autoriza retomada.
- **Proteções de troca:** um bundle utilizável impede a rotação por espera de
  dados; contrato pendente exige reconciliação persistida no símbolo correto.
  Challenger não substitui o bundle enquanto seu shadow estiver incompleto.
- **Reserva de bundles:** consultas e ativação validam integridade e
  compatibilidade; uma identidade reprovada no shadow não é reativada pela
  simples volta ao ativo.

Referências: `training_evaluation.py:132`, `training_evaluation.py:156`,
`training_policy.py:75`, `training_policy.py:88`, `shadow_monitor.py:43`,
`pipeline.py:2580`, `pipeline.py:2587` e `pipeline.py:3255`.

## Tempo necessário após encontrar dados suficientes

Dados suficientes autorizam uma tentativa de treino. Ainda são necessárias a
aprovação offline, a ativação compatível, as oportunidades forward e os gates
de execução. A busca por outro ativo não elimina essas etapas.

Um shadow novo exige 30 resultados não sobrepostos com duração de 15 minutos.
O mínimo teórico é **7h30 a partir da primeira observação**, se houver
oportunidades consecutivas, ticks válidos no vencimento e resultados que
atendam aos critérios. Abstenções, payout inadequado, empates, gaps e mercados
fechados podem ampliar esse tempo. Não é uma previsão de horário de entrada.

Preservar um shadow já avançado evita repetir esse período. Isso sustenta a
regra solicitada: **ativo aguardando dados pode ceder lugar; bundle avançando
no shadow deve manter seu progresso**.

Referências: `config.py:194`, `config.py:237`, `config.py:239`,
`shadow_monitor.py:147`, `shadow_monitor.py:165` e
`shadow_monitor.py:179`.

## Próximas correções sugeridas, restritas ao Modelo de Treino

1. Percorrer candidatos elegíveis além dos cinco primeiros, mantendo orçamento
   por rodada e confirmação de mercado na troca. Priorizar dados existentes
   e capacidade temporal antes de recuperar repetidamente o mesmo ativo.
2. Completar o diagnóstico com as partições internas e impedir tentativas
   repetidas de snapshots estruturalmente inviáveis.
3. Preservar diagnóstico estável durante `history_recovery=busy`.
4. Expor por candidato quantidade de rótulos, capacidade, motivo de bloqueio,
   idade dos dados e ganho da última recuperação. Diferenciar espera de dados,
   corrupção, capacidade insuficiente, reprovação de modelo e shadow em curso.
5. Reduzir prioridade de candidatos com baixa capacidade estrutural ou várias
   recuperações sem ganho, permitindo nova avaliação se os dados mudarem.

Nenhuma dessas sugestões requer alterar pesos do Score, coleta, stake,
limiares econômicos, janela de features ou gates de shadow/demo.

## Verificações e limitações

**99 testes passaram em 2,39 segundos**, em cópia temporária sem artefatos
operacionais, `.env`, rede, DNS ou leitura de modelos. Foram selecionados:

- `tests/test_training_evaluation.py`;
- `tests/test_training_scheduler_policy.py`;
- `tests/test_training_candidate_selection.py`;
- `tests/test_training_registry.py`;
- `tests/test_shadow_resume.py`;
- `tests/test_shadow_monitor.py`;
- `tests/test_training_policy.py`.

Do arquivo de avaliação, dois testes que ajustam estimadores foram
explicitamente desmarcados nesta análise:
`test_changing_final_targets_cannot_change_walk_forward_selection` e
`test_xgboost_training_writes_json_report_with_missing_marker`. O resultado
foi **99 passed, 2 deselected**. A suíte completa não foi repetida.

Adicionalmente foram executadas reproduções independentes do limite de cinco
candidatos, prontidão externa com falha interna, lock ocupado e segmentação
das sessões. As análises de dados calcularam rótulos e divisões temporais,
sem ajuste de estimadores ou substituição de artefatos em uso.

Os links locais deste relatório foram conferidos. Os 30 arquivos Python da
raiz permaneceram idênticos à cópia feita no início desta rodada. A checagem de
espaçamento do novo relatório não apresentou avisos; `git diff --check` global
apontou espaçamento/terminadores nos CSVs operacionais preexistentes de
`Histórico Ativos/`, que foram preservados.

A correção de avaliação e a retomada do shadow foram demonstradas em testes
controlados. A sessão de IA observada não chegou ao treino, e a última sessão
estava com IA desligada. Portanto, não foi validada uma entrada real ou demo
após as alterações, não foi determinada a causa do processo ausente e não há
garantia de aprovação do candidato com 612 amostras.
