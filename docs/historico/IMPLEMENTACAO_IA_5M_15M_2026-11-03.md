> **Consulta histórica:** este registro mantém os achados e resultados da data
> indicada, sem redefinir o comportamento atual. Consulte o [README](../../README.md)
> e a [homologação de Score e coleta](../HOMOLOGACAO_SCORE_COLETA_2026-10-15.md).

# IA em 5 minutos, contratos de 15 minutos e complemento incremental — 03/11/2026

## Resultado

O Modelo de Treino isolado passou a usar 50 candles fechados de 300 segundos
para prever contratos com vencimento de 900 segundos. O perfil está integrado
ao dataset, inferência, pipeline, buffer privado de execução e identidade do
shadow. Score Ponderado, coleta e modos combinados preservam sua base de
900 segundos e o aquecimento técnico de 150 candles.

O complemento reutiliza os ticks e caches existentes antes de buscar informação
nova. Não precisa esperar que a coleta ao vivo reproduza todo o histórico.
Dados insuficientes, ausentes ou conflitantes continuam impedindo a entrada.

Nenhum serviço foi iniciado ou reiniciado, nenhum modelo foi treinado/promovido
para operação e nenhum contrato foi enviado nesta entrega.

## Caminho dos dados

1. A IA identifica seu próprio timeframe em `model_contract.model_timeframe_sec`:
   300 somente quando IA está habilitada e Score/PA/padrões estão desligados.
   Os demais modos continuam usando `CANDLE_TIMEFRAME_SEC=900`, sem alterá-lo.
2. Ticks válidos de caches de 900 segundos podem alimentar o cache de 300
   segundos, sob lock e com hash/vetos verificados. O cache original permanece
   intacto. OHLC de 900 segundos não é dividido artificialmente em três candles.
3. A primeira preparação local tem orçamento de rede zero. Se já houver dados
   temporalmente prontos e, após uma tentativa, IDs novos suficientes para o
   lote de retreino, não solicita novos lotes apenas para preencher as 8.064
   posições possíveis de 28 dias.
4. Se faltar informação, a recuperação do ativo atual encadeia até quatro
   lotes, com teto total de 48 consultas/45 segundos. Candidatos alternativos
   usam um lote de até 12 consultas/15 segundos. Cada lote preserva progresso;
   prontidão, falta de ganho, busy, backoff, cancelamento ou orçamento encerram
   a sequência.
5. O executor usa 50 candles privados de 5 minutos, atuais, fechados e contínuos.
   Leitura/complemento rodam antes do WebSocket ou em worker com cooldown de
   60 segundos. Os callbacks de ticks fazem somente a atualização em memória
   e a avaliação com o contexto disponível.
6. Para lacunas recentes, o worker consulta as últimas seis horas, com até
   dois lotes/24 consultas/25 segundos. A consulta curta não reduz a retenção
   de 28 dias do cache.

Os limites de consulta são orçamentos locais, não garantias de resposta da API.
O histórico público não precisa de autenticação; a retenção efetiva e a
disponibilidade dos intervalos continuam dependentes do provedor.

## Contagem, seleção e integridade

- O piso estrutural é 593 rótulos no perfil 300/900, mantendo 579 no perfil
  de 900 segundos. São verificadas também as partições externas, as classes,
  as divisões internas do walk-forward e a calibração final. Capacidade
  temporal não garante sinais ou aprovação.
- As oportunidades da IA podem aparecer a cada 5 minutos, mas seus rótulos
  de 15 minutos se sobrepõem. A capacidade para avaliação continua contando
  contratos sem sobreposição; não multiplica evidência por três.
- Ticks recuperados contam para a prontidão, mesmo sem aumentar o CSV da
  coleta ao vivo. As tentativas continuam consumindo somente os IDs reais do
  snapshot treinado, isolados por símbolo/política.
- Na IA isolada, candidatos já consultados são excluídos temporariamente das
  próximas pesquisas. Assim o percurso alcança ativos além dos primeiros
  cinco, sem alterar o scanner, seus filtros ou o orçamento de dois candidatos
  por rodada.
- Um bundle avançando no shadow e uma reconciliação pendente continuam
  protegidos contra troca por espera de dados.
- Busy conserva o último snapshot/diagnóstico verificado, quando disponível.
  Não se lê CSV/manifesto de uma publicação incompleta sem coordenação.
- Corrupção sem época segura não é corrigida por adivinhação. Na IA isolada,
  o pipeline pode excluir a origem inteira e preparar uma base independente
  somente de histórico público/cache verificados, registrando hash e motivo.
  O CSV original e os vetos conhecidos não são reescritos.

### Correção após reprovação — 03/11/2026

A sessão real revelou que o complemento encerrava ao encontrar uma amostra
antiga treinável, embora o próximo treino exigisse rótulos novos. A parada agora
consulta `TrainingProgress.observe` e a mesma `retry_plan` usada pelo agendador:
preserva 120 novos rótulos após reprovação de desempenho e o lote menor previsto
para insuficiência de amostra. Isso vale também entre rodadas, sem depender do
executor para recuperar histórico durante o bootstrap.

Foi reproduzido o erro com API simulada e dados temporais completos: antes da
correção, nenhuma consulta era feita após a reprovação. Com a correção, o
bootstrap recupera rótulos históricos ainda não utilizados, atinge o lote e
encerra, mantendo intactos o arquivo de progresso e os IDs já consumidos.
Regressões adicionais cobrem janela com total constante, fronteiras de 119/120
e 40/41 rótulos, retomada do progresso e isolamento entre ativos. O processo em
execução não foi reiniciado, e nenhum critério de aprovação/shadow foi alterado.

Validação desta correção: o teste que reproduzia a ausência de consultas falhou
antes do ajuste e passou depois; **117 testes direcionados passaram em 5,45s**
e a suíte completa passou com **779 testes em 95,80s**, em cópias temporárias
com rede, `.env` e leitura de modelos operacionais bloqueados. Somente
`pipeline.py` mudou entre os fontes executáveis nesta correção. Os testes
usaram histórico e API sintéticos, sem treinar ou promover modelo operacional.

## Compatibilidade e comportamento operacional

Os metadados distinguem 300/900 de 900/900. Um modelo anterior de 15 minutos não
se torna um modelo de 5 minutos por reutilizar seu arquivo. Artefatos anteriores
são preservados; um novo perfil precisa de treino e avaliação compatíveis.

O buffer privado verifica fechamento causal, alinhamento, continuidade, gaps e
quarentena. Uma vela parcial não vira OHLC completo. Um conflito entre o
histórico confirmado e uma vela observada impede a utilização daquele horário.

Propostas, compra, reconciliação, contrato ativo, payout, confiança, risco e
restrição de demo reutilizam o caminho existente. O vencimento continua em
15 minutos. O aquecimento privado não depende do buffer técnico, e não libera
entradas quando seus próprios dados estão incompletos.

## Complemento do cache existente

Foi executada uma preparação pública limitada para `frxGBPJPY`, exclusivamente
no cache privado de 300 segundos, sem autenticação ou assinatura operacional.
Foram reaproveitados **568.420 ticks existentes**. O diagnóstico resultou em
**1.303 rótulos para o piso de 593** antes do complemento conectado.

A primeira tentativa dentro do sandbox falhou na resolução de DNS. A consulta
limitada foi repetida com a permissão de rede e concluiu **8 requisições**,
acrescentando **72 candles de 5 minutos e 6.621 ticks**, sem erros ou timeouts.
O cache final contém **575.041 ticks**; o diagnóstico passou a **1.323 rótulos**,
com capacidade sem sobreposição de **45 na validação e 90 no teste**. As
divisões internas também passaram. O buffer privado confirmou os **50 candles
fechados recentes** necessários à inferência.

Esses números demonstram prontidão dos dados para uma tentativa de treino,
sem comprovar qualidade de estimador, aprovação ou uma entrada operacional.
O provedor reportou `stage=available` para as últimas seis horas consultadas.
Isso não significa que todas as posições dos 28 dias estejam preenchidas.

## Avaliação de uma validação mais curta

O pedido do usuário foi **avaliar** uma nova política. O gate atual de 30
resultados forward permanece em vigor; este trabalho não converteu avaliação
histórica em resultados de shadow.

A [avaliação detalhada](AVALIACAO_VALIDACAO_CURTA_2026-11-03.md) propõe comparar
um holdout histórico recente de pelo menos 30 contratos sem sobreposição com
um canary de 8 resultados forward, mínimo ideal de 2 horas. Seis resultados
teriam mínimo de 1h30, com ainda menos evidência. Não são substitutos
estatisticamente equivalentes aos 30 atuais.

O holdout já usado para aprovar o modelo não pode contar duas vezes como
evidência nova. A política precisaria de versão própria, critérios congelados
e identificação dos períodos e do bundle. Dados recuperados agora podem
completar a avaliação histórica; não fabricam previsões ao vivo passadas.

## Verificações

- Suíte completa em cópia isolada: **770 testes passaram em 83,51 segundos**.
- Após o ajuste final de isolamento do OHLC compartilhado e mais dois casos
  de regressão: **75 testes direcionados passaram em 5,50 segundos**.
- `.env`, rede/DNS e leitura de modelos operacionais bloqueados na suíte.
  Ajustes sintéticos realizados pelos testes produziram apenas artefatos
  temporários; nenhum estimador operacional foi treinado.
- Conferência dos fontes congelados contra o snapshot da retomada: 18 arquivos
  preservados, incluindo configuração, Score, coleta, indicadores, risco,
  autenticação, dados compartilhados e shadow.
- AST confirmou que `_CollectorThread`, `_fetch_historical_ticks`,
  `_scan_group`, `_detect_top_trending_symbols`, `_validate_deriv_token` e
  `_history_store` permaneceram idênticos ao snapshot da retomada.
- Checagem de espaçamento dos arquivos alterados e novos sem avisos. Os CSVs
  operacionais preexistentes e suas diferenças de terminadores foram preservados.

Arquivos centrais: `model_contract.py`, `dataset_builder.py`, `ai_predictor.py`,
`training_history.py`, `training_candles.py`, `training_policy.py`, `pipeline.py`
e o caminho exclusivo da IA em `executor.py`. Testes novos em
`tests/test_ai_timeframe.py`, `tests/test_ai_timeframe_executor.py`,
`tests/test_ai_training_bootstrap.py` e
`tests/test_training_history_incremental.py`.
