# Homologação de Score Ponderado e coleta de dados — 15/10/2026

## Aceitação e regra de manutenção

O usuário relatou que o bot permaneceu funcionando sem erros e continuou
abrindo contratos. Solicitou a homologação do Score Ponderado, determinando
que o módulo não seja mais alterado, e estendeu a homologação à coleta de dados.
Esta é uma aceitação funcional declarada pelo usuário, registrada em
15/10/2026, no fuso America/Sao_Paulo.

**Score Ponderado e coleta ficam congelados.** Não alterar lógica, parâmetros,
base temporal, filtros, tratamento de lacunas ou persistência desses módulos
sem nova solicitação explícita para rever a homologação. A restrição também
vale para alterações indiretas em configuração, indicadores, dependências
compartilhadas ou consumidores. Novos trabalhos em IA, Price Action, padrões,
risco ou interface devem preservar os comportamentos homologados.

A revisão desta entrega não alterou arquivos executáveis. O commit também
consolida implementações, testes e dados locais que já estavam pendentes antes
da revisão, conforme pedido de incluir todas as alterações. Os hashes abaixo
identificam o conteúdo homologado antes de criar o commit; não representam
assinatura de processo ativo nem substituem seus metadados de sessão.

## Score Ponderado

Implementação em `strategy.py`, com indicadores de `indicators.py` e parâmetros
de `config.py`. A [especificação congelada](SCORE_PONDERADO_BOLLINGER.md) registra
pesos 25/20/15/15/15/10, modulação ADX, Bollinger 20/2, limiar 0,06, modo
balanceado e confirmação de três dos cinco votos direcionais. A chamada do
executor usa fechamentos de candles de 15 minutos, limitada aos últimos 50.
Não converter essa base para ticks nem modificar pesos para aumentar entradas.

A integração preserva maioria estrita dos módulos habilitados, motivos
específicos de abstenção e detalhes no status. A troca por mercado desfavorável
conta um fechamento por vez, com limite de 12 e proteções de contrato ativo e
cooldown. A lógica de compra pertence ao executor e não ao cálculo do Score.

## Coleta de dados

O escopo abrange o coletor do pipeline (`_CollectorThread`), o coletor autônomo
`collector.py`, leitura/persistência v2 em `market_data.py`, histórico OHLC por
ativo em `asset_history.py`, calendário em `market_calendar.py`, keepalive em
`api_keepalive.py`, controle de requisições em `request_throttle.py` e os pontos
de autenticação/recuperação em `deriv_session.py`, `pipeline.py` e `executor.py`.

Comportamentos preservados:

- Ticks com símbolo e época identificados; validação de preço, ordenação,
  deduplicação e filtro anti-spike. Registros legados de origem desconhecida
  não são promovidos a ticks nem utilizados como dados identificados de treino.
- `ticks_v2.csv` para ticks e `Histórico Ativos/` para OHLC independentes por
  símbolo, no esquema `epoch,datetime,symbol,price,kind,timeframe,open,high,low`.
  No registro candle, `price` representa o fechamento.
- Histórico de ativos sincronizado incrementalmente: consulta somente lacunas
  e candles posteriores, ordena/deduplica e grava atomicamente sob lock. O
  limite local atual é `ASSET_HISTORY_CANDLES=336` candles de 15 minutos;
  não confundir com o orçamento de treino de sete dias/672 candles.
- Calendário oficial por data UTC, com cache; só pausas comprovadas são
  atravessadas sem candles. Falhas de calendário não fabricam dados.
- Coletor do pipeline com reconexão, nova sessão OTP quando aplicável,
  prazo para primeiro tick, ping JSON a cada 30 segundos e mensagens de
  diagnóstico sem expor URLs de sessão ou credenciais.
- Backfill pesado de ticks desativado no fluxo atual do pipeline. Coleta ao
  vivo e sincronização incremental de OHLC são caminhos distintos. A existência
  de candles não cria ticks históricos nem habilita treino por si só.
- Executor preserva candles válidos nas duas margens das lacunas, solicita
  apenas faixas ausentes e correlaciona respostas por `req_id`. Recuperação
  incompleta bloqueia novas avaliações de entrada; erros/timeouts permitem
  tentativa após 60 segundos, sem duplicar assinatura de ticks ao vivo.
- Status por sessão e símbolo, com quantidade real de candles e aquecimento
  restante; heartbeat do processo não é fabricado por consultas do painel.

## Limites da homologação

A aceitação não inclui os modelos de IA, a política de risco, rentabilidade,
execução real ou ausência de falhas futuras da corretora. O código atual mantém
`ENFORCE_RISK_GATE=False`: pausas e stops são registrados, mas não bloqueiam
propostas/compras. `ENFORCE_ENTRY_CADENCE=False` permite avaliações por tick;
a base de indicadores continua sendo de candles fechados. Esses valores foram
somente documentados, sem ajuste nesta entrega.

Parar o processo não liquida um contrato na corretora. Estado de contratos e
risco deve ser preservado e reconciliado antes de retomar operações. Esta
homologação não iniciou/parou serviços, consultou conta, abriu/fechou contratos
nem treinou/promoveu modelos.

## Verificações da entrega

- Suíte completa em cópia temporária, com token fictício e sem `.env`, modelos,
  ticks, estados ou logs operacionais: **611 testes passaram em 75,97 segundos**.
  Eventuais treinamentos sintéticos da suíte produzem somente artefatos temporários.
- Revisão dos documentos Markdown do repositório: regras atuais corrigidas e
  registros datados mantidos como históricos. Arquitetura antiga duplicada
  removida de `ARCHITECTURE.md`; conteúdo preservado em `historico/ARQUITETURA_ANTIGA.md`.
- Conferência de links/caminhos locais, parâmetros, comandos e hashes; revisão
  de credenciais antes do commit. Identificadores de conta do log são ocultados
  na cópia versionada, preservando o log operacional local. CSVs e estados já
  versionados entram como snapshot solicitado; não são defaults para outra
  conta/sessão e não devem substituir estado operacional durante implantação.

## Referência de integridade

Os arquivos compartilhados abaixo são referências de conteúdo, não uma
proibição de manter outros módulos neles. Qualquer alteração futura precisa
preservar Score/coleta ou ter autorização explícita para rever a homologação.

| Arquivo | SHA-256 |
|---|---|
| `strategy.py` | `4e7f8e704df296e298f70845ca45352965f6b5e70f83f3402525d132f46dec06` |
| `indicators.py` | `dec0ee29f5bc5c7ba82e6d4c0f3c5b4c5f88ff1169bf4c1e2ce981a1a932217e` |
| `config.py` | `84481391cbe0c9d2e57e31bdd387de355a29254a4619e04de31fba7a6142a4bc` |
| `collector.py` | `7e82ef5d6771e73312c4f28afdfe675b73fdd76fdd5be1769e2d85cacdb2e8ee` |
| `pipeline.py` | `9ab346639a7c10682ade9127c49fb3f2dd9d0c9a6ebd9ebb19e908334a3d368b` |
| `market_data.py` | `df5170f7dca203d7e37a55a05f77b9e5077f27e1b5de33377e3b451ba349cb62` |
| `asset_history.py` | `f26c39d4c5947f8206a2537fbdffe9d4d08e28794dee065d0fa783c7c17a43aa` |
| `market_calendar.py` | `a4f6218e78164520a61afe964d458aa220377d85120578dd6c640001420339ac` |
| `executor.py` | `e32e9544a24a8ab21e2b8edeecedb00ed306ad98b2df6afacc81b9f4810ca08b` |
| `api_keepalive.py` | `10353f5c9e7519b9e72f4ea62900f1d1ceed91b8e3d22ec688c8dd431b4b1a04` |
| `request_throttle.py` | `f0f9d9e296d9a14ebb08afe6c5c5664ea662a6e769505f0cb49c9aec9faf12c2` |
| `deriv_session.py` | `894dd6e5b09fe6e459d3fbcadd7705894fb873a4a0acfadd4ad11d107d8f7a48` |
| `runtime_status.py` | `a9da6d75749182fb40554a0e1611b2f22a818dc1ac14fcbffaf5aeb87f86373d` |
| `operational_diagnostics.py` | `ef61b773b3606bbd35b59ec6249949443f03379ab64de188248d790afbb67e17` |
| `feature_engine.py` | `9872f5e3af3498397245e8a7978c7a5c96f4633103f02a97fbb6acc0669e7fd2` |
| `model_contract.py` | `a01581f0f0fa647a0fde70e368380d8ed904d6a78dc38c4eec25ef2e6add3e95` |
