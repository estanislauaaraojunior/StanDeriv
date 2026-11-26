> **Consulta histórica:** este registro mantém os achados e resultados da data
> indicada, sem redefinir o comportamento atual. Consulte o [README](../../README.md)
> e a [homologação de Score e coleta](../HOMOLOGACAO_SCORE_COLETA_2026-10-15.md).

# Funcionamento após início da sessão — 03/11/2026

Atualização: a falha descrita abaixo foi corrigida no código após esta auditoria.
O complemento passou a verificar também os IDs novos exigidos pela política de
retreino. As observações registradas neste documento permanecem históricas;
o processo que já estava em execução não foi reiniciado por essa correção.

## Estado observado

Leituras entre aproximadamente 12:07 e 12:16, fuso America/Sao_Paulo. A sessão
foi iniciada às 10:22:29, com apenas Modelo de Treino habilitado. O ativo atual
é `frxGBPJPY`. A configuração de 300 segundos para análise e 15 minutos para
contrato foi confirmada nos metadados efetivamente produzidos pelo processo.

O processo publica atividade e a coleta recebe ticks atuais. Contudo, **a IA
ainda não está liberada para abrir contratos**: o candidato foi reprovado no
teste histórico. Não existe bundle operacional aprovado nem estado shadow.

O status geral alterna entre `training` e `waiting_model`, mas
`training.stage=model_rejected`: isso representa espera após reprovação, não
um estimador sendo ajustado continuamente. A indicação de 150 candles no
status publicado pelo pipeline não é o bloqueio atual de execução.

Na confirmação das 12:16, o status era `waiting_model`, o heartbeat tinha idade
de aproximadamente nove segundos, os rótulos continuavam em 1.323 e os novos
em 0/120. O diagnóstico tinha 1.955 candles de features e 64 gaps. Não havia
contrato ativo no estado persistido verificado às 12:09.

## O que funcionou

- O perfil de 5 minutos foi carregado: timeframe 300, janela de 50 candles e
  contrato de 15 minutos. O mínimo estrutural publicado é 593.
- O complemento inicial aumentou `frxXAGUSD` de 649 para 818 rótulos, sem
  depender somente da coleta ao vivo.
- O rodízio ultrapassou os cinco candidatos iniciais: scans de 39, 37, 35,
  33, 31 e 29 ativos mostram exclusões progressivas. Houve transferência real
  de `frxXAGUSD` para `frxGBPJPY`, registrada na linha 323 do log observado.
- O progresso persistido registra uma tentativa concluída no ativo atual,
  revisão 1, separando os rótulos já utilizados dos novos.
- A proteção de aprovação impediu a promoção dos candidatos reprovados.

## Treinos observados na sessão

| Conclusão, horário SP | Ativo | Resultado histórico | Motivo da reprovação |
| --- | --- | --- | --- |
| 10:29:24 | `frxXAGUSD` | 6 acertos em 9 contratos, 66,7% | Menos de 30 contratos avaliados. |
| 10:41:37 | `frxGBPJPY` | 22 acertos em 50 contratos, 44% | Abaixo do mínimo configurado de 45%. |

No último treino, `HistGradientBoosting_regularized_4` com calibração
`temperature` foi escolhido. O holdout tinha 265 amostras, das quais 50 geraram
contratos selecionados. Seu período ia de 27/10 às 04:50 a 03/11 às 10:05 SP.
Esses resultados são simulações históricas; não são operações ao vivo.

A AUC foi 0,4793 e a expectativa histórica foi de −0,186 unidade por contrato
sob payout líquido fixo hipotético de 85%. O replay com payouts observados tem
somente oito contratos históricos e três acertos; essa cobertura não comprova
qualidade no mercado atual. Estar uma vitória acima ou abaixo do limiar de 45%
não demonstra rentabilidade: nesse cenário fixo, o equilíbrio aritmético é
aproximadamente 54,05%.

Fonte: `model_metrics.json` e
[relatório do último treino](../../training_reports/f2472b464b9f86742094d9092393493d2da9e4443005411dce739361f2fdeaad-evaluation.json).

## Bloqueio atual: reprovação e renovação insuficiente

O diagnóstico permanece em **1.323/593 rótulos**, com capacidade temporal de
45 contratos sem sobreposição na validação e 90 no teste. Há dados para o
treino estrutural, mas a política de nova tentativa após reprovação por
desempenho exige **120 rótulos ainda não utilizados**. A leitura informa
**0/120**, `ready_for_training=false` e o motivo da última reprovação.

Desde a troca para GBPJPY, pelo menos 23 diagnósticos mantiveram 1.323 rótulos.
Não há evidência de que o shadow tenha sequer iniciado. Reduzir sua duração,
isoladamente, não resolveria esse estágio.

### Falha identificada no complemento após reprovação

Em `pipeline._training_enricher`, a função `ready` verifica apenas se a amostra
já suporta um treino temporal. Com os 1.323 rótulos antigos ela retorna pronta,
e a recuperação encerra com `stop_reason=ready` e zero consultas, **sem levar
em conta os 120 IDs novos exigidos para a nova tentativa**.

Os OHLC privados de 300 segundos observados terminavam às 10:00 SP, enquanto
os ticks continuavam chegando após 12:00. A quantidade de candles agregáveis
cresceu de 1.945 para 1.954, mas os rótulos não aumentaram. Gaps de mais de
10 segundos impedem que candles derivados de ticks preencham todas as janelas.

O complemento recente das últimas seis horas foi conectado ao executor, mas
esse executor aguarda um bundle compatível para iniciar. Consequentemente,
nessa fase de bootstrap após reprovação, o complemento recente não roda e o
processo volta a depender dos ticks ao vivo para renovar amostras.

Referências: `pipeline.py:2493`, `pipeline.py:2508`, `pipeline.py:2552`,
`pipeline.py:3301`, `pipeline.py:3463` e `pipeline.py:4076`.

Essa lacuna contraria o objetivo de completar histórico já disponível. A
próxima correção deve fazer a parada da recuperação considerar tanto capacidade
temporal quanto rótulos novos, e permitir complemento recente durante o
bootstrap, preservando os vetos, limites e o progresso de shadow existente.

Se depender somente de eventos futuros, candles de cinco minutos oferecem
no máximo 12 novos instantes de sinal por hora: 120 rótulos representam cerca
de dez horas em condições contínuas. Isso não é prazo obrigatório: recuperar
lacunas históricas ainda aproveitáveis pode fornecer rótulos novos sem esperar
todo esse período. A disponibilidade efetiva desses intervalos não foi
consultada nesta auditoria.

## Coleta e reconexões

Às 12:08:14, o CSV continha um tick de GBPJPY das 12:08:13, com idade de
aproximadamente 1,3 segundo. Havia 299 ticks nos últimos cinco minutos e 833
nos últimos 15 minutos; o maior gap nesse último intervalo era de 44 segundos.
A coleta está ativa, apesar das interrupções.

Foram observados oito timeouts de ping/pong na sessão. Após a troca para
GBPJPY ocorreram sete desses timeouts e um fechamento remoto, sempre com
reconexão. Os gaps do diagnóstico aumentaram de 55 para 63 e chegaram a 64
na confirmação das 12:16. Isso prejudica
continuidade, mas o log não mede a causa dos timeouts nem permite atribuí-los
com certeza à rede ou aos callbacks.

O lock privado esteve ocupado em tentativas de leitura não bloqueante. Não
foi forçado acesso ao CSV/manifesto durante publicação, nem concluído que um
lock ocupado por instantes representasse deadlock. O PID informado não estava
visível no namespace de processos das ferramentas; a atividade foi confirmada
por heartbeat, log e ticks recentes.

## Escopo desta auditoria

Foram lidos status, logs sanitizados, metadados, relatórios de avaliação,
progresso e ticks sob lock de leitura. Não foram abertos pickles, credenciais
ou `.env`. Nenhum processo, parâmetro, modelo, CSV operacional ou código foi
alterado. Somente este relatório foi acrescentado; não houve consulta à conta,
treino, API de mercado, envio de ordem ou reinício.
