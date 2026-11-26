> **Consulta histórica:** este registro mantém os achados e resultados da data
> indicada, sem redefinir o comportamento atual. Consulte o [README](../../README.md)
> e a [homologação de Score e coleta](../HOMOLOGACAO_SCORE_COLETA_2026-10-15.md).

# Modelo de Treino: demora para abrir contratos e seleção de símbolos

> Implementação posterior, em 30/10/2026: as correções da avaliação, do progresso
> por rótulo e da seleção/retomada foram implementadas após este diagnóstico.
> Consulte o [comportamento atualizado](../../README.md#progresso-e-escolha-de-candidatos-do-modelo-de-treino).
> Os dados e as limitações abaixo registram o estado anterior às alterações.

Análise de 30/10/2026, America/Sao_Paulo. Checkout
`refactor/estrutura-standeriv`, HEAD `20f9293`, incluindo trabalho local
preexistente. Este diagnóstico não altera políticas operacionais.

## Escopo e conclusão

O usuário pediu análise profunda do Modelo de Treino e sugeriu procurar outro
símbolo quando o atual não estiver apto. Acrescentou: **não alterar os outros
módulos**. Nenhum código executável foi alterado; este relatório é o único
arquivo acrescentado.

A espera observada ocorre antes do ajuste dos estimadores: faltam amostras
rotuláveis do símbolo escolhido. Existem também uma inconsistência reproduzida
na avaliação da IA isolada, um problema no contador de dados novos e limitações
no reaproveitamento de modelos e validação por símbolo.

Procurar outro ativo pode reduzir espera improdutiva, principalmente quando já
houver dados ou modelos aptos para ele. Trocar a cada bloqueio pode aumentar a
demora. A troca efetiva está na orquestração, examinada sem alterações.
Permanecem as restrições da
[homologação de Score e coleta](../HOMOLOGACAO_SCORE_COLETA_2026-10-15.md).

## Evidência da sessão

Snapshot de `bot_status.json` às 09:07 de 30/10/2026:

| Campo | Valor persistido |
| --- | --- |
| Início informado | 08:41:14 |
| Gerador selecionado | Modelo de Treino; demais desligados |
| Símbolo | `OTC_NDX` |
| Preparação | `insufficient_samples` |
| Amostras rotuláveis / piso estrutural | 104 / 579 |
| Ticks identificados no snapshot da IA | 96.050 |
| Candles válidos para features | 1.139 |
| Lacunas de ticks | 19 |
| Maior trecho contínuo de ticks | 5.400 segundos |
| Recuperação histórica da rodada | Parcial; 12 consultas; 5.390 ticks adicionados |
| Erros / timeouts dessa recuperação | 0 / 0 |
| Modelo ativo e estado shadow nos caminhos padrão | Ausentes |

O snapshot da IA inclui cache histórico, não apenas ticks ao vivo desta sessão.
O log já mostrava o mínimo de 500 ticks satisfeito, mas faltavam centenas de
rótulos utilizáveis. Muitos ticks/candles não garantem continuidade entre
entrada e vencimento de cada contrato.

O log mostrou avanço de 8 para 91 amostras; o status posterior chegou a 104.
A recuperação está progredindo. Não há evidência de travamento, nem previsão
confiável de quando haverá dados suficientes. O diagnóstico do trecho do feed
não registrava ping timeouts; callbacks não foram medidos.

Esses valores são de arquivos persistidos. A memória do processo não foi
inspecionada: o PID informado não estava visível no namespace da ferramenta.
Nenhuma conexão à conta foi feita para confirmar modo ou saldo.

## Achados prioritários

### 1. A IA isolada é avaliada com um filtro combinado indevido

Referências: `training_evaluation.py:132-170`, `dataset_builder.py:199-204`,
`strategy.py:304-326`.

O avaliador reconhece `model_only` e dispensa sinal/concordância técnicos, mas
continua exigindo:

```text
0,55 × abs(tech_score) + 0,45 × confiança >= 0,30
```

Com geradores técnicos desligados, o dataset pode registrar
`technical_signal=-1` e `tech_score=0`. O limiar efetivo vira aproximadamente
66,67% de confiança. O gerador operacional da IA não usa esse score combinado;
o executor ainda exige uma proposta economicamente válida.

Reprodução sintética, IA isolada, score técnico zero e payout líquido
hipotético de 85%:

| Confiança | Gate econômico | Avaliação combinada |
| --- | --- | --- |
| 60% | Aceita | Zero operações: `low_combined_score` |
| 65% | Aceita | Zero operações: `low_combined_score` |
| 67% | Aceita | Permite operações |

Evidência histórica coerente: o relatório de `frxGBPJPY` de 28/10, dataset
`04ab80fec51813ab7bfff5e238e14e4de89be56cff697d771da0a223b497cf90`, registra
112 bloqueios por esse motivo em 121 linhas de teste. Houve sete contratos no
componente combinado e 69 na IA isolada. São populações/resultados diferentes;
isso não demonstra que a correção aprovaria o modelo ou produziria lucro.

**Prioridade no treinamento:** alinhar a avaliação à decisão da IA isolada,
mantendo confiança, payout, retorno esperado, amostra e separação temporal.
Preservar a avaliação das demais configurações. Identificar a política corrigida
nos relatórios/artefatos e avaliar em dados futuros, sem reutilizar aprovação
antiga silenciosamente ou ajustar cortes para aprovar este holdout.

### 2. A espera inicial pode permanecer no mesmo símbolo sem prazo

Referências: `pipeline.py:2298`, `pipeline.py:3402-3462`,
`pipeline.py:3596-3625`, `tests/test_market_failover.py:561`.

A inicialização usa apenas `candidates[0]`. `_wait_for_ticks` aguarda dados sem
prazo próprio para considerar outro candidato. As falhas iniciais anteriores
à obtenção de métricas tampouco têm o contador do bootstrap posterior.

Existe um torneio antigo, mas sua lista de entrada fica sempre vazia. Um teste
atual exige explicitamente que ele não rode. Religá-lo não é correção local
segura: o torneio inicia/encerra coletores e promove modelos.

**Proposta:** a IA classifica candidatos por preparação/capacidade temporal e,
enquanto um aguarda dados, avalia outro elegível, preservando os caches. A
integração com a seleção do processo depende de orquestração e não foi feita.

### 3. O contador de retreino confunde total na janela com dados novos

Referências: `training_policy.py:15-23`, `pipeline.py:2858-2866`,
`pipeline.py:3002-3017`.

O alvo é `total_anterior + lote`, comparado ao total atual nos 28 dias móveis.
Quando rótulos novos substituem antigos, o total pode ficar constante.
Reprodução pura: 2.637 anteriores; entram 120 e expiram 120; total 2.637;
alvo 2.757; progresso reportado zero. Isso pode impedir retreino apesar de
informação nova suficiente.

**Prioridade no treinamento:** controlar épocas/identidades dos rótulos novos
por símbolo/snapshot e persistir progresso. Preservar os lotes e critérios de
qualidade; corrigir o significado do contador. O consumo desse progresso pelo
agendador atual exige integração fora deste relatório.

Quando só há coleta ao vivo, 30 novos rótulos de 15 minutos representam 7h30
de mercado contínuo, e 120 representam 30h. Recuperação histórica pode encurtar
essa espera; esses cálculos não são estimativas da sessão observada.

### 4. O piso de 579 não assegura capacidade para 30 contratos

Referências: `model_contract.py:15`, `dataset_builder.py:127`,
`training_policy.py:5`, `train_model.py:313`.

579 é o piso estrutural. O preflight do treinador exige também 30 oportunidades
sem sobreposição na validação e no teste. Deslocamentos de poucos segundos nos
primeiros ticks podem sobrepor contratos de candles adjacentes.

Reprodução com funções reais extraídas por AST, em dados artificiais:

| Rótulos | Condição de entrada | Capacidade da validação |
| --- | --- | --- |
| 579 | Entradas alinhadas | 30 |
| 579 | Offsets alternados de 0/2 segundos | 16 |
| 720 | Mesmos offsets | 25 |
| 800 | Mesmos offsets | 31 |

800 não é novo mínimo universal: depende dos timestamps/partições. O preflight
existente protege corretamente o treino. **Melhoria:** antecipar e informar
essa capacidade na preparação da IA, distinguindo dados preparados de condições
para avaliação/aprovação, sem reduzir o mínimo de 30.

### 5. Shadow e armazenamento limitam o benefício da troca

Referências: `config.py:234-239`, `shadow_monitor.py:42-72`,
`shadow_monitor.py:90-161`, `pipeline.py:1073-1089`.

Após aprovação offline, são necessários 30 resultados shadow não sobrepostos
de 15 minutos: **mínimo teórico de 7h30 desde a primeira previsão aceita**.
Ausência de sinais, payout inadequado, gaps, empates e fechamento do mercado
podem alongar a espera. Isso não é tempo de CPU.

Há somente uma identidade shadow ativa. A troca A → B → A não restaura o
progresso arquivado de A, mesmo voltando ao mesmo modelo. A reprodução isolada
confirmou esse comportamento. Shadow pendente não deve ser confundido com
incapacidade de continuar a validação.

A busca por outro símbolo com bundle aprovado consulta sempre o mesmo
`AI_MODEL_PATH`; a compatibilidade exige símbolo exato. Isso não é um catálogo
de modelos por ativo. Aproveitar candidatos preparados requer modelos e
validações recuperáveis pela identidade exata de símbolo, contrato e `model_id`,
com integridade verificada.

Também há diferença econômica: shadow exige confiança acima de 45% e payout
líquido de pelo menos 70%, mas não o gate de retorno esperado da compra. Na
reprodução, 17 acertos em 30 previsões com confiança de 55% e payout de 70%
aprovaram shadow, embora a proposta com essa confiança fosse recusada pelo
executor. Esse é um limite de integração, não resultado da sessão atual.

**Limite do escopo:** shadow, executor, compatibilidade compartilhada e
orquestração foram apenas examinados; nenhum gate foi enfraquecido ou alterado.

### 6. A preparação repete trabalho de disco e CPU

Referências: `training_data.py:93`, `training_history.py:377`,
`pipeline.py:2441`, `pipeline.py:2586`, `train_model.py:180`.

A preparação copia/varre o CSV compartilhado antes de filtrar o símbolo; a fonte
observada tinha aproximadamente 152,9 MB. O diagnóstico relê o snapshot. O
dataset idêntico só é reconhecido depois de reconstruído; atualizar um símbolo
também apaga o cache de diagnóstico inteiro.

Medição única em cópia temporária, com rede bloqueada e cache local já carregado:
preparação 5,262 segundos; diagnóstico 0,918 segundo. Esses 6,18 segundos são
custo parcial, sem leitura/persistência do provedor nem recuperação de rede.
Não são benchmark de treinamento ou do processo ativo.

O treino padrão compara 14 estimadores e 40 receitas/controles de calibração,
em um ou três folds. Reduzir variantes pode diminuir CPU, mas não resolve os
475 rótulos faltantes no snapshot observado nem elimina o shadow.

**Melhorias na IA:** cache por símbolo/conteúdo efetivamente alterado; snapshots
imutáveis reaproveitados entre diagnóstico e treino; preflight antes do ajuste;
evitar regravar cache sem mudança. Preservar hashes, quarentena e vetos temporais,
assim como o escritor e o formato da coleta homologada.

## Procura de candidatos recomendada

| Situação | Decisão proposta para a IA |
| --- | --- |
| Dados ou capacidade insuficientes | Preservar progresso; avaliar próximo elegível |
| Reprovado offline | Registrar motivo/dados; repetir apenas com informação nova suficiente |
| Aprovado offline, shadow pendente | Preservar identidade e continuidade da validação |
| Bundle e shadow aprovados | Priorizar, verificando compatibilidade e condições atuais |
| Nenhum candidato apto | Aguardar dados/mercado novos, com diagnóstico e backoff |

Consumir os candidatos do scanner existente, preservando catálogo, abertura,
piso ADX e preferência de mercados. Não misturar dados/contadores entre ativos
nem mudar o ativo com compra incerta ou contrato pendente. Uma sondagem favorável
não garante compra ou qualidade futura. Preservar as barreiras contra seleção
oportunista decorrente de muitas tentativas sobre os mesmos holdouts.

Sequência recomendada: corrigir a avaliação da IA isolada; corrigir a contagem
de rótulos novos; antecipar capacidade e reaproveitar preparação; depois integrar
a procura de outro candidato. As primeiras melhorias pertencem à camada de
treinamento. A troca do símbolo em execução depende de outros módulos e
permanece proposta, conforme a restrição do usuário.

## Verificações e limites

- Instruções, homologação, Git, código e testes relevantes examinados.
- Status, estado, trecho de log, metadados e relatórios anteriores lidos
  seletivamente; credenciais e identificadores de conta não foram expostos.
- Reproduções sintéticas confirmaram filtro combinado, capacidade temporal,
  janela móvel, arquivo único de modelos e não restauração de shadow.
  A medição de preparação usou cópias de dados em `/tmp`.
- Testes em `/tmp/standeriv-model-audit-byegi6xq`, Python 3.14.4, sem `.env`,
  modelos, estados ou CSVs operacionais, com conexões de rede bloqueadas.
- Oito arquivos selecionados: políticas, progresso, failover, shadow, avaliação,
  preparação, recuperação e integração. Com limite de 15 segundos por teste,
  136 passaram e um excedeu o limite no ajuste sintético de HistGradientBoosting.
  Uma execução anterior foi interrompida nesse teste demorado. O caso restante
  foi reexecutado com limite de threads numéricas em um e passou em 42,01
  segundos. Os 137 casos selecionados passaram entre as execuções; o timeout
  inicial não demonstra regressão funcional. A suíte completa não foi executada.
- A suíte ajustou modelos com dados artificiais apenas em diretórios temporários.
  Nenhum modelo operacional foi treinado, carregado de pickle ou promovido.
- Nenhum bot/coletor foi iniciado, parado ou reiniciado; nenhuma ordem foi
  enviada. Arquivos operacionais e alterações preexistentes foram preservados.

As correções recomendadas ainda não foram implementadas. Melhor preparação e
seleção podem diminuir espera evitável; não garantem modelo aprovado, abertura
de contrato ou rentabilidade.
