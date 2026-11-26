> **Consulta histórica:** este registro mantém os achados e resultados da data
> indicada, sem redefinir o comportamento atual. Consulte o [README](../../README.md)
> e a [homologação de Score e coleta](../HOMOLOGACAO_SCORE_COLETA_2026-10-15.md).

# Canary da IA isolada e acompanhamento — 03/11/2026

Implementação autorizada pelo pedido “aplique o ajuste e teste”, após as
auditorias da ausência de contratos. Restringe-se ao Modelo de Treino isolado
de 5 minutos, seus gates de execução e diagnóstico. Não reinicia o programa,
não substitui modelos e não altera Score Ponderado ou coleta homologada.

## Política aplicada

`training_canary.py` implementa `isolated_ai_canary_v1`: holdout histórico
elegível com pelo menos 30 contratos e **8 resultados forward não sobrepostos**,
dos quais **pelo menos 7 corretos**, com saldo unitário positivo. O prazo
teórico mínimo é de **duas horas desde a primeira previsão elegível**; espera
por fechamento, payout, abstenções, gaps e empates podem ampliá-lo. Não há
aprovação antes do oitavo resultado nem janela móvel até passar.

Os pisos de confiança e payout observado permanecem os existentes. A
liberação continua restrita a demo, com os mesmos gates de probabilidade,
retorno esperado, duração, autenticação, proposta e compra. Passar no canary
não garante que haverá imediatamente uma entrada elegível ou desempenho
positivo futuro. Oito resultados têm menos evidência que os 30 anteriores.

## Identidade, histórico e preservação

A admissão confere símbolo, hash do modelo, timeframe 300, vencimento 900,
aprovação offline, hash do dataset, contagem de contratos, holdout separado
da seleção, divisões com embargo e cronologia anterior ao treinamento.
É uma conferência da evidência do teste existente, não outro teste independente.

O canary usa schema 2 em `shadow_state.json.canary/`, com política completa,
época de congelamento, prova histórica, previsões e resultados. A identidade
inclui bundle, símbolo, timeframe, vencimento e política. Resultados anteriores
à ativação não entram na contagem curta. Reprovação é terminal; arquivo
inválido não permite apagar/reiniciar a tentativa. O ledger não readmite
outro bundle com período histórico sobreposto já utilizado nessa política.

O arquivo original `shadow_state.json` e seus snapshots continuam sendo
usados por `ShadowMonitor`, com 30 resultados/55%. O monitor composto deixa
esse fluxo acumular suas previsões e liquidar qualquer pendência preexistente,
sem importar os resultados para o canary. Se a validação original concluir
antes, ela continua podendo autorizar pela regra original; o status identifica
`approval_path=legacy_30`. Nenhuma aprovação ou reprovação antiga é convertida.

Bundles sem prova suficiente, com período já consumido ou modos combinados
mantêm a regra original. A nova política não muda a especificação do modelo,
os limiares globais de `config.py` ou o piso de dados para treinamento.

O pipeline consulta também a aprovação/reprovação canary para preservar o
bloqueio de bundle reprovado e impedir promoção sobre um shadow incompleto.
Não foi removida a exigência de comparação temporal entre champion e challenger.
Essa comparação, a preservação de estimadores candidatos não promovidos e
eventual mudança dos critérios de aprovação offline permanecem trabalhos
distintos; não se deve contornar falta de período futuro comum.

## Custo da inferência e diagnóstico

- A IA isolada reutiliza a previsão se os 50 OHLC, preços, modelo e limiar
  forem idênticos. A validação causal precede o cache; alterações de janela,
  preço, modelo, limiar ou reset invalidam o resultado. Modos combinados
  mantêm a inferência original sem esse cache.
- O executor publica também liquidações intermediárias, gaps e empates do
  shadow. O status diferencia contagens originais e canary, época da política,
  amostra histórica e resultado unitário simulado.
- A duração publicada mede somente `_handle_training_tick`, não todo o callback
  WebSocket. Os contadores de timeouts separam executor e coletor; os campos
  originais do coletor mantêm seu significado.
- O heartbeat do pipeline não sobrescreve mais os 50 candles privados da IA
  com o aquecimento técnico de 150 candles.

O teste determinístico de cache executa 120 consultas dos mesmos candles com
uma única chamada ao estimador e à extração de features, sem perder a checagem
de candle fechado. Isso demonstra redução de trabalho repetido; não prova
que a causa dos timeouts observados tenha sido eliminada. A confirmação em
execução dependerá das próximas medições.

## Validação e ativação

Testes executados em cópias temporárias, sem `.env`, dados ou modelos
operacionais. Rede/DNS foram bloqueados; carregamento de modelos foi permitido
somente dentro da cópia temporária. A suíte pode ajustar modelos sintéticos
apenas nesses diretórios. O fluxo de proposta usa mocks e não envia ordens.

Cobertura inclui aprovação 7/8 versus reprovação 6/8, ausência de aprovação
precoce, retomada e corrupção de estado, identidade, embargo e ledger,
preservação de resultados/pendências originais, conclusão original antes do
canary, payout mínimo, gaps, empates, gates de demo/entrada, causalidade e
invalidação do cache, status e regressões de modos combinados.

Resultado final em Python **3.14.4**: **816 testes passaram em 89,12 segundos** na cópia
`/tmp/standeriv-5m-validation-o7lqplw3`, com o executor de isolamento
`/tmp/standeriv_5m_run_tests.py`. Rodadas direcionadas anteriores também
passaram. Conferências de sintaxe, links locais e whitespace dos arquivos
alterados passaram. A implementação original de `shadow_monitor.py` foi
comparada à cópia feita no início desta tarefa e permaneceu idêntica.

A conferência isolada dos metadados do modelo operacional, sem carregar
pickle, encontrou 74 contratos históricos e evidência suficiente para
admissão. Nenhum arquivo canary foi criado no diretório operacional pelos
testes ou por essa conferência. O processo existente continua com o código
que já havia carregado; a nova política começa quando ele carregar a versão
atualizada, com seus resultados futuros separados do progresso original.
