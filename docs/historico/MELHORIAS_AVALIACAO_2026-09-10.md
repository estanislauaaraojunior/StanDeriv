# Diagnóstico e avaliação temporal — 10/09/2026

## O que foi aplicado

- `training_evaluation.py`: avaliação única reutilizada pelo treinador. Compara IA,
  técnico e combinação, mantendo risco e contratos sem sobreposição em cada fluxo.
  Contabiliza o primeiro bloqueio por linha, segmentos de contratos selecionados
  (direção, hora UTC e confiança), drawdown realizado, expectancy, Brier multiclasses,
  log loss e curva de confiabilidade em faixas de probabilidade. As faixas usam
  todas as linhas, inclusive sobrepostas, e não representam amostras independentes.
- `train_model.py`: três janelas de ajuste → calibração → avaliação com fit crescente.
  Há embargo entre janelas e partições. O holdout final não participa da seleção.
  O ranking prioriza pelo menos 30 contratos acumulados, pior expectancy, média de
  expectancy, drawdown, log loss e AUC. O ajuste final mantém a divisão temporal
  anterior. Relatórios e hiperparâmetros são persistidos em `training_reports/`.
- `pipeline.py`: gatilho de 120 novos contratos rotulados, em vez de 400 ticks;
  mesmo o backstop temporal precisa de informação nova. O piso estrutural para
  tentar treinar passou a 1.815 amostras no timeframe de 1 minuto e duração de 15.
  Dados disponíveis podem ainda falhar por distribuição de classes ou lacunas.
- `payout_quotes.py`: duas propostas, CALL e PUT, por intervalo de cinco minutos,
  com stake de referência de US$ 1. Correlação por request ID, prazo de 30 segundos,
  deduplicação de respostas e escrita com lock. Nunca envia compra. Falhas de
  amostragem não interrompem a persistência de ticks.
- Avaliação com payouts observados é separada dos cenários fixos. Faz busca as-of
  por ativo, duração, unidade, direção e moeda, com idade máxima de 900 segundos.
  Não usa cotações futuras nem substitui ausência por payout inventado. A razão
  de payout vem da stake cotada: diferenças por stake simulada não são modeladas.
  O relatório identifica o hash do conteúdo de cotações normalizado em texto.
- `feature_engine.py` e `dataset_builder.py`: oito colunas experimentais de retornos
  de 15/30/60 minutos, volatilidade de 30 minutos, distância relativa às médias e
  horário UTC cíclico. As 27 features operacionais e seu esquema são preservados.
  Janelas/timeframes sem cobertura para 60 minutos não podem executar o experimento.
- `training_experiments.py`: comparação baseline × contexto no mesmo development,
  sem usar os resultados finais para selecionar modelos e sem publicar bundle.
- Dashboard: resultados dos componentes, amostra walk-forward, calibração,
  cobertura de payouts e advertências do candidato.

Atualização posterior solicitada pelo usuário: aprovação exige taxa de acerto observada de pelo menos 60%, com um ou mais contratos, e AUC acima de 0,50. Wilson permanece como diagnóstico, sem bloquear aprovação. O payout líquido mínimo de entrada é 60%, comum à simulação, demo e real.
O cenário de 70% líquido e a consistência walk-forward são diagnósticos adicionais;
nenhum threshold foi reduzido para aprovar candidatos. O executor continua
validando o payout efetivamente recebido antes de qualquer compra.

## Limites da interpretação

Cada janela e componente inicia sua simulação de risco separadamente. Médias de
expectancy entre janelas não são retorno de uma carteira contínua. A falta de
contratos em uma janela produz expectancy zero, não evidência de estabilidade.
As janelas finais de treinamentos sucessivos podem se sobrepor; seus resultados
não devem ser somados como contratos independentes nem usados para otimização
manual repetida. A independência aqui assegurada é entre desenvolvimento e
holdout dentro de cada execução, com teste de regressão explícito.

Uma aprovação estatística exige evidência de previsão: executar mais modelos ou
mais vezes sobre o mesmo período não garante aprovação ou rentabilidade.

## Avaliação com dados reais

Foi usado um snapshot isolado de ticks de `frxUSDCAD`. A identificação do dataset,
configuração e resultados completos estão em
`training_reports/improvements-2026-09-10/`. Nenhum modelo operacional foi
substituído por essa experiência.

| Perfil / modelo | Contratos walk-forward | AUC média | Expectancy média entre janelas |
| --- | ---: | ---: | ---: |
| Baseline / RandomForest | 15 | 0,5218 | 0,0822 |
| Baseline / XGBoost | 13 | 0,4943 | 0,0590 |
| Contexto / RandomForest | 15 | 0,5036 | 0,0822 |
| Contexto / XGBoost | 20 | 0,4743 | -0,1623 |

Esses valores usam payout líquido hipotético de 70% e amostra pequena. As features
adicionais não demonstraram benefício consistente, portanto ficaram como
experimento, sem alteração da inferência operacional.

O candidato baseline escolhido nesse snapshot teve AUC final de 0,4894,
22 contratos e 8 acertos, permanecendo reprovado. Isso valida o funcionamento da
avaliação, não demonstra qualidade suficiente para operar.

## Verificação

A suíte passou com 300 testes em cópia temporária sem credenciais operacionais.
Há regressões para embargo e holdout, alteração exclusiva dos rótulos finais,
calibração, empate, filtro combinado, seleção por estabilidade, JSON do XGBoost,
features relativas, causalidade de payouts, deduplicação, ausência de compras,
gatilhos de retreino e tolerância à falha da telemetria. Compilação Python e
verificação de sintaxe JavaScript também passaram.

## Aplicação ao processo ativo

O pipeline foi reiniciado em demo, com `frxUSDCAD` e reutilização dos dados.
A autenticação foi confirmada na API. O primeiro treino operacional com a nova
seleção concluiu e registrou `selection_method=walk_forward_risk_calibration_v1`.
O candidato permaneceu reprovado, como no snapshot de diagnóstico.

A consulta ao vivo de payout recebeu `ContractBuyValidationError`. Uma requisição
independente confirmou a mensagem da Deriv: “Trading is not available from
20:50:00 to 22:00:00.” A ausência de cotações nessa janela foi preservada, sem
inventar valores nem enviar compras. O amostrador continua consultando a cada
cinco minutos e registra propostas válidas quando disponíveis. A escrita com
resposta válida e a seleção causal dessas cotações foram verificadas nos testes.


## Limites solicitados — 2026-09-11

- Configuração central: `MIN_NET_PAYOUT=0.60` e `MIN_APPROVAL_WIN_RATE=0.60`.
- Entradas abaixo do payout líquido mínimo são bloqueadas no backtest, na
  simulação de avaliação e nas propostas de compra, tanto em demo quanto real.
  A fronteira monetária usa cálculo decimal para aceitar exatamente 60%.
- A aprovação exige pelo menos um contrato e acerto observado de 60% ou mais,
  independente do tamanho da amostra. Wilson fica como diagnóstico; AUC acima
  de 0,50 e comparação com o champion continuam exigidos.
- Retorno esperado mínimo, limites de risco, stake e stops permanecem.
- A política integra o contrato dos modelos. Artefatos anteriores ficam
  preservados, mas precisam de reavaliação sob a política atual.
- Validação: 341 testes passaram em cópia isolada, compilação dos módulos e
  `git diff --check`. Reinício em conta demo autenticada, sem contrato pendente,
  com reutilização dos dados e símbolo inicial `1HZ25V`.

## Atualização de política — 2026-09-16

- Por solicitação operacional posterior, `MIN_APPROVAL_WIN_RATE` passou para
  `0.45`: a estratégia completa precisa ter pelo menos um contrato e taxa de
  acerto observada de 45% ou mais para ser elegível.
- AUC abaixo de 0,50 deixou de bloquear sozinha a aprovação; passa a ser
  registrada como alerta diagnóstico de probabilidade fraca. Os filtros de
  payout líquido mínimo, confiança, retorno esperado não negativo, risco,
  modo demo/real e comparação com champion permanecem separados.
- `AI_CONFIDENCE_MIN` foi alinhado ao piso de aprovação de 45% para não bloquear
  candidatos aprovados antes da consulta econômica da proposta. A compra ainda
  exige payout mínimo e retorno esperado não negativo pela cotação recebida.
