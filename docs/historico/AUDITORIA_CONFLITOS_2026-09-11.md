# Auditoria de conflitos na lógica de contratos — 11/09/2026

Escopo: inspeção de configuração, dataset, treino, inferência, estratégia,
backtest, risco, orquestração e execução. Não é certificação de ausência de
outros defeitos. Não foram iniciados serviços, treinados modelos ou comprados
contratos. Código operacional preservado nesta auditoria.

## Achados prioritários

1. **Alto — Backtest geral sem filtro econômico da execução.**
   `backtest.py:547` verifica somente piso de payout, enquanto
   `model_contract.py:150` exige confiança e retorno esperado não negativo.
   Reprodução offline com provedor de sinal controlado, IA habilitada,
   confiança 0,56 e payout líquido 0,70: 3 contratos no backtest; proposta
   equivalente rejeitada pelo executor; retorno esperado -0,048.
   O simulador de treinamento (`training_evaluation.py`) tem filtro econômico;
   portanto os dois avaliadores também divergem entre si.
   Correção indicada: compartilhar a decisão econômica em modo IA, preservando
   explicitamente o modo técnico como comparador separado.

2. **Alto — Filtro de velas ausente no dataset de avaliação.**
   `dataset_builder.py` calcula e acumula alertas, mas passa `candle_alerts=[]`
   para `strategy.get_signal`. `executor.py:698` passa os alertas reais e
   `CANDLE_PATTERN_FILTER=True`. Assim o treinamento pode avaliar contratos
   bloqueados em execução. Correção: reproduzir a mesma política temporal de
   alertas no dataset e reconstruir a avaliação com proveniência atualizada.

3. **Alto — Política econômica não versionada na compatibilidade.**
   `model_contract.specification` registra limites de aprovação e parâmetros,
   mas não registra MIN_EXPECTED_RETURN nem a semântica estrita de confiança.
   A alteração recente de margem 2% para zero muda o conjunto de contratos,
   sem invalidar aprovações anteriores. Reprodução: alterar a constante para
   0,02 mantém `compatible(metadata, symbol) == True`.
   Correção: separar versão de dados da política de avaliação/execução e
   exigir reavaliação do modelo ao mudar esta última.

4. **Médio — Cadência reinicia em eventos diferentes.**
   `executor.py:716` zera o contador após sinal BUY/SELL, mesmo se
   `_send_proposal` recusar ou se a proposta depois for rejeitada pelo payout.
   `training_evaluation.simulate` reinicia após contabilizar um contrato.
   Após duas perdas, com intervalo adaptativo de duas velas, isso pode mudar
   a próxima oportunidade avaliada. Correção: definir um evento único para
   reinício e reproduzi-lo em ambos os fluxos.

5. **Médio — Calendário de risco distinto.**
   A avaliação de treino usa UTC fixo; risco operacional usa `date.today()`.
   Em máquina com outro fuso, stops e sequências de perdas reiniciam em
   momentos diferentes. A diferença está documentada, mas permanece real.
   Correção: um fuso IANA explícito compartilhado e registrado na avaliação.

6. **Médio — Elegibilidade não exige vantagem econômica observada.**
   `training_evaluation.evaluate` exige contratos, win rate >=60% e AUC >0,5.
   Não exige expectancy positiva, amostra mínima robusta ou consistência
   walk-forward. `train_model.py` registra reprovação no payout de 70%,
   ausência de ganho sobre baseline e janelas fracas como avisos, sem veto.
   As cotações observadas também são diagnósticas. Com payout de 60%, acerto
   de 60% equivale a retorno médio por unidade de -4%, apesar do piso histórico.
   Isto é uma limitação da política, não prova de prejuízo de um modelo ativo.
   Correção: definir elegibilidade econômica coerente com o payout avaliado,
   distinguindo probabilidade estimada de resultado observado.

## Ambiguidades e bloqueios intencionais

- Confiança por entrada >55% e acerto histórico >=60% são medidas diferentes.
  A implementação anterior alterou somente a entrada. Se o pedido de 55%
  se referia ao treino, ele não foi atendido nesse aspecto.
- Payout mínimo de 60%, score combinado, concordância técnica/IA, Price Action,
  AUC, modelo compatível e risco continuam obrigatórios. Logo, apenas superar
  55% e ter expectativa não negativa não basta para abrir um contrato.
- Atualização dos arquivos não atualiza módulos de um processo Python já
  iniciado. A revisão não verificou qual versão está carregada no serviço.
- Promoção sobre champion exige reavaliação no mesmo período fora da amostra;
  ausência dessa comparação bloqueia promoção intencionalmente.

## Verificação e continuidade

Foram executadas duas reproduções controladas em cópia isolada em /tmp:
backtest versus validação de proposta e compatibilidade versus margem.
A suíte de 377 testes passou na tarefa anterior; não foi reexecutada nesta
análise. Seu sucesso não cobre as divergências identificadas. Nenhum resultado
operacional novo foi inferido a partir das reproduções sintéticas.

Próxima ação: corrigir primeiro a paridade econômica entre backtest e executor,
com regressão que reproduza o caso 56%/70%; depois alinhar alertas e versionar
política antes de reavaliar modelos. Não flexibilizar risco para ocultar
incompatibilidades.

## Correção do primeiro achado

O backtest com IA passou a reutilizar `model_contract.profitable_proposal`,
com margem acrescida dos custos hipotéticos por unidade de stake. A entrada
56%/70% é rejeitada; equilíbrio exato sem custos é aceito; confiança igual
a 55%, ausente ou inválida é rejeitada. O modo técnico continua separado.
As regressões estão em `tests/test_backtest.py`. Os demais achados desta
auditoria permanecem pendentes. Nenhum modelo ou serviço foi substituído.
