**Análise da lógica dos contratos e das perdas — 08/09/2026**

O sistema apresenta falhas concretas na preparação dos dados, no filtro de IA e na coerência entre treino e execução. Elas permitem operar sem evidência válida de vantagem estatística. O histórico disponível confirma perdas e entradas que passaram apesar da abstenção ou divergência da IA, mas não permite atribuir causalmente cada perda a um único defeito.

Análise sobre o código da revisão `508bd38` e os arquivos locais disponíveis em 08/09/2026. Nenhuma ordem, consulta autenticada, execução do bot, retreino, reinício ou substituição de modelo foi realizada. A única alteração no repositório é este relatório. Testes e reproduções foram executados em uma cópia temporária, sem `.env`, credenciais ou modelos operacionais; os modelos originais foram posteriormente carregados somente para inspeção de metadados em processo separado.

**Resultado observado**

O `operacoes_log.csv` contém 11 operações encerradas, de 04/09 a 08/09, todas com duração registrada de 15 minutos:

| Ativo | Contratos | WIN | LOSS | Resultado líquido (USD) |
| --- | ---: | ---: | ---: | ---: |
| frxEURCHF | 5 | 2 | 3 | -13,33 |
| frxUSDCAD | 2 | 2 | 0 | +17,85 |
| frxAUDUSD | 3 | 0 | 3 | -29,70 |
| frxAUDNZD | 1 | 0 | 1 | -10,00 |
| Total | 11 | 4 | 7 | -35,18 |

- Acerto observado: 36,36%.
- Stakes somados: US$ 110,08; resultado dividido pelos stakes: -31,96%. Isso não é drawdown da conta.
- Em 04/09: +US$ 4,52. Em 08/09: -US$ 39,70, com quatro perdas nas quatro operações registradas nesse dia.
- Os saldos reiniciam em US$ 1.000 entre algumas sessões. Portanto, somar os lucros é mais confiável para este recorte do que subtrair o primeiro saldo do último.
- Onze operações são insuficientes para estimar com precisão a taxa de acerto futura. Como referência descritiva, o intervalo de Wilson de 95% é aproximadamente 15,2%–64,6%, sob hipótese binomial; dependência entre operações limita essa interpretação.

**1. Crítico — o filtro de IA aceita a própria condição de “não operar”**

Evidência: `strategy.py:340–353`, `ai_predictor.py:262–302`, `config.py` (`AI_TECH_WEIGHT=0.55`, `AI_MODEL_WEIGHT=0.45`, `AI_SCORE_MIN=0.30`).

Quando a IA discorda do sinal técnico ou retorna `None`, o score é:

```text
score = 0,55 - 0,45 × (1 - confiança)
      = 0,10 + 0,45 × confiança
```

Consequências:

- Qualquer confiança a partir de aproximadamente 44,44% passa pelo corte de 0,30, mesmo sem concordância.
- Quanto maior a confiança da IA na direção contrária, maior o score que aprova a entrada técnica.
- `predict()` retorna `None` quando os modelos divergem ou quando a confiança é insuficiente. A estratégia desfaz essa abstenção ao aceitar o score resultante.
- O score técnico contínuo não entra na combinação: é substituído por `1.0`. Uma entrada que apenas ultrapassou `SIGNAL_SCORE_MIN=0.05` ganha o mesmo peso técnico de uma entrada muito mais forte.

Reprodução offline do código, isolando o filtro de Price Action:

| Sinal técnico | Resposta da IA | Score calculado | Resultado |
| --- | --- | ---: | --- |
| BUY | SELL, confiança 90% | 0,5050 | BUY autorizado |
| BUY | None, confiança 53% | 0,3385 | BUY autorizado |
| BUY | None, confiança 70% | 0,4150 | BUY autorizado |
| BUY | BUY, confiança 56% | 0,8020 | BUY autorizado |
| BUY | None, confiança 0% | 0,1000 | Bloqueado |

No histórico, sete operações têm score compatível com o ramo de divergência/abstenção: duas vitórias, cinco perdas e -US$ 33,33. Quatro operações registram confiança abaixo de 55%; todas perderam, somando -US$ 40,21. São grupos sobrepostos, não valores somáveis.

Exemplo: a SELL de frxAUDUSD encerrada às 10:01 de 08/09 tem confiança 0,6966 e score 0,4135, compatível com `0,10 + 0,45 × 0,6966`. A confiança alta não significava apoio ao SELL.

O log não guarda a direção de cada modelo; não é possível separar retrospectivamente discordância de abstenção. Bloquear essas sete entradas também bloquearia duas vitórias e alteraria a trajetória das operações seguintes. Esta contagem não é um backtest de uma correção.

Correção indicada: preservar explicitamente a abstenção, usar probabilidade da direção efetivamente comprada e definir separadamente a política para IA indisponível. Se houver combinação de scores, usar a força técnica real e calibrar o resultado; `ai_score=0.80` atualmente não representa 80% de chance de vitória.

**2. Crítico — o CSV atual impede isolamento de símbolo e de frequência temporal**

Evidência: `dataset_builder.py:109–175`, `pipeline.py:1834–1917`, `pipeline.py:199–219`, `ticks.csv` e `pipeline.log`.

O primeiro registro de `ticks.csv` é uma linha de dados de R_10, não o cabeçalho `epoch,datetime,symbol,price`. `pd.read_csv()` interpreta essa linha como nomes das colunas. Como consequência:

1. O filtro por `symbol` é pulado no snapshot e no builder.
2. `epoch` não é reconhecido, portanto a agregação temporal é pulada.
3. O fallback usa a quarta coluna como preço e concatena ativos distintos como uma única série.
4. O horizonte de duas posições passa a representar dois registros de frequência variável, frequentemente segundos, em vez de duas velas de um minuto.

O log da sessão confirma:

```text
[DATASET] Sem epochs — usando 200,000 preços diretamente.
[DATASET] Horizonte target: 2 tick(s)
[DATASET] Linhas geradas  : 97,668
```

Uma leitura posicional offline encontrou 338.380 registros válidos de 23 símbolos. Nas últimas 200.000 linhas válidas há 20 símbolos, 33 transições entre símbolos e 61 retrocessos de horário entre registros adjacentes. Há mistura de ticks e fechamentos históricos. As médias móveis do `dataset.csv.tmp` vão de aproximadamente 0,577 a 40.629,91.

A reprodução com CSV sintético sem cabeçalho, dois ativos e espaçamento de um segundo gerou 50 janelas atravessando a mudança de ativo. O builder também deixou de agregar os registros em velas nessa reprodução.

`_ensure_ticks_header()` só cria cabeçalho se o arquivo estiver vazio ou ausente; não detecta este caso. Não foi possível identificar quando ou por qual processo o cabeçalho original foi perdido.

Mesmo com cabeçalho, o builder só ordena quando o último horário é menor que o primeiro, o que não detecta desordem interna. A decisão de tratar toda a série como candles usa espaçamento médio global, inadequado para registros misturados. Acrescentar apenas o cabeçalho não resolve toda a qualidade temporal.

Correção indicada: leitor com esquema explícito e validação; recuperação em arquivo novo; isolamento por ativo; ordenação e deduplicação; distinção entre tick e OHLC; agregação única e causal. Recusar treinamento quando o esquema ou a proveniência temporal forem ambíguos. Retreinar sobre o arquivo atual perpetua o defeito.

**3. Alto — o alvo aprendido não corresponde à liquidação do contrato**

Evidência: `dataset_builder.py:214–246`, `config.py:181–192`, `ai_predictor.py:329–389`, `executor.py:815–842`.

O alvo usa o sinal da diferença entre o preço atual e a média dos próximos dois preços. Mesmo com candles corretos, isso pergunta se a média das próximas duas velas fica acima do fechamento atual. A operação compra CALL/PUT para 15 minutos.

Rise/Fall compara preço de saída e de entrada ao término do contrato, conforme a [documentação da Deriv](https://doc-deriv.readme.io/docs/risefall). Não compara a média de duas velas com o fechamento anterior ao pedido.

Um exemplo demonstra a incompatibilidade: preço atual 100, próximas duas observações 101 e 101, preço ao final de 15 minutos 99. O target do builder é BUY, mas uma CALL com entrada próxima de 100 perde nesse vencimento.

Além disso, `PRIORITIZE_DURATION=15` sobrepõe as previsões de duração. O dataset atribui 15 a todas as linhas, e a inspeção de `duration_model.pkl` encontrou somente a classe `[15]`. Isso é uma duração fixa com camadas de inferência adicionais, não uma seleção aprendida entre 5, 15 e 30 minutos.

Correção indicada: construir rótulos correspondentes ao ponto de entrada e vencimento simulados, por direção e duração, tratando empates e disponibilidade de ticks. Para duração fixa, representar essa escolha explicitamente; para seleção dinâmica, validar cada prazo em dados fora do treino.

**4. Alto — as features de Price Action não têm o mesmo significado no treino e ao vivo**

Evidência: `pipeline.py:1915`, `feature_engine.py:107–109`, `indicators.py:235–249` e `indicators.py:448–470`, `ai_predictor.py:132–154`, `executor.py:92–94`.

O pipeline usa janela de 50, efetivamente 51 preços. O builder manual usa por padrão 101 preços. `feature_engine` agrupa cada dez preços em uma vela sintética, mas Price Action exige pelo menos 15 dessas velas:

| Entrada | Velas sintéticas completas | Price Action disponível |
| --- | ---: | --- |
| 51 preços, treino pelo pipeline | 5 | Não |
| 101 preços, builder padrão e subjanela TFT | 10 | Não |
| 150 preços | 15 | Sim |
| 500 preços, buffer clássico ao vivo | 50 | Sim |

Quando indisponível, todas as 11 features são substituídas por zero. A leitura integral confirmou zero em todas elas nas 153.928 linhas de `dataset.csv` e nas 97.668 linhas de `dataset.csv.tmp`.

Já o modelo clássico pode receber features não nulas ao vivo. O TFT reconstrói janelas de 101 preços e continua recebendo zeros nessas colunas. Usar a mesma função não garante equivalência quando a janela e a unidade dos dados diferem.

As EMAs, RSI e ADX também usam inicialização e histórico diferentes entre janelas de 51, 101 e até 500 observações. No estado atual, soma-se a divergência entre ticks de treino e candles de inferência.

Correção indicada: contrato único de dados para treino e inferência, com mesma janela, unidade, OHLC, ordem e tratamento de aquecimento. Testar igualdade numérica das features para o mesmo instante histórico, não somente igualdade dos nomes.

**5. Alto — modelos antigos são reutilizados sem comprovar adequação ao ativo escolhido**

Evidência: `pipeline.py:2618–2649`, `train_model.py:407–412`, `ai_predictor.py:65–107`, `model_metrics.json` e metadados dos `.pkl`.

Os três arquivos de modelo têm última modificação em 06/06/2026. A inspeção identificou XGBoost como modelo clássico, 27 features e ausência de metadados de símbolo, período de treino ou horizonte no payload clássico. O log da sessão de frxAUDNZD confirma o carregamento de XGBoost e TFT e a decisão de ignorar o treino inicial porque `model.pkl` existe.

O scaler do modelo clássico contém médias de EMA9 e EMA21 de aproximadamente 22.356,72 e 22.763,55, enquanto os preços das operações forex analisadas estão próximos de 1. Isso evidencia uma diferença substancial de escala na entrada do modelo. Não identifica sozinho o conjunto de treino original nem mede quanto dessa diferença causou as perdas.

O seletor de ativos pode trocar o instrumento entre sessões, mas a existência de um arquivo basta para reutilizá-lo. Não há validação de que o modelo corresponde ao ativo selecionado.

As seis métricas disponíveis são `stage="classical"`, com `model_path="model.pkl.new"`. As últimas duas têm AUC 0,5064 e 0,5029, e acurácia 50,32% e 50,31%. São métricas preliminares dos candidatos; não são prova de desempenho do XGBoost em uso. O log consultado termina durante o treino TFT, após sua primeira época, e não registra promoção concluída nessa sessão. Não se pode afirmar que o treino falhou apenas por isso.

O carregamento offline também emitiu avisos de versão: objetos sklearn serializados com 1.8.0 foram abertos com 1.9.0, além de aviso de serialização antiga do XGBoost. Os payloads carregaram; esses avisos são uma limitação adicional, não uma explicação comprovada para as perdas.

Correção indicada: versionar modelo, ativo, timeframe, horizonte, esquema de features, versões de dependências e intervalo de treino. Exigir compatibilidade antes de operar e registrar a versão efetivamente usada em cada contrato.

**6. Alto — a compra não verifica se a remuneração justifica o risco**

Evidência: `executor.py:867–882`. O handler usa o ID da proposta e envia a compra sem analisar `ask_price` e `payout`, campos documentados na [resposta de proposta da Deriv](https://developers.deriv.com/comparison/proposal/).

Para stake S, payout total P, probabilidade de vitória p e perda integral na derrota, o valor esperado antes de custos adicionais é `p × P - S`. O ponto de equilíbrio é `p = S/P`, equivalente a `1/(1+r)` para lucro líquido r por unidade apostada.

Nas quatro vitórias registradas, o lucro líquido variou de 78,60% a 89,48% do stake, com média simples de 86,45%. Para esse retorno médio, o ponto de equilíbrio ilustrativo é 53,63% de acerto. Esse valor não é o payout médio de todas as propostas, pois os payouts das derrotas não foram registrados.

Em reprodução offline, uma proposta com preço 10 e payout total 10,01 foi comprada normalmente. O sistema não dispõe de filtro econômico nem de probabilidade calibrada para o contrato de 15 minutos. AUC próxima de 0,50 dos candidatos também não demonstra vantagem econômica.

Correção indicada: validar valores finitos e positivos da proposta; registrar preço e payout; avaliar a probabilidade da direção e vencimento específicos contra o retorno oferecido. Scores arbitrários ou acurácia de outro horizonte não substituem essa probabilidade.

**7. Alto — sequência temporal e validação dos modelos permitem avaliações enganosas**

Evidência: `dataset_builder.py:241–245`, `train_model.py:98–115`, `train_model.py:147–158`, `train_model.py:303–316`, `train_model.py:355–359`, `pipeline.py:1676–1829`.

- Linhas com movimento futuro pequeno são removidas pelo builder. Isso condiciona treino e teste a uma informação futura indisponível na entrada. As métricas não representam automaticamente todos os contratos que o bot poderá comprar.
- O TFT cria sequências depois dessa remoção: 50 linhas podem representar intervalos variáveis. Na inferência, as 50 posições são consecutivas. No CSV atual, as sequências também podem atravessar ativos.
- A sequência de treino `X[i:i+50]` termina em `i+49`, mas recebe o target de `i+50`. O clássico usa a feature do próprio instante do target. Há deslocamento adicional de uma linha e incompatibilidade entre os horizontes combinados.
- O stacking usa `StratifiedKFold(shuffle=False)`, que não é uma divisão progressiva no tempo. O treino interno pode usar períodos posteriores ao fold validado. O teste externo tem divisão temporal e gap, um ponto positivo, mas o procedimento interno não reproduz disponibilidade temporal.
- O melhor modelo é escolhido pelo mesmo conjunto chamado de teste. Falta um período final intocado para medir a seleção completa.
- `_load_latest_final_metrics()` escolhe a última métrica final global, inclusive de candidato rejeitado; não identifica necessariamente o champion ativo. As métricas comparadas podem vir de ativos e períodos diferentes.
- Sem métrica anterior, `_should_promote_model()` aceita o candidato sem piso absoluto de qualidade. O gate considera apenas métricas clássicas, embora a promoção também possa substituir TFT e duração.

Correção indicada: manter timestamps e símbolo no dataset, preservar a malha temporal das sequências, separar seleção de avaliação final, usar validação temporal progressiva e comparar champion e challenger no mesmo período e objetivo. Avaliar o ensemble e a estratégia efetivamente executados.

**8. Médio/alto — Price Action, seletor e backtest representam mercados diferentes**

Evidência: `executor.py:1071–1110`, `strategy.py:321–334`, `backtest.py:599–620`, `pipeline.py:314–342`, `pipeline.py:405–494`.

O executor calcula OHLC da vela corrente, mas guarda somente seu fechamento no histórico. Para detectar padrões, reagrupa esses fechamentos em blocos de `max(4, len(closes)//20)`. Com 100 fechamentos são blocos de cinco minutos; com 500, de 25 minutos. As máximas e mínimas intraminuto foram descartadas. Esses padrões não são os padrões das velas reais de um minuto, e a escala muda durante o aquecimento.

O filtro PA da estratégia usa blocos fixos de dez fechamentos. O backtest usa OHLC dos candles de tempo para os alertas. Assim, os filtros de padrões ao vivo e no backtest não são equivalentes.

O scan escolhe ativos usando ticks, enquanto a estratégia decide com fechamentos de minuto. Seu bônus de EMA verifica apenas `ema9 != ema21`, não o alinhamento com MACD. O score mede propriedades do movimento recente, mas não demonstra chance de vitória no vencimento de 15 minutos nem considera payout.

O ADX é explicitamente uma aproximação baseada em fechamentos, não em máximas/mínimas reais. Seu histórico ao vivo só é atualizado quando chega à avaliação de sinal, omitindo períodos de contrato ativo e de pausa. O percentil adaptativo não representa uma sequência contínua de velas.

Correção indicada: reter OHLC temporal, compartilhar a construção das entradas entre backtest e executor e selecionar ativos com avaliação compatível com o horizonte operacional. Não executar um backtest sobre o arquivo atual como se ele reproduzisse fielmente os contratos registrados.

**9. Alto operacional — o ciclo de proposta pode comprar repetidamente ou perder a rastreabilidade**

Evidência: `executor.py:662–715`, `executor.py:809–940`, `executor.py:944–982`.

- `_send_proposal()` marca o horário, mas não marca `_in_trade=True`. O timeout de proposta exige `_in_trade`, portanto não cobre a espera inicial sem resposta. Em outra vela, uma nova proposta pode ser enviada.
- `_handle_proposal()` compra qualquer ID recebido, sem correlacionar requisição, estado, símbolo ou idade da proposta. A reprodução chamou o handler duas vezes com a mesma proposta e houve duas chamadas de compra.
- Uma resposta tardia pode usar direção, stake e indicadores já sobrescritos por outra solicitação.
- O timeout de contrato elimina o ID local sem confirmar a liquidação. Uma atualização posterior desse ID será ignorada.
- A instância nova começa sem contrato aberto e não restaura integralmente o contrato persistido. O dashboard também limpa `active_contract` no início da sessão.
- `entry_price` é o último tick local quando chega a resposta de compra, não o preço de entrada confirmado do contrato. O CSV não registra nenhum desses preços.
- Um contrato encerrado sem campo `profit` vira zero pelo fallback e é classificado como WIN. No caminho normal, o lucro retornado pela API é usado e IDs diferentes são ignorados; não foi encontrada inversão BUY/CALL ou SELL/PUT.

São riscos reproduzidos ou identificados no código, mas não há evidência no CSV disponível de que compras duplicadas ou timeouts tenham causado as sete perdas. O CSV não guarda ID do contrato para reconciliar isso.

Correção indicada: máquina de estados explícita, correlação por requisição, compra idempotente, preservação do ID até reconciliação e validação do resultado recebido. Registrar preços e horários oficiais de entrada/saída e IDs.

**10. Alto para controle e diagnóstico — reinícios apagam risco e o drawdown é inconsistente**

Evidência: `dashboard/server.py:629–660`, `risk_manager.py:32–54`, `risk_manager.py:178`, `risk_manager.py:271–293`, `executor.py:409–421`, `pipeline.py:2680–2697`.

Iniciar pelo dashboard reescreve `risk_state.json`, zerando perdas consecutivas, pausa, PnL e resultados recentes, além de definir o saldo pelo formulário. Assim, um stop chamado de diário pode ser contornado por reinício no mesmo dia. O histórico apresenta reinícios de saldo e contadores compatíveis com esse comportamento; não permite reconstruir todos os eventos de conta.

`RiskManager` define `_initial_balance` antes de carregar o saldo salvo. No fluxo Bearer, o pipeline pode receber o saldo da conta, mas a restauração o substitui pelo saldo do formulário previamente salvo. No fluxo legado, a autorização atualiza saldo e base diária, mas não a base usada no drawdown.

O campo `drawdown_pct` calcula retorno sobre `_initial_balance`, não queda desde um pico de saldo. Os valores próximos de -88% registrados coexistem com sessões partindo de US$ 1.000 e perdas de dezenas de dólares. Esses valores não comprovam perda real de 88% neste recorte.

As proteções existentes são stake de 1%, pausa após três perdas, stop diário de 25% e alerta de drift após 20 operações. Elas limitam exposição em condições específicas, mas não corrigem valor esperado negativo. O drift só alerta; neste recorte de 11 operações, não se completa a janela. O retreino por taxa de acerto também exige 20, embora o gatilho de novos ticks possa disparar antes.

Correção indicada: separar saldo autenticado, saldo simulado e estado persistido; preservar risco diário através de reinícios; reconciliar contratos; calcular drawdown por pico de equity. Manter identidade da conta sem registrar credenciais. No fluxo legado, `_handle_authorize()` também não valida `is_virtual` contra o modo local; isso é uma lacuna de isolamento, não causa direcional das perdas.

**Verificações executadas e limites**

Foram lidos os módulos de configuração, indicadores, estratégia, features, coleta, dataset, treino, Transformer, inferência, execução, sessão, risco, orquestração, backtest e os caminhos de controle/diagnóstico pertinentes do dashboard. Foram examinados os testes correspondentes, sem iniciar os componentes operacionais.

Em `/tmp/standeriv-contract-audit-ei1516ne`, com cópia de código e testes, token fictício e sem `.env`, foi executado:

```bash
DERIV_TOKEN=audit_offline DERIV_APP_ID=1089 PYTHONDONTWRITEBYTECODE=1 \
  /home/suporte-01/Repositorios/StanDeriv/.venv/bin/python -m pytest \
  tests/test_config_thresholds.py tests/test_indicators.py \
  tests/test_strategy.py tests/test_strategy_edge_cases.py \
  tests/test_executor_trade_flow.py tests/test_risk_manager.py \
  tests/test_backtest.py tests/test_pipeline_integration.py -q
```

Resultado: **143 testes passaram em 3,68 segundos**. O ambiente emitiu mensagens `Failed to create stream fd: Operation not permitted`, sem falha dos testes.

Além deles, scripts offline reproduziram o filtro de IA invertido, as janelas PA insuficientes, o CSV sem cabeçalho atravessando símbolos, a compra sem corte de payout, a repetição de compra de uma proposta e a duplicação de histórico CSV/API. Esta última produziu 99 timestamps duplicados e um retrocesso temporal ao acrescentar 99 candles sobrepostos ao buffer; no arquivo atual sem cabeçalho, a pré-carga CSV não reconhece o símbolo, portanto esse defeito é latente e pode aparecer após reparar o leitor.

Passar nos testes não demonstra rentabilidade nem ausência dessas falhas. Por exemplo, helpers de `tests/test_strategy.py` alteram `config.USE_AI_MODEL` e `config.USE_WEIGHTED_SIGNAL`, embora `strategy.py` tenha importado seus valores diretamente; essa alteração não controla o ramo pretendido. Alguns testes de limite apenas verificam se a resposta pertence a BUY/SELL/None, sem validar o bloqueio descrito.

Não foi executado backtest de desempenho: faltam no histórico operacional preços oficiais, payouts de todas as propostas, versões de modelo e direção dos modelos. Os dados misturam ticks e closes e há divergência entre o filtro de velas simulado e o executado. Simular agora e apresentar isso como explicação dos 11 contratos daria precisão indevida. As métricas de junho/setembro também não permitem estabelecer a proveniência completa do treino dos modelos ativos.

**Ordem indicada para correção e validação**

1. Corrigir o filtro de abstenção/divergência e a correlação/idempotência de ordens, com regressões offline.
2. Recuperar e validar os dados em novos arquivos, preservando os originais; unificar símbolo, timestamps, OHLC e janelas.
3. Alinhar rótulo e duração à liquidação real e reconstruir features/sequências sem seleção temporal indevida.
4. Versionar os modelos e avaliar candidatos no mesmo período, com probabilidades calibradas e payout. Retreino só depois de resolver o contrato de dados.
5. Unificar backtest e executor; preservar risco entre sessões e corrigir saldos/drawdown.
6. Validar fora da amostra e depois em conta de demonstração efetivamente identificada, mediante autorização operacional, com rastreabilidade por contrato. Esta etapa não foi executada nesta análise.

A evidência mais forte é a combinação de um filtro de IA que aprova abstenções com treinamento sobre dados de símbolos/frequências misturados e alvo incompatível com a expiração. Corrigir essas falhas torna a avaliação válida; não garante que a estratégia passará a ser lucrativa.
