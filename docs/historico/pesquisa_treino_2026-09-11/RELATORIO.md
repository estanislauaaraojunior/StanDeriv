# Treinamento e qualidade das entradas no StanDeriv

## Diagnóstico e recomendação

O problema imediato é a ausência de um modelo compatível ativo para o símbolo da sessão observada. O problema estatístico é a baixa capacidade de separar entradas favoráveis das demais, combinada com probabilidades insuficientes para os filtros de execução. Repetir o treinamento por mais horas não assegura que esses dois problemas desapareçam.

A recomendação é preservar o treinamento temporal existente e testar, em ambiente offline, uma combinação de **baselines simples, calibração comparada, seleção por qualidade probabilística e avaliação das entradas condicionada ao payout**. O primeiro experimento deve comparar as probabilidades antes e depois da calibração atual. A troca direta por TFT, LSTM, reinforcement learning ou outro modelo complexo não tem justificativa demonstrada nos dados disponíveis.

Esta recomendação procura aumentar a precisão das entradas sustentada por dados futuros. Não há evidência suficiente para afirmar antecipadamente que qualquer método alcançará uma taxa de acerto específica. A aprovação de um candidato, a promoção de seu artefato e a autorização de uma entrada são decisões diferentes.

## Evidência local

O estado foi examinado em 11 de setembro de 2026, na branch `main`, com HEAD `8ae6cf1` e alterações locais em andamento. Os números abaixo foram extraídos dos relatórios existentes, sem treinar modelos, executar contratos ou recalcular backtests. A captura estruturada em [evidencias.json](evidencias.json) registra horário UTC, hashes dos relatórios, hashes do código consultado e campos selecionados. A tabela completa está em [treinos.csv](treinos.csv).

Foram encontrados 16 relatórios diretamente em `training_reports/`: 15 com `eligible=false` e um com `eligible=true`. Esse inventário não representa necessariamente todos os treinamentos já realizados: outros resultados podem ter sido arquivados, e um relatório é identificado pelo hash do dataset. Também não permite somar os contratos entre treinamentos, porque as janelas podem se sobrepor.

| Treino registrado, UTC | Símbolo | Modelo | AUC | Contratos simulados | Acertos | Elegível |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 11/09, 03:12 | 1HZ25V | RandomForest | 0,5089 | 19 | 6 — 31,6% | Não |
| 11/09, 05:12 | 1HZ25V | RandomForest | 0,5101 | 37 | 14 — 37,8% | Não |
| 11/09, 07:12 | 1HZ25V | RandomForest | 0,5211 | 33 | 14 — 42,4% | Não |
| 11/09, 09:13 | 1HZ25V | RandomForest | 0,4959 | 0 | Não aplicável | Não |
| 11/09, 10:55 | 1HZ25V | RandomForest | 0,5187 | 29 | 18 — 62,1% | Sim |
| 11/09, 11:05 | frxUSDCAD | XGBoost | 0,4392 | 5 | 2 — 40,0% | Não |
| 11/09, 11:11 | frxUSDCAD | RandomForest | 0,5211 | 0 | Não aplicável | Não |

Na captura, `state.json` apontava `frxUSDCAD`; `model.pkl` e seu sidecar não existiam no caminho padrão. Existia `model.pkl.candidate`, com metadados de `frxUSDCAD`, `approved=false` e treinamento registrado às 11:11 UTC. O trecho recente de `pipeline.log` informava espera pelo primeiro modelo compatível e ausência de promoção do challenger por não haver contratos no teste. Isso sustenta o diagnóstico da sessão registrada, sem afirmar que os arquivos descrevem integralmente a memória de qualquer processo ativo.

O log também registra um novo ciclo de coleta e treinamento, com arquivamento de oito arquivos antigos. Portanto, um relatório anterior elegível para `1HZ25V` não demonstra a existência de um champion carregado para `frxUSDCAD`. O motivo exato e toda a trajetória da promoção do candidato anterior não foram reconstruídos; não se deve classificá-los como falha de promoção apenas pela presença do relatório.

### O último candidato de frxUSDCAD

O candidato de 11:11 UTC usou 2.856 linhas de ajuste e 1.189 linhas de teste. No teste, havia 660 rótulos de alta, 506 de queda e 23 empates. As 1.189 classificações finais foram de alta. A acurácia de 55,51% é exatamente a fração de altas nessa amostra, ou seja, a mesma acurácia que uma regra retrospectiva de prever alta em todas as linhas produziria. Isso não é um benchmark treinado prospectivamente, mas expõe a limitação da acurácia isolada.

A confiança média foi 52,69%, e o componente `ai_only` registrou 1.189 bloqueios por `low_confidence`. Logo, nenhuma previsão passou pelo piso de 55% usado nessa avaliação. A AUC de 0,5211 indica alguma ordenação dos scores nesse período, mas não demonstra que existam entradas suficientes acima do limiar econômico.

Na estratégia combinada, o primeiro bloqueio observado por linha foi: ausência de sinal técnico em 887 casos, divergência de direção em 105 e baixa confiança em 197. Esses contadores são sequenciais: uma linha pode falhar em vários critérios, mas apenas o primeiro é registrado. Não é correto concluir que remover o bloqueio mais frequente produziria o mesmo número de contratos.

A avaliação com cotações observadas também ficou sem operações. No componente de IA, todas as 1.189 linhas foram bloqueadas por ausência de payout histórico correspondente. Essa diferença revela outro problema de cobertura dos dados, além da confiança, e não autoriza preencher o histórico com cotações futuras.

### O candidato elegível de 1HZ25V

O relatório das 10:55 UTC contém 18 acertos em 29 contratos, com payout líquido hipotético de 85%. O limite inferior de Wilson registrado é 44,00%. O próprio relatório alerta sobre pouca amostra ou falta de consistência nas janelas walk-forward. Essa elegibilidade obedece à política atual do código; não equivale à confirmação estatística de uma vantagem.

Com o mesmo modelo, a estratégia combinada simulou zero contratos no cenário fixo de 70%. Com cotações observadas, produziu 17 contratos e 10 acertos, ou 58,82%, ficando inelegível pelo piso de 60%. As três avaliações selecionam conjuntos diferentes de entradas: não são simplesmente três multiplicações do mesmo histórico de lucros. Esse resultado torna o alinhamento entre seleção, aprovação e payout uma prioridade concreta.

## O treinamento que já existe

`train_model.py` já compara RandomForest e XGBoost em três janelas walk-forward. As janelas separam ajuste, calibração e avaliação. Há um teste final reservado e exclusão de observações cujos vencimentos alcançam a janela de features da próxima partição. Não há motivo para apresentar walk-forward como uma funcionalidade ausente.

O RandomForest usa 300 árvores, profundidade máxima 10 e mínimo de 10 amostras por folha. O XGBoost usa 300 estimadores, profundidade 5 e taxa de aprendizado 0,05. Esses são dois pontos fixos de configuração; o fluxo não realiza uma busca sistemática de regularização.

A calibração atual ajusta `LogisticRegression` sobre o vetor de probabilidades do estimador. Essa transformação multinomial pode mudar decisões e comprimir ou expandir scores. Não é idêntica a todas as variantes de Platt scaling, nem está demonstrado que seja a causa da baixa confiança. É necessário comparar o estimador bruto e o calibrado nas mesmas linhas futuras.

O ranking de candidatos começa pela existência de 30 contratos walk-forward e segue por pior expectancy, média de expectancy, drawdown, log loss e AUC. Uma janela sem contratos recebe expectancy zero. Assim, na ordenação, um candidato que se abstém pode superar outro que opera com resultado negativo; isso é coerente como proteção de capital, mas não informa sozinho qual estimador tem maior capacidade de previsão.

Após escolher a família, o modelo final é reajustado na partição inicial de aproximadamente 50% dos dados, calibrado na seguinte e avaliado nos 20% finais. A partição denominada `validation` é criada, mas não é usada diretamente nesse ajuste final. O development usado no walk-forward inclui informações temporais mais recentes que as usadas pelo estimador final. Isso merece uma comparação controlada entre receitas de ajuste final; não constitui, por si só, um erro de lookahead.

Outro ponto de consistência: `model_contract.specification()` registra o calendário local em `risk_timezone`, enquanto `HistoricalRiskManager` passou a usar UTC por padrão. `training_evaluation.simulate()` instancia esse risco sem informar outro fuso. Antes de comparar novos resultados, a política de fuso efetivamente simulada deve coincidir com a registrada nos metadados.

## Assertividade, cobertura e payout

Para este projeto, convém separar quatro medidas: acurácia sobre todas as linhas rotuladas; precisão nos contratos selecionados; cobertura, ou fração de oportunidades em que o sistema aceita operar; e retorno esperado por unidade de stake. Um modelo pode apresentar boa acurácia com pouca utilidade econômica, ou alta precisão baseada em uma amostra muito pequena.

A literatura de classificação seletiva formaliza o compromisso entre erro nas previsões aceitas e cobertura. Sua aplicação aqui é medir a precisão à medida que o sistema se abstém mais, sem transferir garantias teóricas obtidas em outros ambientes para séries financeiras dependentes.[^1]

Para payout líquido `r`, probabilidade de vitória `p` e custo proporcional `c`, com perda integral do stake quando não há vitória, a conta é:

```text
EV por unidade de stake = p × r − (1 − p) − c
                       = p × (1 + r) − 1 − c

Probabilidade de equilíbrio = (1 + c) / (1 + r)
Probabilidade para margem m = (1 + c + m) / (1 + r)
```

Estas são identidades aritméticas sob essas hipóteses, não uma estimativa da probabilidade real. O contrato atual trata empate como derrota para CALL e PUT; sua probabilidade deve continuar fora da probabilidade de vitória de ambas as direções. A margem configurada em `model_contract.py` é 0,02.

| Payout líquido hipotético | Equilíbrio, sem custos | Probabilidade para margem de 2% | EV com 60% de acertos, sem custos |
| --- | ---: | ---: | ---: |
| 60% | 62,50% | 63,75% | −4,0% do stake |
| 70% | 58,82% | 60,00% | +2,0% do stake |
| 85% | 54,05% | 55,14% | +11,0% do stake |

Portanto, o piso de aprovação de 60% de acerto não garante resultado positivo quando a cotação oferece apenas 60% de payout líquido. A aprovação estatística e o filtro econômico da proposta precisam continuar distintos. A documentação da Deriv descreve `proposal` como a obtenção do preço de um contrato específico; o preço de compra e o payout devem ser lidos e validados, e não deduzidos de uma taxa universal.[^2]

A proposta de experimento é manter o piso operacional e comparar curvas de precisão, cobertura e EV para limiares predefinidos. A documentação do scikit-learn também separa estimar probabilidades de escolher uma ação e alerta contra ajustar o limiar nos mesmos dados usados para treinar o modelo.[^3] No StanDeriv, a escolha precisa ocorrer no development temporal, com teste futuro reservado.

## Métodos de treinamento recomendados

### 1. Baselines e regularização

O primeiro baseline probabilístico deve usar apenas as frequências de classe do período de ajuste. O segundo deve ser uma regressão logística regularizada, com normalização ajustada exclusivamente no treino. Ambos usam os mesmos alvos, períodos, payouts e simulador dos modelos de árvores. A regra de não operar serve como referência econômica de retorno zero, embora não tenha precisão definida.

Esses baselines respondem a uma pergunta que os resultados atuais não resolvem: o sistema está aprendendo algo além da prevalência de alta e queda? O limiar deve ser avaliado sobre probabilidades calibradas; a comparação de ranking também precisa incluir métricas por direção, porque a AUC atual é de alta contra o restante.

O estudo de Grinsztajn, Oyallon e Varoquaux encontrou forte desempenho de árvores em benchmarks tabulares, inclusive diante de redes neurais. Isso sustenta manter árvores como concorrentes sérios, mas não comprova eficácia no vencimento de 15 minutos da Deriv.[^4] A experiência local também não mostrou ganho consistente das oito features experimentais; expandir o contrato operacional de 27 features agora seria prematuro.

Para árvores, a recomendação é uma busca pequena e previamente registrada. No XGBoost, profundidade, peso mínimo por folha, amostragem de linhas/colunas e penalizações controlam a complexidade.[^5] Como espaço exploratório, testar profundidades 2–5, `min_child_weight` em 5–30, `subsample` e `colsample_bytree` em 0,7–1,0 e `reg_lambda` em 1–20. Esses intervalos são uma proposta de engenharia, não parâmetros validados.

No RandomForest, comparar profundidades menores e folhas maiores contra a configuração existente, preservando seeds e dados. Uma busca aleatória limitada é um baseline eficiente de otimização; Bergstra e Bengio mostraram vantagens de eficiência frente a grades em seus experimentos.[^6] Sugere-se um teto inicial de 12 configurações de árvores, compartilhando as mesmas divisões, em vez de centenas de tentativas até surgir uma aprovação.

### 2. Calibração comparada

O experimento prioritário deve conservar o modelo base e comparar: probabilidades brutas como controle diagnóstico, calibração multinomial atual e temperature scaling. Quando houver amostra adequada e uma formulação explícita para as três classes, incluir sigmoid por classe; isotonic fica para uma fase com mais suporte amostral.

A documentação do scikit-learn recomenda calibrar em dados separados e distingue qualidade probabilística de calibração pura: Brier e log loss também refletem discriminação. Curvas de confiabilidade e quantidade de exemplos por faixa são necessárias. Isotonic é mais flexível e mais suscetível a overfitting em amostras pequenas; mil linhas temporalmente sobrepostas não equivalem automaticamente ao suporte independente sugerido em exemplos tabulares.[^7]

Guo e colaboradores estudaram temperature scaling como uma transformação de um parâmetro em classificação de imagens e documentos.[^8] A utilidade para StanDeriv é uma hipótese testável: reduzir graus de liberdade do calibrador e conservar a classe de maior score. Isso não cria separação onde o estimador não tem sinal. Uma distribuição próxima de 50% pode ser a resposta adequada a um problema pouco previsível.

Kull e colaboradores também mostram que certas transformações logísticas podem piorar probabilidades já adequadas e propõem beta calibration.[^9] O trabalho é sobre classificação binária, com resultados destacados em Naive Bayes e AdaBoost; não justifica substituir automaticamente o calibrador multinomial do StanDeriv. Sua principal contribuição à decisão é exigir o controle sem calibração e testar alternativas, em vez de assumir melhora obrigatória.

O relatório do experimento deve registrar, por janela, percentis dos scores brutos e calibrados, frequência de classes previstas, AUC por direção, log loss, Brier, curvas de confiabilidade e cobertura acima do limiar econômico. Se apenas a confiança aumentar, sem melhora da frequência observada de vitórias ou da qualidade probabilística, o experimento falhou.

### 3. Janelas temporais e receita de ajuste final

Preservar a separação cronológica e os embargos atuais no baseline. `TimeSeriesSplit` oferece divisões ordenadas e um intervalo `gap`, mas o gap por quantidade de linhas não resolve sozinho vencimentos variáveis e lacunas.[^10] As colunas `feature_start_epoch`, `entry_epoch` e `expiry_epoch` do projeto permitem verificar diretamente as relações temporais.

Comparar expanding window e rolling window com duas extensões de histórico predefinidas, somente depois de avaliar a calibração. A janela móvel pode reduzir o peso de regimes antigos, mas também descarta exemplos e aumenta variância. A escolha deve considerar dias cobertos, número de decisões não sobrepostas e estabilidade por direção, sem prometer que “mais recente” seja sempre melhor.

Na receita final, testar um ajuste que aproveite uma fatia mais recente do development, seguido por calibração posterior separada. A mesma receita deve aparecer nas janelas internas, para que o procedimento selecionado seja aquele efetivamente avaliado no teste externo. O holdout não deve ser incorporado ao modelo que ainda está sendo avaliado sobre ele.

O dataset operacional lido durante esta pesquisa tinha 2.017 linhas e não correspondia ao conjunto de 1.189 linhas de teste do candidato mais recente. Não se deve recalcular o desempenho daquele candidato usando o arquivo atual como se fosse o snapshot original. Cada experimento deve receber um snapshot imutável e conferir seu hash contra os metadados antes de começar.

### 4. Seleção das entradas e contexto

Os resultados de IA isolada, técnica isolada e combinação variam bastante. Em `frxUSDCAD`, o candidato XGBoost das 11:05 teve 12/20 na IA isolada, 7/11 na técnica e apenas 2/5 na combinação. As amostras são pequenas e os trajetos de risco diferentes; não comprovam superioridade de um componente, mas justificam avaliar como a interseção dos filtros seleciona casos.

Um experimento posterior pode aprender a probabilidade de sucesso de um sinal técnico existente, em vez de tentar prever a direção de todos os candles. Isso adapta a decisão ao conjunto de oportunidades em que o sistema realmente considera operar. O alvo passa a ser “o sinal proposto venceu no vencimento?”, com sinais e features gerados apenas com o passado. Se houver um primeiro modelo aprendido, suas previsões usadas no segundo estágio precisam ser temporalmente fora da amostra.

Esse segundo estágio exige muito mais sinais rotulados do que algumas dezenas e não deve substituir os filtros no processo ativo. É uma hipótese de modelagem motivada pela seleção atual, não uma melhoria demonstrada pela literatura para esse ativo. O baseline simples continua obrigatório.

Para features, priorizar ablações por famílias: tendência, volatilidade, momentum e Price Action. A documentação do scikit-learn alerta que variáveis correlacionadas podem mascarar importâncias individuais.[^11] Por isso, começar removendo uma família de cada vez em modelos reajustados no development; permutações temporais precisam de cuidado adicional para não fabricar sequências irreais. Horário UTC pode ser avaliado em Forex, mas não deve ser considerado preditivo em índices sintéticos sem evidência própria.

## Modelos modernos e limites do ativo

TabPFN merece uma avaliação posterior como concorrente, após os controles simples. O relatório técnico TabPFN-2.5, revisado em fevereiro de 2026, descreve avanços em benchmarks tabulares e maior capacidade de dados.[^12] É uma fonte dos próprios desenvolvedores, não uma validação independente de contratos Deriv. Seu desempenho frente a XGBoost em datasets tabulares não é uma taxa de acerto em operações financeiras.

Antes de qualquer experimento com esse modelo, verificar requisitos de hardware, versão, licença e compatibilidade da inferência local. Nenhum dado operacional precisa ser enviado a um serviço externo para esta pesquisa. A documentação e os estudos consultados não permitem concluir que TabPFN, TFT ou LSTM resolverão o problema observado.

Há uma ressalva específica para `1HZ25V`: a página oficial da Deriv afirma que índices sintéticos, exceto Range Break, podem não ser adequados a indicadores técnicos e atribui padrões históricos perceptíveis à coincidência, citando a ausência de livro de ofertas.[^13] Em contraste, um livro de Vince Stanzione distribuído pela Deriv apresenta uso de padrões gráficos em sintéticos, mas esclarece que as opiniões do autor não representam necessariamente a empresa.[^14]

Essa divergência deve ser explicitada, não resolvida escolhendo o material mais otimista. A descrição do produto impõe cautela à hipótese de previsibilidade; o material educacional não fornece validação fora da amostra para o contrato do projeto. Também não constitui prova matemática de imprevisibilidade de toda a família. A recomendação é manter Forex e cada índice sintético como experimentos separados e exigir evidência específica antes de investir em mais complexidade.

## Protocolo de validação

A pesquisa deve partir de uma pergunta registrada: “a alternativa melhora probabilidades e precisão das entradas em períodos futuros, para o mesmo símbolo e contrato, sem aumentar risco?”. Fixar dataset, definição do alvo, histórico, payout, fuso, seeds, custos e orçamento de busca antes de comparar variantes. A duração operacional de 15 minutos permanece a referência; testar outro vencimento seria um experimento separado.

| Etapa | Dados permitidos | Decisão |
| --- | --- | --- |
| Auditoria | Snapshots identificados e metadados | Verificar cronologia, cobertura, classes, features e payouts |
| Ajuste | Início da janela interna | Estimar parâmetros e transformações |
| Calibração | Bloco seguinte, com separação temporal | Ajustar probabilidades |
| Seleção | Blocos futuros internos | Escolher modelo, calibrador e regra de decisão |
| Avaliação externa | Bloco ainda não usado na escolha | Medir generalização da receita congelada |
| Observação prospectiva | Dados que chegam após congelamento | Confirmar cobertura e comportamento, sem compras |

Treinamentos sucessivos com holdouts sobrepostos não produzem testes independentes. Bailey e colaboradores analisam como múltiplas tentativas de estratégias sobre o mesmo histórico criam backtest overfitting.[^15] No projeto, a regra prática proposta é registrar todas as variantes e não continuar ajustando parâmetros após olhar o teste final; uma nova rodada deve usar um novo bloco externo ou assumir explicitamente caráter exploratório.

Para incerteza, apresentar intervalos e quantidade de contratos, além dos percentuais. Métodos de bootstrap em blocos preservam segmentos temporais e são disponibilizados por bibliotecas estatísticas como `arch`.[^16] O tamanho dos blocos e a estabilidade do processo precisam ser examinados; bootstrap não torna uma amostra minúscula suficiente. Contratos sem sobreposição também podem permanecer dependentes por regime e horário.

Como exemplo aritmético ilustrativo, sob hipótese simplificadora de ensaios independentes, 60 vitórias em 100 operações têm limite inferior de Wilson de aproximadamente 50,2%; 240 em 400, aproximadamente 55,1%. A diferença evidencia por que a mesma taxa de 60% pode representar graus de incerteza muito distintos. A independência não foi comprovada nas séries do StanDeriv, e esses números não são uma proposta automática de novo bloqueio.

Os resultados mínimos a publicar são: log loss e Brier contra baseline de frequências, AUC por direção, precisão e cobertura por faixa de confiança, contratos/dia, expectancy com payout declarado, drawdown, pior janela, distribuição de classes, motivos de bloqueio e cobertura das cotações. Janelas sem contratos devem aparecer como sem evidência de desempenho nas entradas; não como acerto zero nem estabilidade demonstrada.

## Sequência de implementação proposta

| Prioridade | Mudança proposta | Evidência de conclusão |
| --- | --- | --- |
| P0 | Identificar candidato, champion, símbolo e motivo de não promoção; alinhar fuso simulado e metadados | Um registro explica todo o caminho treino → promoção → disponibilidade para inferência |
| P1 | Diagnóstico de probabilidades brutas e calibradas e baseline de frequências/logística | Relatório temporal reproduzível, com hashes, sem publicação de modelo |
| P2 | Comparação pequena de calibradores, mantendo estimadores e períodos | Melhora verificável nas previsões futuras, não apenas scores maiores |
| P3 | Busca limitada de regularização e receita de ajuste final | Ganho repetido nas janelas externas sob a mesma política de risco |
| P4 | Avaliação condicionada ao payout e cobertura das propostas | Diferenças entre payout fixo e observado ficam quantificadas |
| P5 | Ablacões de features e eventual modelo de sucesso do sinal | Benefício adicional acima do baseline congelado |
| P6 | Concorrente TabPFN, apenas se justificável | Comparação local com custo, latência e generalização registrados |

O primeiro lote deve conter apenas baseline atual, baseline de frequências, regressão logística e variantes de calibração do mesmo estimador. Misturar de uma vez calibrador, features, símbolo, duração e risco impediria atribuir uma melhora à sua causa. A busca de árvores deve vir depois, com orçamento fechado.

Os pontos de extensão são `train_model.py`, `training_evaluation.py` e `training_experiments.py`. O código operacional de execução não precisa ser alterado para realizar essas comparações. Novas métricas devem distinguir `eligible`, `promoted`, `loaded` e `entry_allowed`, e indicar se a ausência de operações decorre de indisponibilidade de modelo, falta de oportunidade técnica, confiança, payout ou risco.

As regressões necessárias para uma implementação incluem: nenhuma alteração no resultado de seleção ao modificar apenas os rótulos do holdout; ausência de sobreposição temporal indevida; calibrador ajustado sem acesso ao teste; classes de empate preservadas; cálculo correto do limiar por payout; proibição de cotações futuras; artefatos publicados somente em diretório experimental; e resultado determinístico com dataset e seed fixos.

## Limitações e decisão

A pesquisa identifica causas registradas e propõe experimentos; não mede o ganho dos métodos novos. Os modelos não foram retreinados, os snapshots dos relatórios não foram reconstruídos e não houve reprodução integral das métricas antigas. Os artefatos observados podem mudar com a coleta ativa. A ausência do modelo no caminho padrão foi confirmada na captura, mas não equivale a uma auditoria completa de todos os processos e diretórios.

Os estudos consultados tratam de problemas tabulares, calibração, seleção e validação em populações diferentes. Nenhum deles demonstra que o conjunto atual de features prevê com vantagem suficiente `frxUSDCAD` ou `1HZ25V` em 15 minutos. As versões `stable` da documentação são móveis; qualquer implementação deve conferir as APIs instaladas e registrar suas versões. O artigo de TabPFN publicado na Nature em 2025 foi localizado, mas seu texto integral não ficou acessível; a discussão técnica aqui usa o relatório TabPFN-2.5 efetivamente consultado.

A decisão recomendada é começar pela comparação de calibração e baselines em um snapshot imutável de `frxUSDCAD`. Isso responde diretamente ao colapso das previsões em uma única direção com baixa confiança e permite separar falta de sinal de uma transformação inadequada dos scores. Se a comparação não mostrar vantagem futura, o resultado correto é registrar a ausência de evidência, em vez de baixar os filtros para obter contratos.

## Fontes

As fontes online abaixo foram consultadas em 11/09/2026. Os números específicos do StanDeriv vêm dos artefatos locais inventariados em [evidencias.json](evidencias.json), especialmente os relatórios com prefixos `6122c93` e `3ff7b7b`, além de `pipeline.log`, `state.json` e do sidecar do candidato. A descrição do algoritmo foi conferida no código local, cujos hashes também constam da captura. O documento anterior [Melhorias de avaliação](../MELHORIAS_AVALIACAO_2026-09-10.md) foi usado como contexto, distinguindo seus resultados dos relatórios mais recentes.

[^1]: Ran El-Yaniv e Yair Wiener. [On the Foundations of Noise-free Selective Classification](https://jmlr.org/papers/v11/el-yaniv10a.html). JMLR 11, 1605–1641, 2010. Fundamenta a distinção entre erro seletivo e cobertura; não valida séries financeiras.
[^2]: Deriv. [Price Proposal](https://developers.deriv.com/docs/trading/proposal/) e [Proposal Comparison](https://developers.deriv.com/comparison/proposal/). Documentação oficial, sem data editorial indicada. Preço, payout e diferenças de esquema da API.
[^3]: Scikit-learn developers. [Tuning the decision threshold for class prediction](https://scikit-learn.org/stable/modules/classification_threshold.html). Documentação consultada, versão apresentada 1.9.1. Separação entre estimação e ação; prevenção de overfitting no limiar.
[^4]: Léo Grinsztajn, Edouard Oyallon e Gaël Varoquaux. [Why do tree-based models still outperform deep learning on tabular data?](https://arxiv.org/abs/2207.08815). 2022. Benchmarks tabulares e limitações das comparações de famílias de modelos.
[^5]: XGBoost developers. [XGBoost Parameters](https://xgboost.readthedocs.io/en/stable/parameter.html). Documentação consultada, versão apresentada 3.4.1. Parâmetros de complexidade, amostragem e regularização.
[^6]: James Bergstra e Yoshua Bengio. [Random Search for Hyper-Parameter Optimization](https://jmlr.org/papers/v13/bergstra12a.html). JMLR 13, 281–305, 2012. Eficiência de busca de hiperparâmetros.
[^7]: Scikit-learn developers. [Probability calibration](https://scikit-learn.org/stable/modules/calibration.html). Documentação consultada, versão apresentada 1.9.1. Curvas de confiabilidade, dados separados, métodos e limites de Brier/log loss.
[^8]: Chuan Guo, Geoff Pleiss, Yu Sun e Kilian Q. Weinberger. [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html). ICML/PMLR 70, 1321–1330, 2017. Temperature scaling; experimentos em imagens e documentos.
[^9]: Meelis Kull, Telmo Silva Filho e Peter Flach. [Beta calibration: a well-founded and easily implemented improvement on logistic calibration for binary classifiers](https://proceedings.mlr.press/v54/kull17a.html). AISTATS/PMLR 54, 623–631, 2017. Limitações da calibração logística e alternativa binária.
[^10]: Scikit-learn developers. [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html). Documentação consultada, versão apresentada 1.9.1. Divisões temporais, `gap` e condições de comparação.
[^11]: Scikit-learn developers. [Permutation feature importance](https://scikit-learn.org/stable/modules/permutation_importance.html). Documentação consultada. Interpretação de importâncias e correlação entre features.
[^12]: Léo Grinsztajn e colaboradores. [TabPFN-2.5: Advancing the State of the Art in Tabular Foundation Models](https://arxiv.org/abs/2511.08667). Relatório técnico, novembro de 2025, revisão de fevereiro de 2026. Evidência dos desenvolvedores em benchmarks tabulares.
[^13]: Deriv. [Synthetic indices trading](https://deriv.com/markets/derived-indices/synthetic-indices). Página oficial do produto e FAQ, sem data editorial indicada; o acesso redirecionou para a versão europeia. Limites declarados do uso de indicadores.
[^14]: Vince Stanzione / Deriv. [How to Trade Synthetic Indices](https://docs.deriv.com/marketing/2025/ebook-synthetics-en-hq.pdf), capítulo 8. O PDF informa publicação em 28/04/2026, apesar do caminho conter 2025. Exemplo da divergência entre conteúdo educacional e FAQ; não usado como comprovação de eficácia.
[^15]: David H. Bailey, Jonathan M. Borwein, Marcos López de Prado e Qiji Jim Zhu. [The Probability of Backtest Overfitting](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf). Versão revisada de 27/02/2015. Múltiplas tentativas e seleção retrospectiva de estratégias.
[^16]: Kevin Sheppard / arch developers. [Time-series Bootstraps](https://arch.readthedocs.io/en/latest/bootstrap/timeseries-bootstraps.html). Documentação consultada, versão apresentada 7.2.0. Bootstrap estacionário e métodos em blocos.
