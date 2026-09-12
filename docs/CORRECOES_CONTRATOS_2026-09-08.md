**Correções dos contratos — 08/09/2026**

O fluxo agora exige concordância explícita da IA, modelo compatível com o ativo e vencimento, dados temporais identificados e uma proposta cujo payout seja compatível com a probabilidade prevista. Foram mantidos os limiares dos indicadores, a stake percentual, os stops e as pausas; não houve ajuste para aumentar o número de entradas.

**Comportamento atual**

- `strategy.py`: abstenção, divergência e probabilidades inválidas bloqueiam a entrada. O score combina a força técnica efetiva com a confiança da direção selecionada. `ai_score` não é uma probabilidade de vitória.
- `market_data.py`: lê explicitamente o esquema, ordena por ativo/horário, remove duplicatas idênticas e separa conflitos. Dados legados não ganham uma origem presumida. Novas gravações usam bloqueio de arquivo e distinguem `tick` e `candle`.
- `dataset_builder.py`: usa 100 candles de tempo completos e contínuos por amostra. O alvo compara o primeiro tick disponível depois do sinal com o primeiro tick no vencimento. Mantém resultados de movimento pequeno e representa empate como classe 2; classe 0 é queda e classe 1 é alta. Empate não é uma PUT vencedora.
- `model_contract.py`: versiona ativo, timeframe, janela, features, regra de entrada, classes e expiração. Uma lacuna superior a 10 segundos impede usar a janela como uma sequência contínua de ticks. O dataset não usa candles históricos como preços de entrada de contratos.
- `feature_engine.py`: usa a mesma janela de 100 preços e OHLC temporal no treino e na inferência. As features de Price Action podem ser calculadas no treino, em vez de serem sistematicamente zeradas.
- `train_model.py`: separa treino, calibração, seleção e teste final, com embargo entre features/rótulos das partições. RandomForest e XGBoost são candidatos; a calibração multinomial usa um período posterior ao ajuste do estimador. A escolha usa validação, e o teste final é reservado para avaliação.
- `ai_predictor.py`: recusa modelos sem metadados, de outro ativo/vencimento, não aprovados ou com versões de dependências diferentes. Carrega um único bundle. Sem modelo compatível, não opera por fallback técnico.
- A expiração atual é explicitamente **15 minutos**, configurada em `DURATION`; não depende de uma falsa previsão de duração. O antigo TFT, o stacking não temporal e o classificador de duração não participam do novo caminho operacional. Reativar o ensemble exigiria sua própria calibração e avaliação conjunta; os `.pkl` antigos foram preservados.
- `executor.py`: usa OHLC temporal verdadeiro para padrões, elimina sobreposição de histórico CSV/API e atualiza o histórico de ADX também durante contratos/pausas. O scanner trabalha com candles do mesmo timeframe e verifica alinhamento entre EMA e MACD.
- Proposta e compra têm estados e IDs de requisição separados. A intenção é persistida antes do envio. Mensagens atrasadas ou repetidas não geram uma segunda compra. Sem confirmação de compra, novas entradas permanecem bloqueadas; um timeout não é tratado como cancelamento financeiro.
- Antes da compra, preço, payout e probabilidade precisam ser finitos, e `probabilidade × payout / preço - 1` precisa atingir margem de 2%. Essa margem é uma regra de admissão, não uma garantia de retorno. Dados insuficientes na proposta bloqueiam a compra.
- Contratos abertos permanecem rastreados até confirmação de liquidação. Lucro ausente/inválido não vira vitória. Zero vira `VOID`, sem zerar uma sequência de perdas. O log novo registra ID, versão de modelo, direção da IA, preço/payout e preços/horários oficiais disponíveis.
- `risk_manager.py`: conserva risco diário, pausas e resultados entre reinícios. Isola contas por identificador derivado, calcula drawdown em relação ao pico de saldo e mantém uma fila persistente de resultados ainda não exportados ao CSV. O ID impede contabilização repetida após reconexão/reinício.
- `dashboard/server.py`: iniciar/parar não apaga contratos nem risco. Limpar a visualização não destrói o histórico financeiro. Métricas candidatas são separadas das métricas do bundle ativo.
- CLI e pipeline compartilham exclusão de processo antes de alterar estado. O fluxo legado verifica `is_virtual`; o Bearer verifica a conta e busca novamente o saldo na reconexão.
- O backtest usa os mesmos OHLC para padrões, verifica compatibilidade temporal do modelo e recusa, pela CLI, dados de origem desconhecida ou com lacunas. O modo técnico offline continua disponível.

**Recuperação executada**

A recuperação foi gravada em `recovered_data/2026-09-08/`, ignorado pelo Git, sem substituir `ticks.csv`:

| Arquivo | Conteúdo |
| --- | --- |
| `original.csv` | Snapshot integral do arquivo recebido |
| `prices.csv` | 262.556 registros ordenados e sem conflitos conhecidos |
| `quarantine.json` | Dois registros inválidos e 16.861 registros conflitantes |
| `report.json` | Contagens, símbolos, proveniência e SHA-256 do snapshot |

Foram reconhecidos 338.380 registros válidos de entrada, removidas 58.963 duplicatas idênticas e preservados 23 símbolos. O SHA-256 do snapshot é `13436ab4152508c0002baf74d7d363e65f3b089dd0c9f70362d4ecccccd530bf`.

**Nenhum registro legado foi declarado elegível para treinar contratos.** O arquivo não informa quais linhas são ticks e quais são fechamentos históricos; parte dos conflitos é compatível com essa mistura. A recuperação melhora integridade e rastreabilidade, mas não reconstrói máximas/mínimas nem preços de entrada que nunca foram registrados.

As próximas gravações usam `ticks_v2.csv`, `dataset_v2.csv` e `operacoes_v2.csv`. Os arquivos antigos e os modelos antigos permanecem separados. O dashboard atualizado usa os arquivos v2; o histórico legado continua disponível em seus arquivos originais e no relatório da auditoria.

**Uso após a manutenção**

A manutenção não iniciou conexões, operações, coleta ou retreino operacional. Houve uma tentativa externa de iniciar o pipeline durante as alterações, registrada no log; essas mudanças externas em log/PID/estado foram preservadas.

Sem modelo compatível, o pipeline coleta e tenta treinar automaticamente assim que houver amostras contínuas suficientes. Contratos pendentes têm prioridade de reconciliação independentemente da disponibilidade do modelo.

O histórico inicial mínimo é calculado em `model_contract.minimum_history_candles`: na configuração de candles de 1 minuto e vencimento de 15 minutos, são 9.720 candles, reservando 9.600 amostras antes dos embargos temporais e 120 oportunidades sem sobreposição na fração de teste antes dos filtros. Esse orçamento foi ampliado na [manutenção de 10/09](MANUTENCAO_TREINO_2026-09-10.md). Esse orçamento de coleta não bloqueia o início do treino: `minimum_training_rows` calcula o piso estrutural de 1.815 amostras (incluindo a calibração das janelas walk-forward), e o treinador valida as partições reais, classes e embargo. Lacunas de mercado podem reduzir as linhas aproveitáveis. As exigências estatísticas de aprovação continuam inalteradas. Valores positivos menores de `--history-count` são elevados ao mínimo.

Uma inicialização normal começa agora um novo ciclo de aprendizado. Depois de obter exclusão de execução e antes de iniciar a coleta, arquiva ticks, datasets, modelos anteriores, candidatos e metadados em `recovered_data/training_resets/`, com manifesto. Não apaga contratos, risco, credenciais ou logs. O histórico recente é solicitado novamente nessa inicialização; os CSVs e modelos das execuções anteriores não são usados. `--reuse-training-data` permite reutilização explícita; `--skip-collect` também preserva os dados para uso offline. A presença de contrato pendente suspende o arquivamento para priorizar reconciliação. O arquivamento não é executado sobre o processo ativo pela manutenção.

Um `model.pkl` sem sidecar é considerado legado e ignorado pelo pipeline com aviso de treino automático, sem abrir o pickle ou inventar metadados. O novo treinamento grava o bundle e seu sidecar juntos no fluxo de candidatura e promoção.

O scanner valida os candidatos por score com uma assinatura de ticks e consulta de CALL/PUT na sessao da conta. Historico disponivel sozinho nao admite o ativo: exige tick do mesmo simbolo, preco valido e idade maxima de 10 segundos. Candidato rejeitado e descartado e o proximo e verificado. `--no-scan` prefere o símbolo configurado, mas procura outro se ele estiver indisponível. A validação também exige propostas CALL e PUT aceitas para a moeda e duração configuradas, sem efetuar compras. A validacao ocorre antes de arquivar dados; contratos pendentes continuam tendo prioridade e nao sao redirecionados para outro ativo.

O coletor legado aguarda autorizacao antes de assinar ticks. Desconexoes fecham o socket e a reconexao obtem nova URL de sessao, em vez de reutilizar a URL OTP anterior. Ha heartbeat e prazo de 30 segundos para receber o primeiro tick aceito. Erros definitivos de simbolo/autorizacao encerram a coleta e interrompem a espera pelo treino com o motivo, sem permanecer em zero ticks indefinidamente. A sondagem usa apenas [ticks](https://developers.deriv.com/comparison/ticks/) e [contracts_for](https://developers.deriv.com/comparison/contracts-for/), sem enviar propostas ou compras.

O coletor começa antes da busca dos candles e, ao receber o primeiro tick ao vivo, pagina os ticks históricos até o início desse período. Os ticks ao vivo continuam sendo gravados enquanto o histórico é recuperado e durante o treino. A recuperação também ocorre nas reconexões; épocas já presentes são deduplicadas. Candles completos da API prevalecem sobre agregações parciais na sobreposição. O snapshot de treino usa o mesmo bloqueio de arquivo dos escritores e é ordenado temporalmente antes de limitar seu tamanho.

Os preços de entrada e vencimento continuam vindo de ticks identificados, obtidos pelo endpoint [ticks_history da Deriv](https://developers.deriv.com/comparison/ticks-history/). Candles sozinhos não fornecem esses preços. Se a API não fornecer cobertura contínua, houver fechamento de mercado ou classes insuficientes, o pipeline informa o bloqueio e continua coletando. Não preenche lacunas com preços inventados. Depois de aprovado e promovido, o bundle é carregado pelo bot; as compras dependem da decisão da IA, dos filtros técnicos, do payout e do risco.

Após existir quantidade suficiente de ticks v2 para partições temporais independentes, a preparação offline pode usar um snapshot estável e um ativo explícito:

```bash
cp ticks_v2.csv /tmp/standeriv-ticks-snapshot.csv
.venv/bin/python dataset_builder.py \
  --input /tmp/standeriv-ticks-snapshot.csv \
  --output /tmp/standeriv-dataset-v2.csv \
  --symbol R_100
```

O treino manual gera candidato, não substitui o modelo ativo:

```bash
.venv/bin/python train_model.py \
  --input /tmp/standeriv-dataset-v2.csv \
  --output /tmp/standeriv-model.candidate \
  --symbol R_100
```

O símbolo de treino deve coincidir com `config.get_active_symbol()` no processo. O pipeline define esse valor a partir do ativo selecionado; a CLI de treino aceita `--symbol` para usar o ativo do dataset. `--force-retrain` do pipeline solicita explicitamente treino e só promove candidato aprovado. Não usar esses comandos como simples verificação de imports.

A promoção exige pelo menos 30 contratos direcionais não sobrepostos no teste, AUC superior a 0,5 e limite inferior de Wilson de 95% acima do ponto de equilíbrio do payout líquido de referência de 85%. Um champion existente precisa ser reavaliado nas mesmas linhas, fora do período já visto, e o challenger deve respeitar os deltas configurados. Um candidato rejeitado não redefine quem é o champion.

Essa avaliação aplica os filtros técnicos, a cadência e a gestão de risco da estratégia a contratos não sobrepostos, com payout de referência. Ela não reproduz os payouts variáveis nem a latência da corretora. A calibração e os cortes não demonstram rentabilidade futura. Antes de uso operacional, é necessário validar o conjunto completo fora da amostra e reconciliar os preços oficiais em conta de demonstração identificada.

Se uma compra perder a confirmação e não houver ID suficiente para vinculá-la com segurança, o sistema mantém o bloqueio. Ele não presume que a ordem falhou nem associa por semelhança um contrato possivelmente aberto por outra origem. Essa situação exige reconciliação com o histórico da conta; não apague `active_contract` para liberar entradas.

**Validação**

Resultado final: **244 testes passaram em 3,13 segundos**. `git diff --check` passou para os arquivos da manutenção, excluindo log/PID/estado alterados externamente.

A suíte foi executada em cópia temporária do projeto, com token fictício, modelos ausentes e arquivos de estado/log isolados. Foram adicionadas regressões para IA, origem do CSV, conflitos, rótulos de vencimento/empate, continuidade, igualdade das features, embargo temporal, promoção, requisições duplicadas/atrasadas, saldo na reconexão, fila de exportação e preservação do risco no dashboard.

Também há um teste de treino/calibração/serialização com dados inteiramente artificiais e quatro árvores por estimador, em diretório temporário. Ele valida a implementação, sem usar dados ou substituir modelos operacionais.

Fontes de protocolo verificadas: [compra](https://developers.deriv.com/comparison/buy/), [resultado de contrato](https://developers.deriv.com/comparison/proposal-open-contract/) e [ticks](https://developers.deriv.com/comparison/ticks/).


## Troca automática de mercado (2026-09-10)

Restrições de mercado identificadas nas propostas amostradas, nas propostas do
executor ou no feed solicitam uma troca. Um `ContractBuyValidationError` genérico
não basta: a mensagem precisa identificar indisponibilidade de negociação.
A amostragem ocorre a cada 300 segundos enquanto chegam ticks; erros de ordens
são tratados ao receber a resposta.

O executor bloqueia novas propostas e cancela propostas ainda não compradas.
Compras incertas continuam em reconciliação e contratos abertos são acompanhados
até liquidação antes da troca. Coletor e retreino anteriores terminam antes de
alterar os símbolos dos módulos. O lock de execução permanece adquirido.

A busca verifica os outros candidatos configurados com ticks recentes, catálogo
e propostas CALL/PUT para a duração configurada. Se nenhum passar, aguarda 60
segundos antes de nova busca. Dados e estado de risco são preservados; o novo
ativo exige seu próprio modelo aprovado e compatível. Um mercado disponível não
significa que já exista um modelo elegível para negociar nele.

Validação desta alteração: 315 testes passaram em cópia isolada, além de
compilação dos módulos alterados e `git diff --check`. Consultas autenticadas na
conta demo confirmaram a restrição do `frxUSDCHF` e propostas aceitas em `R_100`
e `R_50`. Após reinício com reutilização dos dados, o fallback real selecionou
`R_10`, iniciou coleta e recuperação do histórico. Nenhuma compra foi necessária
para validar disponibilidade; não foi afirmada aprovação de modelo para o novo
ativo.
