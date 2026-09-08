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

Sem modelo compatível, uma futura execução normal do pipeline mantém apenas a coleta, sem iniciar contratos ou retreino automático. Com `--skip-collect`, encerra em vez de ficar aguardando dados que não chegarão. Contratos pendentes têm prioridade de reconciliação independentemente da disponibilidade do modelo.

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
