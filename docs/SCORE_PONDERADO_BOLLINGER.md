# Score Ponderado homologado com Bollinger

**Homologado pelo usuário em 15/10/2026. Módulo congelado.** Não alterar
fórmulas, pesos, períodos, limiares, filtros, votos ou a base de preços sem nova
autorização explícita para rever a homologação. O escopo completo está no
[registro de Score e coleta](HOMOLOGACAO_SCORE_COLETA_2026-10-15.md).

## Base temporal e chamada operacional

O executor chama `strategy.get_signal()` com os **fechamentos dos candles de
15 minutos**, além dos OHLC e alertas correspondentes. A estratégia limita a
sequência aos últimos `FEATURE_WINDOW=50` fechamentos. Os períodos dos
indicadores são observações dessa sequência: Bollinger usa os últimos 20
fechamentos, e não os últimos 20 ticks ao vivo.

`ENFORCE_ENTRY_CADENCE=False` permite avaliar a cada tick, mas o preço usado no
score é o último fechamento disponível. Os indicadores permanecem iguais até
outro candle fechar. A troca por avaliações desfavoráveis conta apenas uma
avaliação por fechamento. O aquecimento do executor é `MIN_CANDLES=150`;
isto não altera a janela de cálculo de 50.

## Cálculo congelado

| Componente | Peso | Normalização antes da soma |
|---|---:|---|
| EMA9 versus EMA21 | 25% | `clip((EMA9-EMA21)/(EMA21*0.001+eps), -1, 1)` |
| Histograma MACD | 20% | `clip(hist/(abs(hist)+abs(EMA21*0.0001)+eps), -1, 1)` |
| Posição nas Bandas de Bollinger | 15% | `clip((preço-central)/((superior-inferior)/2), -1, 1)` |
| Preço versus EMA9 | 15% | `clip((preço-EMA9)/(EMA9*0.001+eps), -1, 1)` |
| Momentum de 3 observações | 15% | `clip(momentum/(preço*0.0005+eps), -1, 1)` |
| RSI | 10% | `(RSI-50)/50` |

`eps=1e-12`. A soma é multiplicada por `min(ADX/40, 1)`. No modo homologado
`ENTRY_MODE="balanced"`, há também gate de ADX: piso 30, adaptável para cima
pelo percentil 50 do histórico. ADX abaixo de 75% da média das últimas dez
observações de ADX também bloqueia. `ADX_ADAPTIVE=True` é preservado.

Bollinger usa `BB_PERIOD=20` e `BB_STD=2.0`. Acima da banda central contribui
para alta; abaixo, para baixa. Tocar uma banda não determina uma operação.
Bandas ausentes, não finitas ou sem largura válida bloqueiam somente o módulo
ponderado com `bollinger_invalid`. Os cálculos usam valores sem arredondamento;
`bb_score` e `tech_score` expostos são arredondados a quatro casas decimais.

`SIGNAL_SCORE_MIN=0.06`: score >= 0,06 propõe BUY; score <= -0,06 propõe SELL.
Entre esses limites, o módulo se abstém com `technical_score_below_min`.
A direção proposta ainda exige pelo menos três dos cinco votos originais:
EMA9-EMA21, histograma MACD, preço-EMA9, momentum e RSI-50. Valor positivo vota
BUY, negativo vota SELL e zero abstém. Bollinger não acrescenta um sexto voto.
Falta de três confirmações gera `technical_component_no_majority`.

O modo alternativo `opportunity` continua implementado, com limiar 0,05 e piso
0,35 para o multiplicador ADX. Sua existência não autoriza trocar o modo
balanceado homologado nem ajustar esses parâmetros.

## Consenso, diagnóstico e execução

`USE_WEIGHTED_SIGNAL=True` habilita um voto independente do Score Ponderado.
IA, Price Action e padrões também podem votar quando habilitados; a estratégia
exige maioria estrita do total de módulos ativos na mesma direção. Abstenções
e erros não diminuem o denominador. Os defaults de código habilitam Score e IA;
o painel pode selecionar apenas o Score para uma sessão.

Com apenas Score ativo, a estratégia preserva o motivo específico de rejeição,
em vez de substituí-lo por falta de maioria entre módulos. O status inclui
votos, detalhes dos módulos, momentum, Bollinger e confirmações técnicas.

O sinal não compra contratos por conta própria. Autenticação, histórico,
contrato pendente, proposta e payout são verificados pelo executor. Sem IA,
`valid_payout_proposal()` valida preço/payout e o piso líquido de 60%, sem
inventar probabilidade de vitória. Com IA ativa, continuam os gates de bundle,
probabilidade calibrada, retorno esperado e shadow/demo.

O bloqueio de risco depende de `ENFORCE_RISK_GATE`, atualmente `False`. A
homologação do Score não modifica nem homologa essa política.

## Verificação

As regressões existentes estão em `tests/test_weighted_bollinger.py`,
`tests/test_strategy.py`, `tests/test_strategy_edge_cases.py`,
`tests/test_indicators.py` e `tests/test_entry_recovery.py`. A suíte completa
executada em cópia isolada para esta homologação passou com **611 testes**.
Os arquivos executáveis foram preservados na revisão documental.

Os testes confirmam comportamento funcional. A homologação registra a
aceitação operacional do usuário; não demonstra vantagem estatística ou
rentabilidade futura.
