> **Consulta histórica:** este registro mantém os achados e resultados da data
> indicada, sem redefinir o comportamento atual. Consulte o [README](../../README.md)
> e a [homologação de Score e coleta](../HOMOLOGACAO_SCORE_COLETA_2026-10-15.md).

# Avaliação de uma validação mais curta — 03/11/2026

Este documento é proposta. Não altera código, limiares, estado shadow nem aprova modelos. Nenhum treino, autenticação, cotação conectada ou ordem foi executado.

Atualização posterior: o usuário autorizou implementar e testar o ajuste.
A política de 8 resultados/7 acertos foi implementada em `training_canary.py`,
com estado separado e preservação da validação original. O texto abaixo
registra o estudo e suas limitações; a implementação é descrita no
[registro de entrega](IMPLEMENTACAO_CANARY_8_RESULTADOS_2026-11-03.md).

## Recomendação

Separar a evidência de qualidade histórica da verificação operacional ao vivo: manter um holdout histórico recente com pelo menos 30 contratos elegíveis sem sobreposição, seguido por um canary forward de 8 resultados de 15 minutos do mesmo bundle. O mínimo teórico dessa parte forward é 2h a partir da primeira previsão elegível; abstenções, mercado fechado, empates ou gaps aumentam o tempo. A alternativa de 6 resultados tem mínimo de 1h30 e oferece ainda menos informação. Recomendo 8 como proposta para discussão, não como substituto estatisticamente equivalente aos 30 resultados atuais.

O holdout histórico pode ser recuperado agora para completar dados antigos. Isso não transforma resultados históricos em previsões forward: o modelo é construído depois desses eventos e os dois blocos devem permanecer distintos no relatório.

## Comparação

| Política | Mínimo forward ideal | O que comprova / limite |
| --- | ---: | --- |
| Atual: 30 resultados sequenciais de 15 min | 7h30 | Mais observações novas do bundle exato; não garante independência estatística ou desempenho futuro |
| Proposta: holdout histórico >=30 + canary 8 | 2h | Qualidade histórica e integração atual; evidência forward menor |
| Alternativa: holdout histórico >=30 + canary 6 | 1h30 | Verificação operacional ainda mais curta, incerteza maior |
| 30 previsões a cada 5 min, vencimento 15 min | 2h40 | Resultados sobrepostos; não contar como 30 contratos independentes |

No último caso, entradas em 0,5,...,145 minutos fecham até 160 minutos. Uma extração sem sobreposição permite apenas 10 contratos (entradas 0,15,...,135). O número efetivo de observações depende da dependência temporal e não é obtido simplesmente dividindo ou multiplicando a contagem bruta.

## O holdout que existe no código

`train_model.temporal_partitions` separa ajuste, calibração, validação e teste e aplica embargo por `expiry_epoch < próxima feature_start_epoch` (linhas 103–118). A seleção do estimador/calibrador ocorre no development antes do teste final (313–345). `_metrics` exige ao menos 30 contratos executáveis para aprovação (121–130). O teste final atual já compõe `metadata.approved` (400–403).

Esse teste pode ocupar a etapa histórica da proposta se era desconhecido quando a receita e os limiares foram congelados. Não se deve somar novamente o mesmo conjunto como um segundo teste independente ou chamá-lo de shadow forward. Se resultados de um período já orientaram receita, limiares ou várias tentativas de promoção, ele deixou de ser holdout intocado para essas decisões. É necessário reservar um período posterior desconhecido ou coletar dados novos; reproduzir o mesmo teste não cria evidência adicional.

A futura implementação precisa registrar hash do bundle/dataset, intervalos/IDs de avaliação, congelamento da política e relação entre períodos de seleção e de aprovação. Um ledger de conjuntos consumidos evita reusar os mesmos resultados em várias tentativas até encontrar uma aprovação. Trocar símbolo, calibrador, features ou modelo invalida o canary daquele bundle; um simples retorno ao mesmo bundle e à mesma política pode retomar o progresso sem perder resultados válidos.

## Pré-condições de uma versão curta

1. Política versionada e aceita explicitamente antes de ser aplicada. O usuário pediu avaliar uma política mais curta; isso não autoriza baixar automaticamente o gate atual.
2. Histórico e forward identificados separadamente; embargo temporal, rótulos por ticks reais e continuidade mantidos. “Sem sobreposição” não significa garantia de independência estatística: as features e regimes podem ser correlacionados.
3. Holdout com pelo menos 30 contratos realmente selecionados pela estratégia (não apenas 30 linhas ou capacidade teórica), aprovações atuais preservadas e comparação de desempenho por payout identificada.
4. Payout observado somente com cotações anteriores ao sinal e idade permitida, sem preencher retrospectivamente com proposta atual. `QuoteHistory.lookup` já aplica essa regra; a avaliação observada hoje é diagnóstica, não gate da aprovação. Sem cobertura suficiente, relatar cenário hipotético de payout e limitar o candidato a estudo/demo; não alegar rentabilidade observada.
5. Canary do bundle exato após congelamento, com símbolo, timeframe=300, duração=900, hash e versão da política; mesmos pisos de confiança/payout e vetos de gaps/empates. Preservar o limite de demo e todos os gates de execução existentes.
6. Definir antecipadamente N fixo, taxa mínima, resultado líquido, tratamento de falhas e rejeição. Uma opção concreta a avaliar é 8 resultados com pelo menos 7 acertos, payout observado >= piso atual e saldo unitário positivo. Esse 7/8 é uma escolha conservadora de operação, não limiar estatístico validado; precisa de avaliação offline e aceitação explícita. Não aprovar antes do oitavo resultado nem repetir janelas móveis até passar.
7. Persistência com identidade de política, além de símbolo/bundle/timeframe/duração. O estado atual schema=1 não identifica seus limiares: mudar apenas `minimum_results` pode reinterpretar arquivo antigo. Não converter aprovações/reprovações antigas nem misturar replay histórico com forward.
8. Continua sem autorização para iniciar bot, substituir modelo operacional ou abrir contrato real. Aprovação matemática de um candidato e autorização operacional são decisões distintas.

## Incerteza quantificada

Cálculo determinístico do intervalo de Wilson bilateral de 95%, z=1,96 (mesma fórmula de `training_evaluation.wilson_lower`), sob o modelo binomial de observações independentes:

| Exemplo hipotético | Acerto observado | Intervalo de Wilson 95% |
| --- | ---: | ---: |
| 5/6 | 83,33% | 43,65%–96,99% |
| 7/8 | 87,50% | 52,91%–97,76% |

Não são métricas deste bot. Mesmo taxas observadas altas em amostras pequenas deixam grande incerteza. Correlação temporal também limita a interpretação desse intervalo. Essa é a razão para o canary curto funcionar como verificação operacional acompanhada de holdout, e não substituir sozinho a evidência de desempenho.

Fórmula: centro=(p+z²/(2n))/(1+z²/n); margem=z*sqrt(p(1-p)/n+z²/(4n²))/(1+z²/n). Fonte metodológica primária: [NIST — Confidence intervals](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm). A separação temporal com intervalo entre treino/teste é coerente com [scikit-learn — TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html); o código do projeto usa embargo por timestamp mais específico que um gap fixo em linhas.

## Verificações realizadas

Leitura dos documentos obrigatórios, trainer, avaliação, shadow, cotações e testes relevantes. Cálculo puro da fórmula de Wilson e dos prazos, sem imports operacionais. Esta avaliação não alterou gates ou estado operacional. O estudo original foi produzido em diretório temporário e incorporado a este relatório; a implementação de candles de 5 minutos está descrita em [relatório separado](IMPLEMENTACAO_IA_5M_15M_2026-11-03.md).
