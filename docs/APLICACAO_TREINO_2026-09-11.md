# Aplicação das recomendações de treinamento

## Implementação

A pesquisa de [treinamento e qualidade das entradas](pesquisa_treino_2026-09-11/RELATORIO.md)
foi aplicada ao fluxo de treinamento existente, sem reiniciar o programa ou publicar
artefatos no caminho do modelo operacional.

- `train_model.py`: inclui frequência de classes e regressão logística com
  normalização ajustada no treino. Compara calibração multinomial e temperature
  scaling com o mesmo estimador ajustado por janela. As probabilidades brutas
  são um controle diagnóstico. Bundles multinomiais antigos continuam legíveis.
- Busca determinística de regularização: seed 42, quatro configurações adicionais
  por padrão e limite de dez, além das duas configurações anteriores de árvores.
  Com o orçamento padrão, são oito estimadores e 22 receitas diagnósticas,
  das quais 14 calibradas participam da seleção.
- Seleção usa apenas development: prioriza ganho de log loss sobre frequência
  de classes, suporte em todas as janelas, expectancy, drawdown e qualidade
  probabilística. Uma receita pode ser selecionada para avaliação mesmo sem
  ganho sobre o baseline; nesse caso, a limitação fica explícita nos avisos.
  Os critérios de elegibilidade existentes não foram relaxados.
- O ajuste final usa a mesma proporção 80/20 de ajuste/calibração das janelas
  internas, com embargo. A família escolhida pela receita anterior também é
  reajustada pela divisão antiga e avaliada no mesmo holdout, como comparação.
- `training_evaluation.py`: percentis de confiança, AUC por direção, cobertura,
  frequência de contratos e curvas de precisão/retorno com payouts 60/70/85%.
  As curvas são diagnósticas; não há busca de limiar no teste final. Precisão
  fica indefinida (`null`) nas curvas sem operações.
- `model_contract.py`: define o fuso da avaliação como UTC. Relatórios e sidecars
  novos registram `evaluation_risk_timezone`, distinguindo-o de `risk_timezone`,
  que continua representando o calendário local operacional. Isso evita atribuir
  a simulação UTC ao calendário local sem alterar o risco em execução.
- `training_experiments.py --study training`: exige destino novo, captura
  snapshots com verificação de estabilidade e executa sem carregar o champion.
  Grava candidato experimental e comparação; nunca promove o artefato.
- Dashboard: seleciona a linha walk-forward pelo identificador da receita,
  evitando confundir a versão bruta com a calibrada; exibe o método e o fuso.
- Dependência direta de SciPy declarada para ajuste numérico da temperatura.

## Avaliação executada

Comando executado na cópia atual do código:

```bash
.venv/bin/python training_experiments.py --study training \
  --input dataset_v2.csv \
  --output training_reports/recommendations-2026-09-11 \
  --symbol frxUSDCAD --search-trials 4 --payout-quotes payout_quotes.csv
```

O destino desse comando já existe; uma nova execução precisa de outro diretório.
Relatório: [comparison.json](../training_reports/recommendations-2026-09-11/comparison.json).
Os artefatos de experimento são ignorados pelo Git, mas permanecem disponíveis
localmente nesse diretório.

Dataset SHA-256:
`42d5f5ab49c57440a5c09729b89d599358f7aa023bf5dad7f883fe5663863fc2`.
São 2.017 linhas de `frxUSDCAD`, com 404 no teste final. A avaliação vai de
`1789099620` a `1789124700` em epoch UTC. A configuração utiliza candles de um
minuto e vencimento de 15 minutos. O manifesto registra seeds, hiperparâmetros,
hashes dos módulos e as fronteiras temporais do ajuste, calibração e teste.

| Medida no mesmo teste final | Receita anterior | Receita nova |
| --- | ---: | ---: |
| Modelo/calibrador | XGBoost/multinomial | RandomForest/temperature |
| Linhas no ajuste final | 955 | 1.083 |
| AUC | 0,4392 | 0,4812 |
| Log loss, menor é melhor | 0,7723 | 0,8451 |
| Brier multiclasses, menor é melhor | 0,5239 | 0,5513 |
| Contratos, payout hipotético 85% | 5 | 0 |
| Vitórias | 2 | Não aplicável |
| Elegível | Não | Não |

A receita nova teve AUC maior, mas pior log loss e Brier; permaneceu abaixo de
0,5 na AUC e não gerou contratos selecionados. Não foi demonstrado ganho de
assertividade nesse snapshot. O resultado bruto do mesmo RandomForest produziu
dois contratos, ambos perdedores. A baixa confiança após calibração não é,
portanto, evidência suficiente para justificar a remoção do calibrador.

No development, a receita selecionada somou três contratos, com duas janelas
sem operações. Seu log loss médio foi 1,0039, contra 0,8050 do baseline de
frequências. O relatório sinaliza essa ausência de ganho e a falta de suporte
temporal. A avaliação com cotações observadas não produziu contratos; nas 11
linhas que ultrapassaram os filtros técnicos anteriores, faltava cotação
histórica compatível. Não houve preenchimento por payout futuro.

Esse teste é uma avaliação exploratória sobre histórico já disponível na pesquisa,
não uma confirmação prospectiva inédita. A receita foi escolhida sem usar o
holdout nesta execução. Nenhum parâmetro foi retocado com base no seu resultado.

## Validação

- 38 testes direcionados de avaliação e recomendações passaram.
- A suíte completa passou com **366 testes**, em cópia temporária sem `.env`,
  modelos, dados ou arquivos operacionais.
- Compilação Python e verificação de sintaxe JavaScript passaram.
- Há regressões para temperatura, classe de empate ausente no ajuste,
  compatibilidade de pickle anterior, baseline sem acesso a rótulos futuros,
  orçamento determinístico, piso de confiança, metadados do fuso, isolamento,
  recusa de destino existente e invariância da seleção a alterações no holdout.

## Continuidade

O método novo está disponível para os próximos processos que carregarem o código
e pode ser exercitado offline pelo comando documentado. O processo ativo não foi
reiniciado nem recebeu um novo champion. Os filtros de entrada, limites de risco,
modo demo e critérios de promoção existentes foram preservados.

O próximo passo é coletar um bloco futuro com cotações correspondentes e repetir
o protocolo congelado. Experimentos de janela móvel, segundo estágio de seleção
de sinais ou TabPFN continuam condicionados à qualidade e ao tamanho dos dados;
não foram adicionados para tentar forçar aprovação nesse holdout.
