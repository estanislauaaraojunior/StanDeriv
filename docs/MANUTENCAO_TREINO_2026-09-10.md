# Treinamento e ausência de contratos — 10/09/2026

## Diagnóstico observado

A sessão já treinava candidatos, mas nenhum era aprovado. O dashboard mostrava
somente as métricas do modelo ativo, dando a impressão de que não havia treino.
Houve tentativas consecutivas com o mesmo hash de dataset e as mesmas métricas.

O orçamento inicial de 2.520 candles deixava, no máximo, cerca de 30 oportunidades
sem sobreposição no teste de 20%, antes dos filtros técnicos e das pausas de risco.
O corte do snapshot em 200 mil registros também limitava o crescimento do período
de avaliação, dependendo da frequência de ticks do ativo.

Durante a validação com símbolo fixado, foi reproduzida uma divergência entre o
símbolo usado pela coleta e `config.get_active_symbol()` no treinador. A atualização
dos consumidores ocorria somente quando o símbolo escolhido diferia do default.
Isso resultava em rejeição do dataset por incompatibilidade.

## Alterações

- `model_contract.py`: orçamento de histórico considera a fração do teste e reserva
  120 oportunidades sem sobreposição antes dos filtros. Na configuração atual,
  solicita 9.720 candles. Lacunas e disponibilidade da API podem reduzir a cobertura;
  o piso estrutural de 974 amostras para tentar o treino foi preservado.
- `pipeline.py`: preserva todo o snapshot, evita repetir treino de rótulos idênticos
  na mesma execução, mantém o último candidato rejeitado do retreino em
  `model.pkl.candidate` com sidecar e imprime métricas e motivos de reprovação.
  Sincroniza sempre o ativo nos consumidores e verifica o tipo da conta também
  na autenticação inicial legada.
- `train_model.py`: registra os critérios que impediram a aprovação, preservando
  o embargo temporal, a exigência de 30 contratos, o limite de Wilson e a AUC mínima.
- `dashboard/app.js` e `dashboard/index.html`: mostram métricas do candidato quando
  não há modelo aprovado, identificam explicitamente esse estado e exibem o bloqueio.
- `tests/test_training_progress.py`: regressões de cobertura, preservação de mais
  de 200 mil registros, retreino com rótulos novos, sincronização do ativo e recusa
  de conta real/desconhecida no modo demo. Substitui o antigo teste textual do corte.

## Validação

- 287 testes passaram em cópia temporária, sem credenciais ou artefatos operacionais.
- Compilação Python aprovada nessa cópia.
- Sintaxe de `dashboard/app.js` verificada com o runtime Node do VS Code
  (`ELECTRON_RUN_AS_NODE=1 /usr/share/code/code --check dashboard/app.js`).
- Autenticação real da API confirmou `account_type=demo`.
- Proposta CALL de 15 minutos para `frxUSDCAD` aceita: preço US$ 0,50, payout total
  US$ 0,85. A tentativa com US$ 0,35 informou mínimo de US$ 0,50. Não houve compra.
  A proposta confirma disponibilidade do contrato, não valida a estratégia.
- Pipeline reiniciado com reutilização de dados, mantendo `frxUSDCAD` e modo demo.
  O histórico passou de 397 mil ticks e produziu 5.829 amostras rotuladas.

O treino observado às 16:27:15 UTC produziu um candidato com 2.799 amostras de
treino, 1.166 de teste, AUC 0,5413 e 12 contratos avaliados, dos quais 5 vencedores.
O limite inferior de Wilson foi 19,3%, abaixo do equilíbrio de 54,1% do payout
líquido de referência de 85%. O candidato permaneceu reprovado. O payout total
da proposta ao vivo corresponde a retorno líquido de 70%; essa cotação isolada
não substitui a hipótese de avaliação nem demonstra resultado futuro.

## Ponto de continuidade

O treino voltou a concluir com o ativo consistente e o histórico ampliado. A
coleta e o retreino continuam. Ainda não há modelo aprovado para operações
automáticas; nenhum filtro foi removido para forçar compras. O próximo trabalho
é avaliar a estratégia fora da amostra com mais contratos independentes e payouts
observados, antes de considerar alterações no modelo ou na estratégia.

Protocolo consultado: [proposal](https://developers.deriv.com/comparison/proposal/),
[buy](https://developers.deriv.com/comparison/buy/) e
[proposal open contract](https://developers.deriv.com/comparison/proposal-open-contract/).
