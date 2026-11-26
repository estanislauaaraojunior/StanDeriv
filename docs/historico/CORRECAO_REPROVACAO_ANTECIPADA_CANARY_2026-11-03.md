> **Consulta histórica:** este registro mantém os achados e resultados da data
> indicada, sem redefinir o comportamento atual. Consulte o [README](../../README.md)
> e a [homologação de Score e coleta](../HOMOLOGACAO_SCORE_COLETA_2026-10-15.md).

# Reprovação antecipada do canary — 03/11/2026

Correção autorizada após a
[análise do canary sem entradas](ANALISE_CANARY_SEM_ENTRADAS_2026-11-03_1540.md).
O objetivo é deixar de esperar oito resultados quando os sete acertos
necessários já são impossíveis. Não reduz o critério de aprovação nem abre
compras com o modelo reprovado.

## Comportamento

`training_canary.py` encerra a tentativa quando **acertos + resultados
restantes < 7**. O segundo erro, em qualquer posição, já satisfaz essa
condição. A aprovação continua exigindo oito resultados, ao menos sete
acertos e saldo unitário positivo, com os gates de confiança, payout, dados,
demo e execução preservados.

O resultado é persistido como `rejection_reason=minimum_wins_unreachable`,
com a época do primeiro vencimento que tornou a meta impossível. Nenhuma
sequência ainda capaz de alcançar sete acertos é rejeitada por essa regra.
Não há reinício de janela para tentar até passar.

O schema 2 e a identidade da política são mantidos: o conjunto de sequências
aprováveis é idêntico. A validação de persistência passa a aceitar reprovação
antecipada comprovada pelos próprios resultados, sem permitir aprovação
antes do oitavo resultado. Decisões finais antigas permanecem preservadas.

## Estados existentes e pendências

`saved_canary` reconhece estados antigos inviáveis na consulta somente leitura,
devolvendo o término inferido sem modificar os arquivos. Ao carregar esse
estado, o monitor persiste a normalização, preservando todos os resultados,
a época original da política e uma eventual previsão simulada pendente.
O arquivo original do shadow e seus snapshots permanecem preservados.

Uma pendência simulada mantida após o término não altera mais o ledger nem
reabre a tentativa. Ela não é um contrato comprado. O monitor composto
publica a reprovação uma única vez por instância, acionando o callback já
existente de troca por falha shadow.

`pipeline.py` também reconhece o canary reprovado do bundle ativo na partida,
após tentar recuperar um bundle válido do registro. Sem contrato financeiro
pendente, solicita outro mercado antes de iniciar coleta/treino para aquele
bundle. A conferência vincula símbolo e hash do artefato. Pedidos explícitos
de retreino ou coleta nova preservam o fluxo próprio.

Se existe contrato financeiro pendente, a reconciliação continua tendo
prioridade. A rotina original do executor não fecha seu socket para trocar
de mercado enquanto `_in_trade` permanecer verdadeiro. A correção não cancela,
substitui nem apaga esse contrato.

## Verificações

Testes em cópias temporárias, sem `.env`, modelos ou dados operacionais, com
rede/DNS bloqueados e carregamento de modelos limitado à cópia. As propostas
de execução usam mocks. Ajustes sintéticos realizados pela suíte escrevem
somente nos diretórios temporários.

- **102 testes direcionados passaram em 3,62 segundos.**
- Foram percorridas **as 256 combinações possíveis de oito resultados**,
  verificando rejeição somente quando a meta fica impossível e aprovação
  somente ao completar oito com sete ou mais acertos. A retomada foi conferida
  após cada prefixo dessas sequências.
- Foram verificados marcador forjado, persistência, consulta somente leitura,
  retorno ao bundle reprovado após trocar de identidade, callback único,
  ausência de envio de proposta na reprovação e preservação do shadow original.
- Foram verificados o guard de inicialização e a preservação de contrato
  financeiro em reconciliação, sem fechar o socket ou alterar seu estado.
- Uma cópia do snapshot operacional foi reproduzida em diretório temporário,
  sem imports da configuração real ou pickle: seus **quatro resultados** e a
  previsão pendente foram preservados, a migração foi válida e o canary foi
  reconhecido como reprovado. O arquivo shadow original copiado permaneceu
  byte a byte igual. Nenhum original operacional foi gravado nessa reprodução.

## Escopo e ativação

Os únicos arquivos executáveis alterados nesta tarefa são
`training_canary.py` e trechos exclusivos da IA em `pipeline.py`. A comparação
com a cópia inicial confirmou que `executor.py`, `shadow_monitor.py`,
`config.py`, `ai_predictor.py`, `collector.py`, `strategy.py`,
`runtime_status.py` e `training_history.py` permaneceram idênticos nesta tarefa.
Os módulos homologados de Score/coleta e seus parâmetros não foram alterados.

O processo ativo não foi reiniciado e nenhum modelo operacional foi
substituído. A correção será aplicada quando o processo carregar o código
atualizado. Testes offline não demonstram ausência de toda falha futura,
nem garantem que um próximo candidato será aprovado ou abrirá contratos.
Os picos de latência relatados na auditoria não foram tratados por esta
correção de decisão terminal.
