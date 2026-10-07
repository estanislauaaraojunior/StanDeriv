# Registro da Refatoração

## Escopo

Trabalho isolado na branch `refactor/estrutura-standeriv`, no worktree
`StanDeriv-refactor`. A linha de base é `e181105`, derivada do código em uso,
sem copiar arquivos de estado operacional. O diretório principal `StanDeriv`
permaneceu intocado; não houve conexão, execução de bot, operação financeira,
retreino ou reinício de serviço.

## Organização

Os módulos de runtime continuam na raiz. Entry points, imports absolutos usados
pelos scripts e caminhos construídos a partir de `__file__` tornam uma mudança
para pacote de maior risco do que o benefício nesta refatoração. `dashboard/` e
`tests/` já estavam separados por responsabilidade.

Relatórios datados e material de pesquisa foram agrupados em `docs/historico/`:

- `ANALISE_LOSSES_2026-09-08.md`
- `APLICACAO_TREINO_2026-09-11.md`
- `AUDITORIA_CONFLITOS_2026-09-11.md`
- `MANUTENCAO_TREINO_2026-09-10.md`
- `MELHORIAS_AVALIACAO_2026-09-10.md`
- `pesquisa_treino_2026-09-11/`
- `ARQUITETURA_ANTIGA.md`

Guias e prompts vigentes permanecem diretamente em `docs/`. O novo
`ARCHITECTURE.md` descreve a estrutura atual. Links internos foram atualizados
e verificados recursivamente.

## Código removido

- `transformer_model.py`: implementação legada sem uso no fluxo v2. Foram
  preservadas referências de caminho/diagnóstico de artefatos antigos onde
  permanecem no código; nenhum modelo, dataset ou arquivo de estado foi apagado.
- Funções sem uso removidas: `_extract_features_sequence`, `_parse_tick_row`,
  `setup_pat_options_session`, `extract_feature_vector`, `_best_from` e
  `_trim_ticks`.
- Quatorze imports sem uso foram removidos após inspeção estática. Falsos
  positivos de rotas do Flask foram tratados como usos, não como código morto.

`bot.py`, `collector.py`, a estratégia, o risco, os caminhos de autenticação e
os contratos operacionais foram preservados. Mudanças nesses arquivos durante
a documentação foram apenas docstrings, sem alteração da lógica executável.

## Documentação de funções

Foram acrescentadas docstrings às funções Python dos módulos e testes, e JSDoc
às funções de `dashboard/app.js`. A comparação de AST ignorando docstrings
passou nos arquivos Python alterados; a verificação de JavaScript também
confirmou que remover comentários JSDoc recompõe o conteúdo executável original.

## Verificações

- Baseline: 448 testes aprovados.
- Após remoção de código sem uso: 448 testes aprovados.
- Após docstrings nos testes: 448 testes aprovados.
- `node --check dashboard/app.js` passou.
- Os links locais dos 13 documentos Markdown foram verificados após a
  reorganização.

A suíte atual emitiu 50 avisos de depreciação relacionados a
`datetime.utcnow()` em `pipeline.py`; não são causados pela documentação nem
pelas remoções descritas acima.

## Aprovação e integração

Esta branch não foi mesclada nem enviada ao remoto. A conclusão dos testes
indica que a suíte observada continua passando, mas a aprovação de que todo o
comportamento original do bot foi preservado e a decisão de merge pertencem ao
usuário. Nenhuma execução conectada foi usada para essa aprovação.