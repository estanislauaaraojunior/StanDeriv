#!/usr/bin/env python3
"""Confere os módulos homologados de Score Ponderado e coleta.

Somente leitura: não importa módulos do projeto, não lê .env nem dados.
1. Compara o SHA-256 atual com a tabela de docs/HOMOLOGACAO_SCORE_COLETA_*.md.
2. Para arquivos divergentes, compara por AST (ignorando docstrings) com a
   versão do commit de homologação e lista as definições alteradas.

Uso: check_homologacao.py [--ref COMMIT] [--json]
Saída 0 = nenhum arquivo congelado alterado; 1 = há alterações a revisar.
"""
import argparse, ast, glob, hashlib, json, re, subprocess, sys
from pathlib import Path

ROOT = Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'],
                                    cwd=Path(__file__).parent, text=True).strip())
# Congelados por inteiro: qualquer mudança exige autorização explícita.
FROZEN = {'strategy.py', 'indicators.py', 'config.py', 'collector.py', 'market_data.py',
          'asset_history.py', 'market_calendar.py', 'api_keepalive.py',
          'request_throttle.py', 'deriv_session.py', 'feature_engine.py',
          'operational_diagnostics.py'}
# Compartilhados: aceitam mudanças exclusivas da IA, mas os trechos de coleta/Score
# abaixo devem permanecer idênticos.
CRITICAL = {'pipeline.py': ['_CollectorThread', '_fetch_historical_ticks', '_scan_group',
                            '_detect_top_trending_symbols', '_validate_deriv_token',
                            '_history_store']}


def git(*args):
    return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True)


def doc_table():
    docs = sorted(glob.glob(str(ROOT / 'docs' / 'HOMOLOGACAO_SCORE_COLETA_*.md')))
    if not docs:
        sys.exit('ERRO: documento de homologação não encontrado em docs/')
    text = Path(docs[-1]).read_text(encoding='utf-8')
    rows = re.findall(r'^\|\s*`([^`]+)`\s*\|\s*`([0-9a-f]{64})`\s*\|', text, re.M)
    return Path(docs[-1]).relative_to(ROOT).as_posix(), dict(rows)


def strip_docstrings(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], 'value', None), ast.Constant) \
                    and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
    return tree


def definitions(source):
    """Mapa nome-qualificado -> dump AST; inclui atribuições de módulo (constantes)."""
    tree = strip_docstrings(ast.parse(source))
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out[node.name] = ast.dump(node)
            if isinstance(node, ast.ClassDef):
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        out[f'{node.name}.{sub.name}'] = ast.dump(sub)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name):
                        out[f'={n.id}'] = ast.dump(node.value) if node.value else ''
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            out[f'import:{ast.unparse(node)}'] = ''
    return out


def homologation_ref(doc):
    r = git('log', '--diff-filter=A', '--format=%H', '--', doc)
    return r.stdout.split()[-1] if r.stdout.split() else 'HEAD'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', help='commit de referência (padrão: o que criou o doc de homologação)')
    ap.add_argument('--json', action='store_true')
    a = ap.parse_args()
    doc, table = doc_table()
    ref = a.ref or homologation_ref(doc)
    report = {'documento': doc, 'ref': ref[:10], 'arquivos': []}
    attention = False
    for name, expected in sorted(table.items()):
        path = ROOT / name
        cur = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        item = {'arquivo': name, 'hash_ok': cur == expected,
                'classe': 'congelado' if name in FROZEN else 'compartilhado'}
        if cur != expected:
            old = git('show', f'{ref}:{name}')
            if old.returncode != 0 or cur is None:
                item['erro'] = 'arquivo ausente no ref ou no checkout'
            else:
                before, after = definitions(old.stdout), definitions(path.read_text(encoding='utf-8'))
                item['so_docstrings_ou_formatacao'] = before == after
                item['alteradas'] = sorted(k for k in before.keys() & after.keys() if before[k] != after[k])
                item['adicionadas'] = sorted(after.keys() - before.keys())
                item['removidas'] = sorted(before.keys() - after.keys())
                crit = [c for c in CRITICAL.get(name, [])
                        if any(k == c or k.startswith(c + '.') for k in
                               item['alteradas'] + item['adicionadas'] + item['removidas'])]
                item['trechos_criticos_alterados'] = crit
            if name in FROZEN and not item.get('so_docstrings_ou_formatacao'):
                attention = True
            if item.get('trechos_criticos_alterados') or 'erro' in item:
                attention = True
        report['arquivos'].append(item)
    report['requer_revisao'] = attention
    if a.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"Referência: {doc} | commit {report['ref']}")
        for it in report['arquivos']:
            if it['hash_ok']:
                print(f"  OK         {it['arquivo']}")
                continue
            if 'erro' in it:
                print(f"  ERRO       {it['arquivo']}: {it['erro']}")
                continue
            tag = 'DOCSTRING ' if it['so_docstrings_ou_formatacao'] else 'ALTERADO  '
            print(f"  {tag} {it['arquivo']} [{it['classe']}]")
            for k in ('alteradas', 'adicionadas', 'removidas'):
                if it[k]:
                    print(f"      {k}: {', '.join(it[k][:25])}{' …' if len(it[k]) > 25 else ''}")
            if it['trechos_criticos_alterados']:
                print(f"      !! trechos críticos de coleta/Score: {', '.join(it['trechos_criticos_alterados'])}")
        print('RESULTADO:', 'REVISAR' if attention else 'congelados preservados')
    return 1 if attention else 0


if __name__ == '__main__':
    sys.exit(main())
