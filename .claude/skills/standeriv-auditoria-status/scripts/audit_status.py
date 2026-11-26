#!/usr/bin/env python3
"""Resumo somente leitura do estado operacional do StanDeriv.

Não importa módulos do projeto, não desserializa pickle, não lê .env, não abre
conexões e não grava nada. Hash de modelo é calculado sobre os bytes do arquivo.
Uso: audit_status.py [--root DIR] [--log-lines N] [--json]
"""
import argparse, glob, hashlib, json, os, re, time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

SP = ZoneInfo('America/Sao_Paulo')
REDACT = [
    (re.compile(r'(pat_|ory_at_)[A-Za-z0-9_\-\.]+'), r'\1***'),
    (re.compile(r'(wss?://[^\s?]+)\?\S+'), r'\1?***'),
    (re.compile(r'\b(CR|VRTC|VRW|MF|MX|ROT|DOT)\d{4,}\b'), r'\1***'),
    (re.compile(r'(?i)(token|authorization|otp|bearer)(["\s:=]+)\S+'), r'\1\2***'),
    (re.compile(r'[\w.+-]+@[\w-]+\.[\w.]+'), '***@***'),
]


def sp(epoch):
    return datetime.fromtimestamp(epoch, SP).strftime('%d/%m %H:%M:%S') if epoch else None


def load(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        return {'_erro': f'{type(exc).__name__}'} if Path(path).exists() else None


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def age(iso_or_epoch):
    if not iso_or_epoch:
        return None
    try:
        ts = float(iso_or_epoch) if isinstance(iso_or_epoch, (int, float)) else \
            datetime.fromisoformat(str(iso_or_epoch).replace('Z', '+00:00')).timestamp()
        return round(time.time() - ts, 1)
    except ValueError:
        return None


def summarize_results(state, minimum_results=None, minimum_wins=None):
    results = state.get('results') or []
    wins = sum(1 for r in results if r.get('correct'))
    out = {
        'simbolo': state.get('symbol'), 'model_id': str(state.get('model_id') or '')[:16],
        'timeframe/duracao': f"{state.get('timeframe_sec')}/{state.get('duration_sec')}",
        'resultados': len(results), 'acertos': wins,
        'ultimos': [('✓' if r.get('correct') else '✗') + f" {sp(r.get('entry_epoch'))}"
                    f" {'BUY' if r.get('direction') == 1 else 'SELL' if r.get('direction') == 0 else r.get('direction')}"
                    f" payout={r.get('net_payout')}" for r in results[-8:]],
        'pendente': ({'entrada': sp(state['pending'].get('entry_epoch')),
                      'vencimento': sp(state['pending'].get('expiry_epoch')),
                      'confianca': round(state['pending'].get('confidence', 0), 3)}
                     if isinstance(state.get('pending'), dict) and state['pending'] else None),
        'aprovado_em': sp(state.get('approved_at_epoch')), 'reprovado_em': sp(state.get('rejected_at_epoch')),
    }
    if state.get('rejection_reason'):
        out['motivo_reprovacao'] = state['rejection_reason']
    if minimum_results and minimum_wins is not None:
        remaining = max(minimum_results - len(results), 0)
        out['meta'] = f'{minimum_wins}/{minimum_results}'
        out['max_acertos_possivel'] = wins + remaining
        out['meta_ainda_possivel'] = wins + remaining >= minimum_wins
    return out


def tail_lines(path, n, max_bytes=262144):
    with open(path, 'rb') as fh:
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        fh.seek(max(size - max_bytes, 0))
        lines = fh.read().decode('utf-8', 'replace').splitlines()[1 if size > max_bytes else 0:]
    out = []
    for line in lines[-n:]:
        for rx, rep in REDACT:
            line = rx.sub(rep, line)
        out.append(line[:240])
    return out


def last_tick(path):
    lines = tail_lines(path, 1, 8192) if Path(path).exists() else []
    if not lines:
        return None
    cols = lines[-1].split(',')
    try:
        return {'simbolo': cols[2], 'horario': sp(float(cols[0])), 'idade_s': age(float(cols[0]))}
    except (IndexError, ValueError):
        return {'linha': 'formato inesperado'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=str(Path(__file__).resolve().parents[4]))
    ap.add_argument('--log-lines', type=int, default=25)
    ap.add_argument('--json', action='store_true')
    a = ap.parse_args()
    R = Path(a.root)
    rep = {'raiz': str(R), 'lido_em': datetime.now(SP).strftime('%d/%m/%Y %H:%M:%S SP')}

    st = load(R / 'bot_status.json') or {}
    keys = ['status', 'running', 'pid', 'symbol', 'session_id', 'started_at', 'stopped_at',
            'use_trained_model', 'use_weighted_signal', 'use_price_action_filter',
            'use_candle_pattern_filter', 'model_timeframe_sec', 'candle_source', 'candle_count',
            'min_candles', 'warmup_remaining_candles', 'last_decision', 'decision_counts',
            'last_error', 'last_api_error', 'training_callback_timing']
    rep['status'] = {k: st.get(k) for k in keys if k in st}
    rep['status']['idade_ultimo_tick_s'] = age(st.get('last_tick_at'))
    rep['status']['idade_heartbeat_s'] = age(st.get('last_seen_at') or st.get('last_loop_at'))
    tr = st.get('training') or {}
    rep['treino_publicado'] = {k: tr.get(k) for k in ('stage', 'symbol', 'labels', 'minimum', 'new_labels',
                                                     'retry_target', 'ready_for_training',
                                                     'last_rejection', 'history_recovery') if k in tr} or None
    rep['shadow_publicado'] = st.get('shadow') or None

    state = load(R / 'state.json') or {}
    ac = state.get('active_contract')
    rep['contrato_ativo'] = ({'contract_id': ac.get('contract_id'), 'symbol': ac.get('symbol'),
                              'status': ac.get('status') or ac.get('phase')} if isinstance(ac, dict) and ac
                             else ac or None)

    model, meta = R / 'model.pkl', load(R / 'model.pkl.metadata.json')
    if model.exists():
        digest = sha256(model)
        m = meta or {}
        rep['modelo_ativo'] = {
            'sha256': digest[:16], 'confere_metadata': digest == m.get('artifact_sha256'),
            'simbolo': m.get('symbol'), 'timeframe': m.get('timeframe_sec'), 'nome': m.get('model_name'),
            'aprovado': m.get('approved'), 'treinado_em': m.get('trained_at'),
            'modificado_em': sp(model.stat().st_mtime)}
    else:
        rep['modelo_ativo'] = None
    rep['candidatos'] = sorted(Path(p).name for p in glob.glob(str(R / 'model.pkl.candidate.*'))
                               if not p.endswith('.json'))

    sh = load(R / 'shadow_state.json')
    rep['shadow_original'] = summarize_results(sh, 30, 17) if isinstance(sh, dict) and 'results' in sh else sh
    rep['canary'] = []
    for p in sorted(glob.glob(str(R / 'shadow_state.json.canary' / '*.json'))):
        c = load(p) or {}
        pol = c.get('policy') or {}
        item = summarize_results(c, pol.get('results'), pol.get('minimum_wins'))
        item['politica'] = pol.get('version')
        item['arquivo'] = Path(p).name[:16]
        rep['canary'].append(item)
    rep['estados_arquivados'] = len(glob.glob(str(R / 'shadow_state.json.states' / '*')))

    prog = R / '.training_inputs' / 'progress.json'
    rep['progresso_ia'] = {'modificado_em': sp(prog.stat().st_mtime)} if prog.exists() else None
    rep['ultimo_tick_csv'] = last_tick(R / 'ticks_v2.csv')
    log = R / 'pipeline.log'
    rep['log'] = tail_lines(log, a.log_lines) if log.exists() else None

    if a.json:
        print(json.dumps(rep, ensure_ascii=False, indent=2, default=str))
        return
    for k, v in rep.items():
        if k == 'log':
            print(f'\n== log (últimas {a.log_lines} linhas, sanitizadas)')
            for line in v or []:
                print('  ' + line)
        elif isinstance(v, (dict, list)):
            print(f'\n== {k}')
            print(json.dumps(v, ensure_ascii=False, indent=2, default=str))
        else:
            print(f'{k}: {v}')


if __name__ == '__main__':
    main()
