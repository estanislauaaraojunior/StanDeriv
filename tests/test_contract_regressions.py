"""Regressões dos defeitos encontrados na auditoria de 08/09."""
import csv
import json
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import config
import ai_predictor
import strategy
import feature_engine
from market_data import FIELDS, canonicalize, read_records, recover
from dataset_builder import build_dataset
from model_contract import FEATURE_WINDOW, compatible, specification
from risk_manager import RiskManager


@pytest.mark.parametrize('direction,confidence', [('SELL', .9), (None, .53), (None, .7), ('BUY', .54), ('BUY', float('nan'))])
def test_ai_disagreement_abstention_and_invalid_probability_block(direction, confidence):
    """Garante que discordância, abstenção ou probabilidade inválida da IA bloqueiam o sinal."""
    with patch('strategy.ind.price_action_features', return_value = None), patch('ai_predictor.predict', return_value = (direction, confidence)):
        signal, values = strategy._apply_ai_filter('BUY', [100] * 100, {'tech_score': .8}, use_ai_model = True)
    assert signal is None and 'win_probability' not in values


def test_ai_score_preserves_actual_technical_strength():
    """Garante que o ai_score combina a força técnica real com a probabilidade do modelo."""
    with patch('strategy.ind.price_action_features', return_value = None), patch('ai_predictor.predict', return_value = ('BUY', .8)):
        _, weak = strategy._apply_ai_filter('BUY', [100] * 100, {'tech_score': .05}, use_ai_model = True)
        _, strong = strategy._apply_ai_filter('BUY', [100] * 100, {'tech_score': .9}, use_ai_model = True)
    assert weak['ai_score'] < strong['ai_score']
    assert weak['ai_score'] == round(.55 * .05 + .45 * .8, 4)


def test_recovery_preserves_original_and_marks_unknown_provenance(tmp_path):
    """Garante que a recuperação preserva o arquivo original, deduplica e marca proveniência desconhecida sem elegibilidade de treino."""
    p = tmp_path / 'legacy.csv'
    p.write_text('120,x,A,2\n60,x,A,1\n60,x,A,1\n90,x,B,5\nbad,x,A,nan\n')
    before = p.read_bytes()
    report = recover(p, tmp_path / 'recovered')
    clean, rejected = read_records(tmp_path / 'recovered/prices.csv')
    assert p.read_bytes() == before
    assert report['duplicates'] == 1 and report['rejected'] == 1 and report['training_eligible'] == 0
    assert [(r['symbol'], r['epoch']) for r in clean] == [('A', 60), ('A', 120), ('B', 90)]
    with pytest.raises(ValueError):
        recover(p, tmp_path / 'recovered')


def test_conflicting_prices_quarantined_instead_of_arbitrarily_selected(tmp_path):
    """Garante que preços conflitantes no mesmo instante vão para quarentena em vez de serem escolhidos."""
    p = tmp_path / 'ticks.csv'
    p.write_text('60,x,A,1\n60,x,A,2\n120,x,A,3\n')
    records, _ = read_records(p)
    clean, conflicts, duplicates = canonicalize(records)
    assert len(clean) == 1 and len(conflicts) == 2 and duplicates == 0


def test_legacy_data_cannot_silently_train(tmp_path):
    """Garante que dados legados sem proveniência fazem build_dataset levantar ValueError."""
    p = tmp_path / 'ticks.csv'
    p.write_text('60,x,R_100,1\n120,x,R_100,2\n')
    with pytest.raises(ValueError, match = 'proveniência'):
        build_dataset(p, tmp_path / 'dataset.csv')


def write_ticks(path, minutes = 130, price_fn = None):
    """Grava um CSV v2 de ticks sintéticos de R_100 a cada 2 segundos."""
    base = 1_800_000_000
    with path.open('w', newline = '') as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        for offset in range(0, minutes * 60, 2):
            price = price_fn(offset) if price_fn else 100 + math.sin(offset / 90) + offset * .0001
            w.writerow([base + offset, '', 'R_100', price, 'tick', 0, '', '', ''])


def test_labels_use_entry_and_expiry_not_mean_next_two_prices(tmp_path):
    """Garante que os rótulos usam preço de entrada e de vencimento, e não a média dos próximos preços."""
    source = tmp_path / 'ticks.csv'
    out = tmp_path / 'dataset.csv'
    # Primeira entrada ocorre no segundo 6000, após 100 candles completos.
    def price(t):
        """Define preço 100 na entrada, 101 até o vencimento e 99 depois dele."""
        if t < 6000:
            return 100 + math.sin(t / 50)
        if t == 6000:
            return 100
        if t < 6900:
            return 101
        return 99
    write_ticks(source, price_fn = price)
    build_dataset(source, out)
    df = pd.read_csv(out)
    first = df.iloc[0]
    assert first.entry_epoch == 1_800_006_000 and first.expiry_epoch == 1_800_006_900
    assert first.target == 0 and first.entry_price == 100 and first.exit_price == 99
    metadata = json.loads(Path(str(out) + '.metadata.json').read_text())
    assert compatible(metadata, 'R_100')


def test_ties_kept_as_separate_outcome_without_future_noise_filter(tmp_path):
    """Garante que empates são mantidos como resultado separado (alvo 2) sem filtro de ruído futuro."""
    source = tmp_path / 'ticks.csv'
    out = tmp_path / 'dataset.csv'
    write_ticks(source, price_fn = lambda t: 100.)
    build_dataset(source, out)
    df = pd.read_csv(out)
    assert set(df.target) == {2}
    assert len(df) > 1 and (df.signal_epoch.diff().dropna() == 60).all()


def test_train_and_inference_features_are_identical_and_pa_available(tmp_path):
    """Garante que as features de treino e de inferência são idênticas e incluem Price Action."""
    source = tmp_path / 'ticks.csv'
    out = tmp_path / 'dataset.csv'
    write_ticks(source)
    build_dataset(source, out)
    from indicators import ticks_to_candles_by_time
    records, _ = read_records(source)
    candles = ticks_to_candles_by_time(records, 60)[: FEATURE_WINDOW]
    prices = [c['close'] for c in candles]
    live = ai_predictor._extract_features(prices, candles)
    trained = pd.read_csv(out).iloc[0][feature_engine.FEATURES].astype(float).values
    np.testing.assert_allclose(live, trained, rtol = 1e-10, atol = 1e-12)
    assert any(live[i] != 0 for i, k in enumerate(feature_engine.FEATURES) if k.startswith('pa_'))


def test_historical_candles_can_warm_features_but_live_ticks_label_contracts(tmp_path):
    """Garante que velas históricas aquecem as features, mas apenas ticks ao vivo rotulam contratos."""
    import dataset_builder
    source = tmp_path / 'ticks.csv'
    out = tmp_path / 'dataset.csv'
    base = 1_800_000_000
    with source.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(FIELDS)
        for i in range(FEATURE_WINDOW):
            epoch = base + i * config.CANDLE_TIMEFRAME_SEC
            price = 100 + i * .01
            writer.writerow([epoch, '', 'R_100', price, 'candle',
                             config.CANDLE_TIMEFRAME_SEC, price - .01, price + .02, price - .02])
        for offset in range(FEATURE_WINDOW * config.CANDLE_TIMEFRAME_SEC,
                            (FEATURE_WINDOW * config.CANDLE_TIMEFRAME_SEC) + 2400):
            writer.writerow([base + offset, '', 'R_100', 101 + offset * .00001,
                             'tick', 0, '', '', ''])

    estimated = dataset_builder.count_labelable_contracts(source, symbol='R_100')
    build_dataset(source, out)
    frame = pd.read_csv(out)
    assert len(frame) == estimated
    assert len(frame) > 0
    assert frame.iloc[0].signal_epoch == base + FEATURE_WINDOW * config.CANDLE_TIMEFRAME_SEC
    assert frame.iloc[0].entry_epoch == frame.iloc[0].signal_epoch
    assert frame.iloc[0].expiry_epoch == frame.iloc[0].entry_epoch + config.DURATION * 60


def test_feature_window_ignores_older_prices():
    """Garante que o mapa de features depende apenas da janela recente de preços e velas."""
    candles = [dict(epoch = i * 60, open = 100 + i * .1, high = 101 + i * .1, low = 99 + i * .1, close = 100 + i * .1) for i in range(200)]
    prices = [c['close'] for c in candles]
    assert feature_engine.compute_feature_map(prices, candles) == feature_engine.compute_feature_map(prices[-100:], candles[-100:])


def test_gap_prevents_false_contiguous_training(tmp_path):
    """Garante que uma lacuna nos dados impede treino como se fossem contínuos."""
    source = tmp_path / 'ticks.csv'
    out = tmp_path / 'dataset.csv'
    write_ticks(source)
    rows = list(csv.reader(source.open()))
    rows = [r for i, r in enumerate(rows) if i == 0 or not 1_800_005_990 < int(r[0]) < 1_800_006_200]
    with source.open('w', newline = '') as f:
        csv.writer(f).writerows(rows)
    with pytest.raises(ValueError, match = 'contínua'):
        build_dataset(source, out)


def test_duration_cannot_disagree_with_model_contract(tmp_path):
    """Garante que lookforward divergente do contrato do modelo é rejeitado e a duração prevista é a configurada."""
    with pytest.raises(ValueError, match = 'vencimento'):
        build_dataset(tmp_path / 'missing', tmp_path / 'out', lookforward = 2)
    assert ai_predictor.predict_duration([]) == config.DURATION


def test_old_model_or_wrong_symbol_is_incompatible():
    """Garante que metadados vazios ou de outro símbolo são incompatíveis."""
    assert not compatible({}, 'R_100')
    assert not compatible(specification('R_25'), 'R_100')


def test_sequence_targets_end_at_last_feature_and_gaps_fail():
    """Garante que as sequências terminam na última feature e que lacunas de época levantam ValueError."""
    from train_model import _build_sequence_dataset
    X = np.arange(6).reshape(-1, 1)
    seq = _build_sequence_dataset(X, 3, epochs = np.arange(6) * 60)
    assert seq[: , - 1, 0].tolist() == [2, 3, 4, 5]
    with pytest.raises(ValueError):
        _build_sequence_dataset(X, 3, epochs = [0, 60, 120, 240, 300, 360])


def test_risk_restart_keeps_daily_limits_and_peak():
    """Garante que reiniciar o RiskManager preserva limites diários, pausa, drawdown e ignora contrato repetido."""
    r = RiskManager(1000)
    r.record_result('R_100', 'BUY', 10, 15, - 10, {}, contract_id = '1')
    r.record_result('R_100', 'BUY', 10, 15, - 10, {}, contract_id = '2')
    r.record_result('R_100', 'BUY', 10, 15, - 10, {}, contract_id = '3')
    restored = RiskManager(9000)
    assert restored.balance == 970 and restored.is_paused()
    assert restored._daily_profit == - 30 and restored.to_state_dict()['drawdown_pct'] == 3
    assert not restored.record_result('R_100', 'BUY', 10, 15, - 10, {}, contract_id = '3')
    assert restored.balance == 970


def test_zero_profit_does_not_reset_losses_or_become_win():
    """Garante que lucro zero não zera as perdas consecutivas nem conta como vitória."""
    r = RiskManager(1000)
    r.record_result('R_100', 'BUY', 10, 15, - 10, {})
    r.record_result('R_100', 'BUY', 10, 15, 0, {})
    assert r._consec_losses == 1 and r._recent_results == [0]


def test_outbox_recovers_csv_after_interrupted_append(monkeypatch):
    """Garante que o outbox recupera a linha do CSV após uma gravação interrompida, sem duplicá-la."""
    r = RiskManager(1000)
    original = r._flush_outbox
    count = 0
    def interrupted():
        """Simula falha de disco na segunda chamada de _flush_outbox."""
        nonlocal count
        count += 1
        if count == 2:
            raise OSError('simulated disk interruption')
        original()
    monkeypatch.setattr(r, '_flush_outbox', interrupted)
    with pytest.raises(OSError):
        r.record_result('R_100', 'BUY', 10, 15, - 10, {}, contract_id = 'one')
    restored = RiskManager(1000)
    assert restored.balance == 990 and not restored._outbox
    assert not restored.record_result('R_100', 'BUY', 10, 15, - 10, {}, contract_id = 'one')
    with (restored._path.parent / 'operations.csv').open() as f:
        assert len(list(csv.DictReader(f))) == 1


def test_training_partitions_purge_overlapping_labels_and_features():
    """Garante que as partições temporais eliminam sobreposição entre rótulos e features posteriores."""
    from train_model import temporal_partitions
    epochs = np.arange(4000) * 60
    frame = pd.DataFrame(dict(signal_epoch = epochs, feature_start_epoch = epochs - 6000,
                            entry_epoch = epochs, expiry_epoch = epochs + 900, target = np.arange(4000) % 2))
    parts = temporal_partitions(frame)
    for earlier, later in zip(parts, parts[1:]):
        assert earlier.expiry_epoch.max() < later.feature_start_epoch.min()
    assert parts[-1].index.min() == 3200


def test_model_gate_rejects_missing_quality_and_incomparable_periods():
    """Garante que a promoção rejeita candidato sem qualidade ou com período de avaliação incomparável ao campeão."""
    from pipeline import _should_promote_model
    assert not _should_promote_model(None, {'auc': .9, 'accuracy': .9})[0]
    candidate = dict(eligible = True, auc = .7, accuracy = .65, f1 = .65, evaluation_start = 100, evaluation_end = 200, n_test = 40)
    assert _should_promote_model(None, candidate)[0]
    assert not _should_promote_model({'auc': .6}, candidate)[0]
    candidate['champion_metrics'] = dict(candidate, auc = .6, evaluation_start = 99)
    assert not _should_promote_model({'auc': .6}, candidate)[0]
    candidate['champion_metrics'] = dict(candidate, auc = .6, evaluation_start = 100)
    assert _should_promote_model({'auc': .6}, candidate)[0]


def test_model_loader_refuses_legacy_and_mismatching_contract(tmp_path, monkeypatch):
    """Garante que o carregador recusa modelos legados ou com contrato divergente."""
    import joblib
    path = tmp_path / 'model.pkl'
    monkeypatch.setattr(config, 'AI_MODEL_PATH', str(path))
    for payload in ({'model': None, 'features': feature_engine.FEATURES},
                    {'model': None, 'features': feature_engine.FEATURES, 'metadata': specification('R_25')}):
        joblib.dump(payload, path)
        ai_predictor.reset()
        assert ai_predictor.status()['ready'] is False
        assert ai_predictor.predict([100] * 100)[0] is None


def test_model_loader_checks_sidecar_before_pickle(tmp_path, monkeypatch):
    """Garante que o metadado sidecar é verificado antes de abrir o pickle."""
    import joblib
    path = tmp_path / 'model.pkl'
    path.write_bytes(b'old pickle')
    monkeypatch.setattr(config, 'AI_MODEL_PATH', str(path))
    monkeypatch.setattr(joblib, 'load', lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('pickle aberto cedo demais')))
    ai_predictor.reset()
    assert ai_predictor.status()['ready'] is False


def test_missing_model_sidecar_error_is_actionable(tmp_path):
    """Garante que a ausência do sidecar gera FileNotFoundError com mensagem acionável."""
    from model_contract import load_validated_bundle
    path = tmp_path / 'model.pkl'
    path.write_bytes(b'old pickle')
    with pytest.raises(FileNotFoundError) as exc:
        load_validated_bundle(path, config.get_active_symbol())
    assert 'metadado do modelo não encontrado' in str(exc.value)
    assert '--force-retrain' in str(exc.value)


def test_one_minute_and_fifteen_minute_contracts_cannot_share_bundle(monkeypatch):
    """Garante que um bundle de 15 minutos é incompatível quando a duração configurada é 1."""
    metadata = specification(config.get_active_symbol())
    monkeypatch.setattr(config, 'DURATION', 1)
    assert not compatible(metadata, config.get_active_symbol())


def test_distinct_accounts_keep_distinct_risk_history():
    """Garante que contas distintas mantêm saldo, lucro diário e drawdown separados."""
    r = RiskManager(1000)
    r.bind_account('account-a', 1000)
    r.record_result('R_100', 'BUY', 10, 15, - 10, {}, contract_id = 'a1')
    r.bind_account('account-b', 2000)
    assert r.balance == 2000 and r._daily_profit == 0
    r.record_result('R_100', 'BUY', 10, 15, - 20, {}, contract_id = 'b1')
    r.bind_account('account-a', 990)
    assert r.balance == 990 and r._daily_profit == - 10
    assert r.to_state_dict()['drawdown_pct'] == 1


def test_writer_refuses_to_append_v2_rows_to_legacy_file(tmp_path):
    """Garante que append_record recusa gravar linhas v2 em arquivo legado sem alterá-lo."""
    from market_data import append_record
    path = tmp_path / 'old.csv'
    path.write_text('60,x,R_100,100\n')
    original = path.read_bytes()
    with pytest.raises(ValueError):
        append_record(path, 61, 'R_100', 101)
    assert path.read_bytes() == original


def test_scanner_requests_time_candles(monkeypatch):
    """Garante que o scanner pede velas por tempo com a granularidade configurada."""
    import pipeline
    sent = []
    class FakeSocket:
        """WebSocket falso que autentica e responde histórico de velas vazio."""
        def __init__(self, url, on_open, on_message, on_error, on_close):
            """Guarda os callbacks de abertura, mensagem e fechamento."""
            self.on_open = on_open
            self.on_message = on_message
            self.on_close = on_close
        def run_forever(self):
            """Simula a abertura da conexão."""
            self.on_open(self)
        def send(self, raw):
            """Registra a requisição e responde com autorização ou velas vazias."""
            message = json.loads(raw)
            sent.append(message)
            if 'authorize' in message:
                self.on_message(self, json.dumps({'msg_type': 'authorize'}))
            else:
                self.on_message(self, json.dumps({'msg_type': 'candles', 'candles': []}))
        def close(self):
            """Dispara o callback de fechamento."""
            self.on_close(self)
    monkeypatch.setattr(pipeline, '_get_ws_url', lambda: 'offline')
    monkeypatch.setattr(pipeline, '_uses_bearer_auth', lambda: False)
    monkeypatch.setattr(pipeline.websocket, 'WebSocketApp', FakeSocket)
    pipeline._fetch_prices_for_symbol('R_100', 100)
    request = next(m for m in sent if 'ticks_history' in m)
    assert request['style'] == 'candles' and request['granularity'] == config.CANDLE_TIMEFRAME_SEC


def test_history_fetch_paginates_and_counts_candles(tmp_path, monkeypatch):
    """Garante que o histórico é buscado em páginas de até 1000 velas e gravado como velas sem erros."""
    import pipeline
    sent = []
    now = 1_800_200_000

    def candles(end, amount):
        """Gera amount velas consecutivas terminando em end (ou na última vela fechada)."""
        latest = now - config.CANDLE_TIMEFRAME_SEC if end == 'latest' else int(end)
        start = latest - (amount - 1) * config.CANDLE_TIMEFRAME_SEC
        return [
            {'epoch': epoch, 'open': 100, 'high': 101, 'low': 99, 'close': 100}
            for epoch in range(start, latest + 1, config.CANDLE_TIMEFRAME_SEC)
        ]

    class FakeSocket:
        """WebSocket falso que autentica e responde páginas de velas conforme count e end."""
        def __init__(self, url, on_open, on_message, on_error, on_close):
            """Guarda os callbacks de abertura, mensagem e fechamento."""
            self.on_open = on_open
            self.on_message = on_message
            self.on_close = on_close
        def run_forever(self):
            """Simula a abertura da conexão."""
            self.on_open(self)
        def send(self, raw):
            """Registra a requisição e responde com autorização ou com uma página de velas."""
            message = json.loads(raw)
            sent.append(message)
            if 'authorize' in message:
                self.on_message(self, json.dumps({'msg_type': 'authorize'}))
                return
            amount = min(int(message['count']), 1000)
            self.on_message(self, json.dumps({'msg_type': 'candles', 'candles': candles(message['end'], amount)}))
        def close(self):
            """Dispara o callback de fechamento."""
            self.on_close(self, None, None)

    monkeypatch.setattr(pipeline, 'TICKS_CSV', str(tmp_path / 'ticks.csv'))
    monkeypatch.setattr(pipeline, 'SYMBOL', 'frxEURAUD')
    monkeypatch.setattr(pipeline, '_get_ws_url', lambda: 'offline')
    monkeypatch.setattr(pipeline, '_uses_bearer_auth', lambda: False)
    monkeypatch.setattr(pipeline, '_probe_symbol', lambda symbol: (True, ''))
    monkeypatch.setattr(pipeline.websocket, 'WebSocketApp', FakeSocket)
    monkeypatch.setattr(pipeline.time, 'time', lambda: now)

    assert pipeline._fetch_historical_ticks(1200) == 1200
    requests = [m for m in sent if 'ticks_history' in m]
    assert [r['count'] for r in requests] == [1000, 200]
    records, rejected = read_records(tmp_path / 'ticks.csv')
    assert not rejected
    assert len(records) == 1200 and {r['kind'] for r in records} == {'candle'}


def test_history_fetch_refuses_closed_market_before_request(tmp_path, monkeypatch):
    """Garante que o histórico não abre WebSocket quando o mercado está fechado."""
    import pipeline
    monkeypatch.setattr(pipeline, 'TICKS_CSV', str(tmp_path / 'ticks.csv'))
    monkeypatch.setattr(pipeline, 'SYMBOL', 'frxEURAUD')
    monkeypatch.setattr(pipeline, '_probe_symbol', lambda symbol: (False, 'market closed'))

    def unexpected_socket(*args, **kwargs):
        """Falha o teste se um WebSocket de histórico for aberto."""
        pytest.fail('nao deve abrir WebSocket de historico para mercado fechado')

    monkeypatch.setattr(pipeline.websocket, 'WebSocketApp', unexpected_socket)
    assert pipeline._fetch_historical_ticks(10) == 0


def test_scan_skips_locally_closed_market_before_fetch(monkeypatch):
    """Garante que o scanner pontua com 0 e não busca preços de mercado fechado localmente."""
    import pipeline
    fetched = []
    monkeypatch.setattr(pipeline, '_local_market_open', lambda symbol: symbol != 'frxEURUSD')
    monkeypatch.setattr(pipeline, '_fetch_prices_for_symbol',
                        lambda symbol, count: fetched.append(symbol) or [100 + i for i in range(60)])
    results = pipeline._scan_group(['frxEURUSD', 'R_100'])
    assert results['frxEURUSD'] == 0.0
    assert fetched == ['R_100']


def test_pipeline_training_wait_counts_only_live_ticks(tmp_path, monkeypatch):
    """Garante que a espera de treino conta apenas ticks ao vivo, não as velas históricas."""
    import pipeline
    path = tmp_path / 'ticks.csv'
    base = 1_800_000_000
    with path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(FIELDS)
        for i in range(2000):
            writer.writerow([base + i * config.CANDLE_TIMEFRAME_SEC, '', 'frxEURGBP', 0.86,
                             'candle', config.CANDLE_TIMEFRAME_SEC, 0.85, 0.87, 0.84])
        for i in range(2):
            writer.writerow([base + 2000 * config.CANDLE_TIMEFRAME_SEC + i, '', 'frxEURGBP',
                             0.86, 'tick', 0, '', '', ''])
    monkeypatch.setattr(pipeline, 'TICKS_CSV', str(path))
    monkeypatch.setattr(pipeline, 'SYMBOL', 'frxEURGBP')
    assert pipeline._count_market_records() == 2002
    assert pipeline._count_ticks() == 2


def test_pipeline_counts_runtime_symbol_after_scan(tmp_path, monkeypatch):
    """Garante que a contagem de ticks considera apenas o símbolo de runtime após o scan."""
    import pipeline
    path = tmp_path / 'ticks.csv'
    base = 1_800_000_000
    with path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(FIELDS)
        writer.writerow([base, '', 'R_100', 100, 'tick', 0, '', '', ''])
        for i in range(3):
            writer.writerow([base + i, '', 'frxEURAUD', 1.5, 'tick', 0, '', '', ''])
    monkeypatch.setattr(pipeline, 'TICKS_CSV', str(path))
    monkeypatch.setattr(pipeline, 'SYMBOL', 'frxEURAUD')
    assert pipeline._count_ticks() == 3


def test_pipeline_fallback_uses_open_synthetic_when_default_unavailable(monkeypatch):
    """Garante que, sem mercado padrão disponível, o fallback escolhe um sintético aberto (R_25)."""
    import pipeline
    monkeypatch.setattr(pipeline, '_scan_group', lambda symbols: {s: 0.0 for s in symbols})

    def fake_probe(symbol):
        """Considera aberto apenas o símbolo R_25."""
        return (symbol == 'R_25'), ('closed' if symbol != 'R_25' else '')

    monkeypatch.setattr(pipeline, '_probe_symbol', fake_probe)
    assert pipeline._detect_trending_symbol() == 'R_25'


def test_pipeline_raises_history_to_feature_warmup_minimum():
    """Garante que a contagem efetiva de histórico é elevada ao mínimo de aquecimento das features."""
    import pipeline
    assert pipeline._minimum_history_candles() > FEATURE_WINDOW
    assert pipeline._effective_history_count(1) == pipeline._minimum_history_candles()
    assert pipeline._effective_history_count(pipeline._minimum_history_candles() + 10) == pipeline._minimum_history_candles() + 10
    assert pipeline._effective_history_count(0) == 0


def test_startup_minimum_survives_temporal_embargo():
    """Garante que o histórico mínimo de partida ainda deixa partições temporais com amostra suficiente após o embargo."""
    from model_contract import minimum_history_candles, duration_seconds
    from train_model import temporal_partitions
    horizon = duration_seconds(config.DURATION, config.DURATION_UNIT)
    count = minimum_history_candles() - FEATURE_WINDOW - math.ceil(horizon / config.CANDLE_TIMEFRAME_SEC)
    signals = 1_800_000_000 + np.arange(count) * config.CANDLE_TIMEFRAME_SEC
    df = pd.DataFrame({'entry_epoch': signals, 'expiry_epoch': signals + horizon,
                       'feature_start_epoch': signals - FEATURE_WINDOW * config.CANDLE_TIMEFRAME_SEC,
                       'target': np.arange(count) % 2})
    parts = temporal_partitions(df)
    assert all(len(part) >= 30 for part in parts)
    assert len(parts[-1]) * config.CANDLE_TIMEFRAME_SEC >= 30 * horizon


def test_training_starts_with_2280_rows_after_market_gaps(monkeypatch):
    """Garante que 2280 contratos rotuláveis bastam para iniciar o treino sem aguardar o orçamento de coleta."""
    import pipeline
    monkeypatch.setattr(pipeline, '_count_ticks', lambda: 150492)
    monkeypatch.setattr(pipeline, '_count_labelable_contracts_for_training', lambda: 2280)
    def unexpected_wait(*args):
        """Falha o teste se o pipeline dormir aguardando coleta."""
        pytest.fail('Dados suficientes ficaram aguardando o orcamento de coleta')
    monkeypatch.setattr(pipeline.time, 'sleep', unexpected_wait)
    pipeline._wait_for_ticks(200)


def test_new_training_start_archives_learning_only(tmp_path, monkeypatch):
    """Garante que o início de um novo treino arquiva só artefatos de aprendizado e preserva os operacionais."""
    import pipeline
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pipeline, 'TICKS_CSV', 'ticks_v2.csv')
    monkeypatch.setattr(pipeline, 'DATASET_CSV', 'dataset_v2.csv')
    monkeypatch.setattr(pipeline, 'AI_MODEL_PATH', 'model.pkl')
    learning = ['ticks_v2.csv', 'ticks.csv', 'dataset.csv', 'dataset_v2.csv',
                'model.pkl', 'model.pkl.new.metadata.json', 'model_metrics.json',
                'transformer_model.pkl', 'duration_model.pkl']
    operational = ['state.json', 'risk_state.json', 'risk_accounts.json', 'operacoes_v2.csv',
                   '.env', 'pipeline.log']
    for name in learning + operational:
        (tmp_path / name).write_text(name)
    archive = pipeline._prepare_training_start()
    manifest = json.loads((archive / 'manifest.json').read_text())
    assert len(manifest) == len(learning)
    for entry in manifest:
        assert (archive / entry['backup']).read_text() == Path(entry['source']).name
        assert not Path(entry['source']).exists()
    for name in operational:
        assert (tmp_path / name).read_text() == name
    assert pipeline._prepare_training_start() is None


@pytest.mark.parametrize('options', [dict(pending={'symbol': 'R_100'}),
                                     dict(reuse=True), dict(skip_collect=True)])
def test_training_reset_preserves_reconciliation_and_reuse(tmp_path, monkeypatch, options):
    """Garante que o reset de treino não arquiva o modelo quando há reconciliação pendente, reuso ou coleta pulada."""
    import pipeline
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'model.pkl').write_text('old model')
    assert pipeline._prepare_training_start(**options) is None
    assert (tmp_path / 'model.pkl').read_text() == 'old model'


def test_pipeline_missing_sidecar_requests_new_training(tmp_path, monkeypatch, capsys):
    """Garante que modelo sem sidecar não é carregado e o pipeline sinaliza treino inicial automático."""
    import pipeline
    model = tmp_path / 'model.pkl'
    model.write_bytes(b'legacy pickle must not be loaded')
    monkeypatch.setattr(pipeline, 'AI_MODEL_PATH', str(model))
    assert pipeline._load_latest_final_metrics() is None
    assert 'treino inicial automatico' in capsys.readouterr().out


def test_collector_backfills_while_live_ticks_continue(tmp_path, monkeypatch):
    """Garante que o coletor preenche o histórico anterior em páginas enquanto ticks ao vivo continuam, sem conflitos."""
    import pipeline
    from market_data import ensure_header
    from unittest.mock import Mock
    path = tmp_path / 'ticks.csv'
    ensure_header(path)
    monkeypatch.setattr(pipeline, 'TICKS_CSV', str(path))
    monkeypatch.setattr(pipeline, '_minimum_history_candles', lambda: 2)
    base = 1_800_000_000
    collector = pipeline._CollectorThread('R_100')
    ws = Mock()
    def tick(epoch):
        """Entrega um tick ao vivo ao coletor."""
        collector._on_message(ws, json.dumps({'tick': {'epoch': epoch, 'quote': 100, 'symbol': 'R_100'}}))
    def history(epochs):
        """Entrega ao coletor uma resposta de histórico com as épocas dadas."""
        collector._on_message(ws, json.dumps({'msg_type': 'history',
                              'history': {'times': epochs, 'prices': [100] * len(epochs)}}))
    tick(base)
    request = next(json.loads(call.args[0]) for call in ws.send.call_args_list
                   if 'ticks_history' in json.loads(call.args[0]))
    assert request['end'] == base - 1
    assert request['start'] == base - 120
    tick(base + 1)
    history(list(range(base - 60, base)))
    assert json.loads(ws.send.call_args.args[0])['end'] == base - 61
    tick(base + 2)
    history(list(range(base - 120, base - 60)))
    assert collector.history_ready.is_set()
    records, rejected = read_records(path)
    records, conflicts, duplicates = canonicalize(records)
    assert not rejected and not conflicts and not duplicates
    assert [r['epoch'] for r in records] == list(range(base - 120, base + 3))


def test_complete_history_ohlc_survives_partial_live_overlap():
    """Garante que o OHLC completo da vela histórica sobrevive à sobreposição parcial com ticks ao vivo."""
    from dataset_builder import _merge_feature_candles
    historical = dict(epoch=1_800_000_000, kind='candle', open=100, high=110, low=90, price=101)
    ticks = [dict(epoch=1_800_000_050, price=101), dict(epoch=1_800_000_060, price=102)]
    candles = _merge_feature_candles([historical], ticks)
    assert candles[0]['high'] == 110 and candles[0]['low'] == 90


def test_dataset_training_does_not_apply_live_candle_blocker(tmp_path, monkeypatch):
    """Garante que a construção do dataset não aplica alertas de vela ao vivo (candle_alerts vazio)."""
    import dataset_builder
    source = tmp_path / 'ticks.csv'
    out = tmp_path / 'dataset.csv'
    base = 1_800_000_000
    with source.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(FIELDS)
        for i in range(9000):
            writer.writerow([base + i, '', 'R_100', 100 + i * .001, 'tick', 0, '', '', ''])

    calls = []
    original_get_signal = dataset_builder.strategy.get_signal

    def wrapped_get_signal(*args, **kwargs):
        """Registra candle_alerts recebido e delega ao get_signal original."""
        calls.append(kwargs.get('candle_alerts'))
        return original_get_signal(*args, **kwargs)

    monkeypatch.setattr(dataset_builder.strategy, 'get_signal', wrapped_get_signal)
    build_dataset(source, out)
    assert calls
    assert all(alerts == [] for alerts in calls)


def test_pipeline_waits_for_compatible_model_before_executor(monkeypatch):
    """Garante que o pipeline aguarda um modelo compatível antes de liberar o executor."""
    import pipeline
    calls = {'count': 0}

    def fake_metrics():
        """Retorna None na primeira consulta e métricas elegíveis nas seguintes."""
        calls['count'] += 1
        if calls['count'] == 1:
            return None
        return {'eligible': True}

    monkeypatch.setattr(pipeline, '_load_latest_final_metrics', fake_metrics)
    assert pipeline._wait_for_compatible_model(poll_sec=0)
    assert calls['count'] == 2


def test_pipeline_refreshes_latest_history_even_when_old_candles_exist(tmp_path, monkeypatch):
    """Garante que o pipeline busca o histórico mínimo recente mesmo havendo velas antigas no arquivo."""
    import pipeline
    path = tmp_path / 'ticks.csv'
    base = 1_800_000_000
    with path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(FIELDS)
        for i in range(2000):
            epoch = base + i * config.CANDLE_TIMEFRAME_SEC
            writer.writerow([epoch, '', 'R_100', 100, 'candle',
                             config.CANDLE_TIMEFRAME_SEC, 99, 101, 98])

    monkeypatch.setattr(pipeline, 'TICKS_CSV', str(path))
    monkeypatch.setattr(pipeline, 'SYMBOL', 'R_100')
    assert pipeline._count_candles() == 2000
    assert pipeline._history_fetch_count_for_start(1) == pipeline._minimum_history_candles()


def test_minimum_quality_gate_uses_nonoverlapping_contracts():
    """Garante que o gate de qualidade conta apenas contratos sem sobreposição e reprova amostra pequena."""
    from train_model import _metrics
    class Model:
        """Modelo falso que sempre prevê 0.9 para a classe 1."""
        classes_ = np.array([0, 1])
        def predict_proba(self, X):
            """Retorna [0.1, 0.9] para todas as linhas."""
            return np.tile([.1, .9], (len(X), 1))
    frame = pd.DataFrame(np.zeros((100, len(feature_engine.FEATURES))), columns = feature_engine.FEATURES)
    frame['target'] = 1
    frame['technical_signal'] = 1
    frame['tech_score'] = .9
    frame['entry_epoch'] = np.arange(100) * 60
    frame['expiry_epoch'] = frame.entry_epoch + 900
    result = _metrics(Model(), frame)
    assert result['trades'] == 7 and not result['eligible']


def test_dataset_does_not_use_other_symbols(tmp_path):
    """Garante que o dataset ignora ticks de outros símbolos presentes no CSV."""
    source = tmp_path / 'ticks.csv'
    out = tmp_path / 'dataset.csv'
    write_ticks(source)
    with source.open('a', newline = '') as f:
        w = csv.writer(f)
        for i in range(100):
            w.writerow([1_800_000_000 + i, '', 'other', 100000, 'tick', 0, '', '', ''])
    build_dataset(source, out)
    frame = pd.read_csv(out)
    assert set(frame.symbol) == {'R_100'} and frame.ema9.max() < 200


def test_training_calibration_and_serialization_on_synthetic_data(tmp_path, monkeypatch):
    """Exercita o fluxo novo com quatro árvores e dados artificiais, sem modelos reais."""
    import joblib, train_model, xgboost
    from model_contract import fingerprint, write_json_atomic
    forest = train_model.RandomForestClassifier
    boosted = xgboost.XGBClassifier
    monkeypatch.setattr(train_model, 'RandomForestClassifier', lambda ** kw: forest(n_estimators = 4, max_depth = 2, min_samples_leaf = 2, random_state = 42, n_jobs = 1))
    monkeypatch.setattr(xgboost, 'XGBClassifier', lambda ** kw: boosted(n_estimators = 4, max_depth = 2, random_state = 42, n_jobs = 1))
    n = 4000
    labels = np.arange(n) % 2
    frame = pd.DataFrame(np.zeros((n, len(feature_engine.FEATURES))), columns = feature_engine.FEATURES)
    frame['ema9'] = labels
    frame['target'] = labels
    frame['technical_signal'] = labels
    frame['tech_score'] = .9
    frame['symbol'] = config.get_active_symbol()
    frame['signal_epoch'] = 1_800_000_000 + np.arange(n) * 60
    frame['entry_epoch'] = frame.signal_epoch
    frame['expiry_epoch'] = frame.entry_epoch + 900
    frame['feature_start_epoch'] = frame.signal_epoch - 6000
    source = tmp_path / 'synthetic.csv'
    frame.to_csv(source, index = False)
    write_json_atomic(str(source) + '.metadata.json', dict(specification(config.get_active_symbol()), dataset_sha256 = fingerprint(source)))
    dest = tmp_path / 'candidate.pkl'
    result = train_model.train(source, dest)
    payload = joblib.load(dest)
    assert payload['model'].__class__.__module__ == 'train_model'
    assert result['eligible'] and payload['metadata']['approved']
    assert result['n_test'] == 800 and result['trades'] < 100
    monkeypatch.setattr(config, 'AI_MODEL_PATH', str(dest))
    ai_predictor.reset()
    assert ai_predictor.status()['ready']
    assert ai_predictor.model_id() == fingerprint(dest)
    sidecar = json.loads(Path(str(dest) + '.metadata.json').read_text())
    assert sidecar['artifact_sha256'] == fingerprint(dest)


def test_model_quality_gate_requires_technical_agreement():
    """Garante que o gate de qualidade não aprova nem conta operações sem concordância do sinal técnico."""
    from train_model import _metrics

    class Model:
        """Modelo falso com alta confiança na classe 1."""
        classes_ = np.array([0, 1])

        def predict_proba(self, X):
            """Retorna [0.01, 0.99] para todas as linhas."""
            return np.tile([.01, .99], (len(X), 1))

    frame = pd.DataFrame(np.zeros((100, len(feature_engine.FEATURES))), columns=feature_engine.FEATURES)
    frame['target'] = 1
    frame['tech_score'] = .9
    frame['entry_epoch'] = np.arange(100) * 60
    frame['expiry_epoch'] = frame.entry_epoch + 900
    for technical in (-1, 0):
        frame['technical_signal'] = technical
        result = _metrics(Model(), frame)
        assert result['trades'] == 0
        assert result['eligible'] is False
