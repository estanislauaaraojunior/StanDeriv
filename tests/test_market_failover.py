"""Market failover without purchases, model mixing or risk resets."""
import json
import threading
from pathlib import Path
from unittest.mock import Mock

import pytest
import pipeline
from payout_quotes import ProposalRecorder


@pytest.mark.parametrize('code,message,expected', [
    ('ContractBuyValidationError', 'Trading is not available from 20:50:00 to 22:00:00.', True),
    ('ContractBuyValidationError', 'Trading on forex contracts with duration less than 5 hours is not available', True),
    ('MarketIsClosed', 'Market closed', True),
    ('ContractBuyValidationError', 'Minimum stake is 5', False),
    ('AuthorizationRequired', 'Authorize first', False),
])
def test_recorder_only_requests_switch_for_market_restriction(tmp_path, code, message, expected):
    """Garante que o recorder só pede troca de mercado para erros de restrição de mercado e nunca compra nem grava cotação."""
    callback, ws = Mock(), Mock()
    recorder = ProposalRecorder(tmp_path/'quotes.csv', 'R_100', 15, 'm', on_unavailable=callback)
    recorder.sample(ws, bearer=True, now=100)
    request = json.loads(ws.send.call_args.args[0])
    recorder.handle({'req_id': request['req_id'], 'error': {'code': code, 'message': message}}, now=101)
    assert callback.called is expected
    assert not (tmp_path/'quotes.csv').exists()
    assert all('buy' not in json.loads(c.args[0]) for c in ws.send.call_args_list)


def test_stale_error_cannot_switch_market(tmp_path):
    """Garante que erro de proposta recebido tarde demais não aciona a troca de mercado."""
    callback, ws = Mock(), Mock()
    recorder = ProposalRecorder(tmp_path/'quotes.csv', 'R_100', 15, 'm', on_unavailable=callback)
    recorder.sample(ws, bearer=True, now=100)
    request = json.loads(ws.send.call_args.args[0])
    recorder.handle({'req_id': request['req_id'], 'error': {'code': 'MarketIsClosed'}}, now=200)
    callback.assert_not_called()


def test_unfavorable_market_requests_switch(monkeypatch):
    """Garante que mercado desfavorável sinaliza o evento de troca de mercado."""
    event = threading.Event()
    monkeypatch.setattr(pipeline, '_market_switch', event)
    monkeypatch.setattr(pipeline, '_runtime_symbol', lambda: 'frxGBPJPY')
    pipeline._request_market_switch_for_unfavorable(
        'frxGBPJPY',
        'no_technical_signal',
        12,
        {'adx': 12.7},
    )
    assert event.is_set()


def test_initial_wait_observes_market_change(monkeypatch):
    """Garante que as esperas iniciais por ticks e por modelo levantam MarketUnavailable quando há troca de mercado."""
    event = threading.Event(); event.set()
    monkeypatch.setattr(pipeline, '_market_switch', event)
    with pytest.raises(pipeline.MarketUnavailable):
        pipeline._wait_for_ticks(200)
    with pytest.raises(pipeline.MarketUnavailable):
        pipeline._wait_for_compatible_model()


def test_compatible_model_fallback_uses_live_approved_symbol(monkeypatch):
    """Fallback exige bundle compatível, ADX recente e catálogo aberto."""
    checked = []

    def load_model(_path, symbol):
        """Simula bundle válido apenas para frxGBPUSD."""
        checked.append(symbol)
        if symbol != 'frxGBPUSD':
            raise ValueError('modelo sem validação compatível')
        return {}, {'symbol': symbol, 'approved': True}

    monkeypatch.setattr(pipeline, 'load_validated_bundle', load_model)
    monkeypatch.setattr(pipeline, 'usable', lambda metadata, symbol: metadata['symbol'] == symbol)
    monkeypatch.setattr(pipeline, '_probe_symbol', lambda symbol: (symbol == 'frxGBPUSD', 'fechado'))
    monkeypatch.setattr(pipeline, '_refresh_active_symbols', lambda: True)
    monkeypatch.setattr(pipeline, '_active_symbols_catalog', {
        symbol: {'market': 'forex', 'underlying_symbol_type': 'forex',
                 'is_synthetic': False, 'exchange_is_open': True,
                 'is_trading_suspended': False}
        for symbol in ('frxEURUSD', 'frxGBPUSD')
    })
    monkeypatch.setattr(pipeline, '_scan_adx_values', {'frxGBPUSD': 25})
    monkeypatch.setattr(pipeline, '_scan_adx_epochs', {'frxGBPUSD': 100})
    monkeypatch.setattr(pipeline, '_adx_is_fresh', lambda _symbol: True)
    monkeypatch.setattr(pipeline, '_scan_group', lambda symbols: {symbol: 1 for symbol in symbols})

    assert pipeline._first_live_symbol_with_compatible_model(['frxEURUSD', 'frxGBPUSD']) == 'frxGBPUSD'
    assert checked == ['frxEURUSD', 'frxGBPUSD']


def test_compatible_model_fallback_rejects_closed_symbol(monkeypatch):
    """Garante que o fallback não escolhe símbolo com mercado fechado, mesmo com modelo compatível."""
    monkeypatch.setattr(
        pipeline,
        'load_validated_bundle',
        lambda _path, symbol: ({}, {'symbol': symbol, 'approved': True}),
    )
    monkeypatch.setattr(pipeline, 'usable', lambda _metadata, _symbol: True)
    monkeypatch.setattr(pipeline, '_probe_symbol', lambda _symbol: (False, 'mercado fechado'))
    monkeypatch.setattr(pipeline, '_refresh_active_symbols', lambda: True)
    monkeypatch.setattr(pipeline, '_active_symbols_catalog', {
        'frxGBPUSD': {'market': 'forex', 'underlying_symbol_type': 'forex',
                      'is_synthetic': False, 'exchange_is_open': False,
                      'is_trading_suspended': False},
    })
    monkeypatch.setattr(pipeline, '_scan_group', lambda symbols: {symbol: 50 for symbol in symbols})

    assert pipeline._first_live_symbol_with_compatible_model(['frxGBPUSD']) == ''


def test_session_cleanup_precedes_symbol_change(monkeypatch):
    """Garante que coletor e retreino são encerrados antes de o novo símbolo ser escolhido e iniciado."""
    order = []
    lock = Mock()
    monkeypatch.setattr(pipeline, 'acquire_execution_lock', lambda _: lock)
    monkeypatch.setattr(pipeline, '_shutdown', threading.Event())
    monkeypatch.setattr(pipeline, '_market_switch', threading.Event())
    monkeypatch.setattr(pipeline, '_model_ready', threading.Event())
    monkeypatch.setattr(pipeline, '_retrain_state', {})
    monkeypatch.setattr(pipeline, '_reset_ai_predictor', lambda: None)
    def session(resources, symbol):
        """Registra a sessão e, na primeira, cria recursos falsos e sinaliza MarketUnavailable."""
        order.append(('session', symbol))
        if symbol is None:
            resources['collector'] = Mock(stop=lambda: order.append('stop'), is_alive=Mock(side_effect=[True, False]),
                                          join=lambda timeout: order.append('joined'))
            resources['retrain'] = Mock(is_alive=Mock(side_effect=[True, False]),
                                       join=lambda timeout: order.append('train_finished'))
            raise pipeline.MarketUnavailable('OLD')
    def compatible(candidates):
        """Verifica que a limpeza terminou e o símbolo antigo foi excluído; não acha compatível."""
        assert order[-1] == 'train_finished'
        assert 'OLD' not in candidates
        return ''
    def select(candidates, *, exclude_symbols=()):
        """Verifica que a limpeza terminou e o símbolo antigo foi excluído da busca ADX."""
        assert order[-1] == 'train_finished'
        assert candidates == ['NEW']
        assert 'OLD' in exclude_symbols
        return 'NEW'
    monkeypatch.setattr(pipeline, '_main_session', session)
    monkeypatch.setattr(pipeline, '_scan_candidates', lambda: ['OLD', 'NEW'])
    monkeypatch.setattr(pipeline, '_first_live_symbol_with_compatible_model', compatible)
    monkeypatch.setattr(pipeline, '_select_strong_market', select)
    pipeline.main()
    assert order == [('session', None), 'stop', 'joined', 'train_finished', ('session', 'NEW')]
    lock.close.assert_called_once()


def test_market_change_prefers_symbol_with_compatible_model(monkeypatch):
    """Garante que a troca de mercado prefere o símbolo com modelo compatível sem usar o fallback ao vivo."""
    order = []
    lock = Mock()
    monkeypatch.setattr(pipeline, 'acquire_execution_lock', lambda _: lock)
    monkeypatch.setattr(pipeline, '_shutdown', threading.Event())
    monkeypatch.setattr(pipeline, '_market_switch', threading.Event())
    monkeypatch.setattr(pipeline, '_model_ready', threading.Event())
    monkeypatch.setattr(pipeline, '_retrain_state', {})
    monkeypatch.setattr(pipeline, '_reset_ai_predictor', lambda: None)

    def session(_resources, symbol):
        """Registra a sessão e sinaliza MarketUnavailable na primeira."""
        order.append(('session', symbol))
        if symbol is None:
            raise pipeline.MarketUnavailable('OLD')

    monkeypatch.setattr(pipeline, '_main_session', session)
    monkeypatch.setattr(pipeline, '_scan_candidates', lambda: ['OLD', 'LIVE_ONLY', 'APPROVED'])
    monkeypatch.setattr(pipeline, '_first_live_symbol_with_compatible_model', lambda candidates: 'APPROVED')
    fallback = Mock(return_value='LIVE_ONLY')
    monkeypatch.setattr(pipeline, '_first_live_symbol', fallback)

    pipeline.main()

    assert order == [('session', None), ('session', 'APPROVED')]
    fallback.assert_not_called()
    lock.close.assert_called_once()


def test_shadow_rejection_switches_to_strong_market_without_market_failure_budget(monkeypatch):
    """Garante que reprovação shadow busca outro mercado forte sem marcar o antigo como indisponível."""
    lock = Mock()
    order = []
    budget = Mock(return_value=True)
    failed = Mock()
    monkeypatch.setattr(pipeline, 'acquire_execution_lock', lambda _: lock)
    monkeypatch.setattr(pipeline, '_shutdown', threading.Event())
    monkeypatch.setattr(pipeline, '_market_switch', threading.Event())
    monkeypatch.setattr(pipeline, '_model_ready', threading.Event())
    monkeypatch.setattr(pipeline, '_retrain_state', {})
    monkeypatch.setattr(pipeline, '_reset_ai_predictor', lambda: None)
    monkeypatch.setattr(pipeline, '_switch_budget_exhausted', budget)
    monkeypatch.setattr(pipeline, '_mark_market_failure', failed)
    monkeypatch.setattr(pipeline, '_runtime_symbol', lambda: 'OLD')
    monkeypatch.setattr(pipeline, '_scan_candidates', lambda: ['OLD', 'NEW'])

    def select(candidates, *, exclude_symbols=()):
        """Retorna somente NEW, confirmando exclusão do bundle reprovado."""
        order.append(('scan', list(candidates), set(exclude_symbols)))
        return 'NEW'

    def session(_resources, symbol):
        """Simula reprovação shadow em OLD e sessão normal em NEW."""
        order.append(('session', symbol))
        if symbol is None:
            pipeline._request_shadow_validation_switch('OLD')
            raise pipeline.MarketUnavailable('OLD')

    monkeypatch.setattr(pipeline, '_select_strong_market', select)
    monkeypatch.setattr(pipeline, '_main_session', session)

    pipeline.main()

    assert order == [('session', None), ('scan', ['NEW'], {'OLD'}), ('session', 'NEW')]
    budget.assert_not_called()
    failed.assert_not_called()
    lock.close.assert_called_once()


def test_market_switch_budget_enters_degraded_state(monkeypatch):
    """Garante que estourar o orçamento de trocas de mercado leva o pipeline ao estado degraded."""
    lock = Mock()
    writes = []
    monkeypatch.setattr(pipeline, 'acquire_execution_lock', lambda _: lock)
    monkeypatch.setattr(pipeline, '_shutdown', threading.Event())
    monkeypatch.setattr(pipeline, '_market_switch', threading.Event())
    monkeypatch.setattr(pipeline, '_model_ready', threading.Event())
    monkeypatch.setattr(pipeline, '_retrain_state', {})
    monkeypatch.setattr(pipeline, '_reset_ai_predictor', lambda: None)
    monkeypatch.setattr(pipeline, 'MARKET_SWITCH_MAX_PER_WINDOW', 1)
    monkeypatch.setattr(pipeline, 'MARKET_SWITCH_WINDOW_SEC', 3600)
    monkeypatch.setattr(pipeline, '_scan_candidates', lambda: ['OLD', 'NEW'])
    monkeypatch.setattr(pipeline, '_write_pipeline_status', lambda **kwargs: writes.append(kwargs))

    def session(_resources, _symbol):
        """Sempre sinaliza MarketUnavailable."""
        raise pipeline.MarketUnavailable('OLD')

    monkeypatch.setattr(pipeline, '_main_session', session)

    pipeline.main()

    assert writes[-1]["status"] == "degraded"
    assert writes[-1]["running"] is False
    lock.close.assert_called_once()


def test_top_trending_symbols_selects_best_live_markets(monkeypatch):
    """Prioriza não sintéticos e depois ordena por score entre os ativos elegíveis."""
    monkeypatch.setattr(pipeline, '_ADX_TREND_MIN', 30.0)

    scores = {'A': 18.0, 'B': 40.0, 'C': 9.0, 'D': 30.0, 'E': 25.0, 'F': 12.0, 'G': 50.0}
    monkeypatch.setattr(pipeline, '_scan_adx_values', {'A': 31.0, 'B': 35.0, 'C': 29.0,
                                                       'D': 40.0, 'E': 20.0, 'F': 35.0, 'G': 50.0})
    monkeypatch.setattr(pipeline, '_scan_adx_epochs', {symbol: 100 for symbol in scores})
    monkeypatch.setattr(pipeline, '_adx_is_fresh', lambda _symbol: True)
    monkeypatch.setattr(pipeline, '_active_symbols_catalog', {
        symbol: {'market': 'forex' if symbol in {'A', 'B', 'C'} else 'synthetic_index',
                 'underlying_symbol_type': 'forex' if symbol in {'A', 'B', 'C'} else 'synthetic_index',
                 'is_synthetic': symbol not in {'A', 'B', 'C'},
                 'exchange_is_open': True, 'is_trading_suspended': False}
        for symbol in scores
    })
    monkeypatch.setattr(pipeline, '_refresh_active_symbols', lambda: True)
    monkeypatch.setattr(pipeline, '_catalog_scan_candidates', lambda: list(scores))
    monkeypatch.setattr(pipeline, '_scan_group', lambda candidates: {symbol: scores[symbol] for symbol in candidates})
    monkeypatch.setattr(pipeline, '_probe_symbol', lambda symbol: (symbol != 'D', 'fechado'))

    assert pipeline._detect_top_trending_symbols(limit=5) == ['B', 'A']


def test_top_trending_symbols_falls_back_to_open_eligible_synthetic(monkeypatch):
    """Sintético aberto com ADX 25 é usado somente quando não há não sintético elegível."""
    monkeypatch.setattr(pipeline, '_refresh_active_symbols', lambda: True)
    monkeypatch.setattr(pipeline, '_catalog_scan_candidates', lambda: ['FOREX', 'SYNTH'])
    monkeypatch.setattr(pipeline, '_active_symbols_catalog', {
        'FOREX': {'market': 'forex', 'underlying_symbol_type': 'forex',
                  'is_synthetic': False, 'exchange_is_open': True, 'is_trading_suspended': False},
        'SYNTH': {'market': 'synthetic_index', 'underlying_symbol_type': 'synthetic_index',
                  'is_synthetic': True, 'exchange_is_open': True, 'is_trading_suspended': False},
    })
    scanned = []

    def scan(symbols):
        scanned.extend(symbols)
        return {symbol: 1 for symbol in symbols}

    monkeypatch.setattr(pipeline, '_scan_group', scan)
    monkeypatch.setattr(pipeline, '_scan_adx_values', {'FOREX': 24.99, 'SYNTH': 25.0})
    monkeypatch.setattr(pipeline, '_scan_adx_epochs', {'FOREX': 100, 'SYNTH': 100})
    monkeypatch.setattr(pipeline, '_adx_is_fresh', lambda _symbol: True)
    monkeypatch.setattr(pipeline, '_probe_symbol', lambda _symbol: (True, ''))

    assert pipeline._detect_top_trending_symbols(limit=1) == ['SYNTH']
    assert scanned == ['FOREX', 'SYNTH']


def test_training_rank_prefers_eligible_expected_return():
    """Garante que o ranking de treino prefere candidato elegível com maior expectativa de retorno."""
    rejected = {'eligible': False, 'expectancy': 99, 'win_rate': 1, 'auc': 1, 'f1': 1, 'trades': 500}
    approved_low = {'eligible': True, 'expectancy': 0.02, 'win_rate': .55, 'auc': .6, 'f1': .5, 'trades': 20}
    approved_high = {'eligible': True, 'expectancy': 0.08, 'win_rate': .52, 'auc': .58, 'f1': .48, 'trades': 12}

    assert max([rejected, approved_low, approved_high], key=pipeline._training_rank_key) is approved_high


def test_symbol_order_uses_prior_opportunity_history():
    """Garante que o torneio prioriza símbolos com histórico de entradas simuladas."""
    history = {
        'AAA': (0, 0.0, 2, 1.0, .55),
        'CCC': (0, 0.0, 20, 8.0, .50),
    }
    assert pipeline._order_symbols_by_opportunity(['AAA', 'BBB', 'CCC'], history) == ['CCC', 'AAA', 'BBB']


def test_market_opportunity_history_reads_candidate_metadata(monkeypatch, tmp_path):
    """Garante que metadados de candidatos alimentam o histórico por símbolo."""
    model_path = tmp_path / 'model.pkl'
    candidate = tmp_path / '.training_candidates' / 'attempt-1'
    candidate.mkdir(parents=True)
    metadata = {
        'symbol': 'R_50',
        'metrics': {
            'trades': 0,
            'auc': .55,
            'uncalibrated_metrics': {'trades': 12, 'win_rate': .5},
            'approval_diagnostics': {'component_trades': {'ai_only': 7, 'combined': 0}},
        },
    }
    (candidate / 'model.pkl.metadata.json').write_text(json.dumps(metadata))
    monkeypatch.setattr(pipeline, 'AI_MODEL_PATH', str(model_path))
    history = pipeline._load_market_opportunity_history()
    assert history['R_50'][2] == 12


def test_initial_symbol_tournament_promotes_best_approved(monkeypatch, tmp_path):
    """Garante que o torneio inicial promove o modelo do melhor símbolo aprovado e encerra todos os coletores."""
    active = {'symbol': 'R_100'}
    started = []
    stopped = []
    trained = []
    model_path = tmp_path / 'model.pkl'

    class Collector:
        """Coletor falso que registra início e parada por símbolo."""
        def __init__(self, symbol):
            """Guarda o símbolo e inicia sem erro fatal."""
            self.symbol = symbol
            self.fatal_error = ''
        def start(self):
            """Registra o início do coletor."""
            started.append(self.symbol)
        def stop(self):
            """Registra a parada do coletor."""
            stopped.append(self.symbol)
        def is_alive(self):
            """Informa que a thread já terminou."""
            return False
        def join(self, timeout=None):
            """Não aguarda nada."""
            return None

    def set_symbol(symbol):
        """Atualiza o símbolo ativo simulado."""
        active['symbol'] = symbol

    def run_training(*, dataset_path, model_path, **_kwargs):
        """Simula o treino gravando modelo e metadados e devolvendo métricas do símbolo ativo."""
        symbol = active['symbol']
        trained.append((symbol, dataset_path, model_path))
        Path(model_path).write_text(f'model-{symbol}')
        Path(model_path + '.metadata.json').write_text(f'metadata-{symbol}')
        metrics = {
            'LOW': {'eligible': True, 'expectancy': 0.01, 'win_rate': .60, 'auc': .55, 'f1': .40, 'trades': 10},
            'BEST': {'eligible': True, 'expectancy': 0.09, 'win_rate': .58, 'auc': .62, 'f1': .50, 'trades': 8},
            'BAD': {'eligible': False, 'expectancy': 1.0, 'win_rate': 1.0, 'auc': 1.0, 'f1': 1.0, 'trades': 99, 'rejection_reasons': ['reprovado']},
        }[symbol]
        return True, metrics

    monkeypatch.setattr(pipeline, 'AI_MODEL_PATH', str(model_path))
    monkeypatch.setattr(pipeline, 'DATASET_CSV', str(tmp_path / 'dataset.csv'))
    monkeypatch.setattr(pipeline, 'TRANSFORMER_MODEL_PATH', str(tmp_path / 'transformer.pkl'))
    monkeypatch.setattr(pipeline, 'DURATION_MODEL_PATH', str(tmp_path / 'duration.pkl'))
    monkeypatch.setattr(pipeline, '_runtime_symbol', lambda: active['symbol'])
    monkeypatch.setattr(pipeline, '_set_runtime_symbol', set_symbol)
    monkeypatch.setattr(pipeline, '_ensure_ticks_header', lambda: None)
    monkeypatch.setattr(pipeline, '_CollectorThread', Collector)
    monkeypatch.setattr(pipeline, '_fetch_historical_ticks', lambda _count: 0)
    monkeypatch.setattr(pipeline, '_wait_for_ticks', lambda _min_ticks, collector=None: None)
    monkeypatch.setattr(pipeline, '_run_training', run_training)
    monkeypatch.setattr(pipeline, '_reset_ai_predictor', lambda: None)
    scans = []
    def next_market_batch(**kwargs):
        """Retorna o lote seguinte e registra que o símbolo reprovado foi excluído."""
        scans.append({**kwargs, 'exclude_symbols': set(kwargs['exclude_symbols'])})
        return ['LOW', 'BEST']
    monkeypatch.setattr(pipeline, '_detect_top_trending_symbols', next_market_batch)

    winner, metrics = pipeline._run_initial_symbol_tournament(
        ['BAD'],
        history_count=100,
        min_ticks=10,
        resources={},
    )

    assert winner == 'BEST'
    assert metrics['expectancy'] == 0.09
    assert active['symbol'] == 'BEST'
    assert model_path.read_text() == 'model-BEST'
    assert Path(str(model_path) + '.metadata.json').read_text() == 'metadata-BEST'
    assert started == ['BAD', 'LOW', 'BEST']
    assert stopped == ['BAD', 'LOW', 'BEST']
    assert [item[0] for item in trained] == ['BAD', 'LOW', 'BEST']
    assert scans[0]['exclude_symbols'] == {'BAD'}


def test_main_session_passes_top_symbols_limit(monkeypatch, tmp_path):
    """Garante que _main_session repassa --top-symbols como limite da detecção e inicia coletor e bot."""
    captured = {}
    monkeypatch.setattr(pipeline.sys, 'argv', ['pipeline.py', '--demo', '--top-symbols', '3'])
    monkeypatch.setattr(pipeline, '_validate_deriv_token', lambda demo: True)

    def detect_top(no_scan=False, limit=0):
        """Captura o limite recebido e devolve um único símbolo."""
        captured['limit'] = limit
        return ['AAA']

    monkeypatch.setattr(pipeline, '_detect_top_trending_symbols', detect_top)
    monkeypatch.setattr(pipeline, '_ensure_operable_symbol', lambda symbol: symbol)
    monkeypatch.setattr(pipeline, '_set_runtime_symbol', lambda symbol: captured.setdefault('runtime', symbol))
    monkeypatch.setattr(pipeline, '_prepare_training_start', lambda **_kwargs: None)
    monkeypatch.setattr(pipeline, '_run_initial_symbol_tournament', lambda candidates, **_kwargs: ('AAA', {'eligible': True}))
    monkeypatch.setattr(pipeline, '_banner', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline, '_ensure_ticks_header', lambda: None)
    monkeypatch.setattr(pipeline, '_fetch_historical_ticks', lambda _count: 0)
    monkeypatch.setattr(pipeline, '_count_ticks', lambda: 0)
    monkeypatch.setattr(pipeline, '_load_latest_final_metrics', lambda: {'eligible': True})
    monkeypatch.setattr(pipeline, '_uses_bearer_auth', lambda: False)

    class Collector:
        """Coletor falso que apenas registra o símbolo recebido."""
        fatal_error = ''
        def __init__(self, symbol):
            """Registra o símbolo do coletor criado."""
            captured['collector'] = symbol
        def start(self):
            """Não inicia nada."""
            return None

    class Bot:
        """Bot falso que apenas registra a execução."""
        def __init__(self, **_kwargs):
            """Inicializa sem execução registrada."""
            self.run_called = False
        def run(self):
            """Registra que o bot foi executado."""
            captured['bot_run'] = True

    monkeypatch.setattr(pipeline, '_CollectorThread', Collector)
    monkeypatch.setattr(pipeline, 'RiskManager', lambda initial_balance: Mock())
    monkeypatch.setattr(pipeline, 'DerivBot', Bot)
    monkeypatch.setattr(pipeline, '_check_market_switch', lambda: None)
    monkeypatch.setattr(pipeline, '_retrain_loop', lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline.threading, 'Thread', lambda *args, **kwargs: Mock(start=lambda: None))
    monkeypatch.setattr(pipeline, 'TICKS_CSV', str(tmp_path / 'ticks.csv'))

    pipeline._main_session({'lock': Mock()})

    assert captured['limit'] == 3
    assert captured['collector'] == 'AAA'
    assert captured['bot_run'] is True


def test_main_session_rejects_invalid_top_symbols(monkeypatch):
    """Garante que --top-symbols 0 encerra _main_session com SystemExit."""
    monkeypatch.setattr(pipeline.sys, 'argv', ['pipeline.py', '--demo', '--top-symbols', '0'])
    with pytest.raises(SystemExit):
        pipeline._main_session({'lock': Mock()})


def test_initial_training_failure_starts_only_bootstrap_retrain(monkeypatch, tmp_path):
    """Garante que falha no treino inicial inicia apenas o retreino de bootstrap e ainda executa o bot."""
    started_threads = []
    captured = {'symbol': 'R_100'}

    class FakeThread:
        """Thread falsa que apenas registra as inicializações."""
        def __init__(self, *args, **kwargs):
            """Guarda os kwargs e o nome da thread."""
            self.kwargs = kwargs
            self.name = kwargs.get('name')
        def start(self):
            """Registra a thread iniciada."""
            started_threads.append(self)
        def join(self, timeout=None):
            """Thread falsa termina imediatamente."""
            return None

    class Collector:
        """Coletor falso sempre ativo e sem erro fatal."""
        fatal_error = ''
        def __init__(self, symbol):
            """Guarda o símbolo recebido."""
            self.symbol = symbol
        def start(self):
            """Não inicia nada."""
            return None
        def stop(self):
            """Não para nada."""
            return None
        def is_alive(self):
            """Informa que o coletor está ativo."""
            return True

    class Bot:
        """Bot falso que apenas registra a execução."""
        def __init__(self, **_kwargs):
            """Ignora os argumentos."""
            pass
        def run(self):
            """Registra que o bot foi executado."""
            captured['bot_run'] = True

    monkeypatch.setattr(pipeline.sys, 'argv', ['pipeline.py', '--demo', '--top-symbols', '1'])
    monkeypatch.setattr(pipeline, 'TICKS_CSV', str(tmp_path / 'ticks.csv'))
    monkeypatch.setattr(pipeline, 'AI_MODEL_PATH', str(tmp_path / 'model.pkl'))
    monkeypatch.setattr(pipeline, '_validate_deriv_token', lambda demo: True)
    monkeypatch.setattr(pipeline, '_detect_top_trending_symbols', lambda **_kwargs: ['R_100'])
    monkeypatch.setattr(pipeline, '_ensure_operable_symbol', lambda symbol: symbol)
    monkeypatch.setattr(pipeline, '_set_runtime_symbol', lambda symbol: captured.update(symbol=symbol))
    monkeypatch.setattr(pipeline, '_runtime_symbol', lambda: captured['symbol'])
    monkeypatch.setattr(pipeline, '_prepare_training_start', lambda **_kwargs: None)
    monkeypatch.setattr(pipeline, '_run_initial_symbol_tournament', lambda candidates, **_kwargs: ('', None))
    monkeypatch.setattr(pipeline, '_banner', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline, '_ensure_ticks_header', lambda: None)
    monkeypatch.setattr(pipeline, '_CollectorThread', Collector)
    monkeypatch.setattr(pipeline, '_fetch_historical_ticks', lambda _count: 0)
    monkeypatch.setattr(pipeline, '_count_ticks', lambda: 200)
    monkeypatch.setattr(pipeline, '_wait_for_ticks', lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, '_run_training', lambda **_kwargs: (True, {'eligible': False, 'rejection_reasons': ['x']}))
    monkeypatch.setattr(pipeline, '_should_promote_model', lambda current, challenger: (False, 'x'))
    monkeypatch.setattr(pipeline, '_load_latest_final_metrics', lambda: None)
    monkeypatch.setattr(pipeline, '_wait_for_compatible_model', lambda collector=None: True)
    monkeypatch.setattr(pipeline, '_uses_bearer_auth', lambda: False)
    monkeypatch.setattr(pipeline, 'RiskManager', lambda initial_balance: Mock())
    monkeypatch.setattr(pipeline, 'DerivBot', Bot)
    monkeypatch.setattr(pipeline, '_check_market_switch', lambda: None)
    monkeypatch.setattr(pipeline.threading, 'Thread', FakeThread)

    pipeline._main_session({'lock': Mock()})

    assert [thread.name for thread in started_threads] == ['RetrainBootstrap']
    assert started_threads[0].kwargs.get('kwargs') == {'bootstrap_until_ready': True}
    assert captured['bot_run'] is True


def test_ai_mode_selects_one_trending_symbol_before_incremental_history(monkeypatch, tmp_path):
    """Garante que IA seleciona o melhor ativo e só então atualiza o histórico incremental dele."""
    calls = []
    captured = {'symbol': 'R_100', 'bot_run': False}

    class FakeThread:
        """Thread falsa para impedir loops reais de retreino."""
        def __init__(self, *args, **kwargs):
            """Guarda somente o nome da thread."""
            self.name = kwargs.get('name')
        def start(self):
            """Não executa thread real."""
            return None

    class Collector:
        """Coletor falso sempre vivo."""
        fatal_error = ''
        def __init__(self, symbol):
            """Guarda o símbolo recebido."""
            calls.append(('collector', symbol))
        def start(self):
            """Não inicia conexão real."""
            return None
        def stop(self):
            """Não encerra conexão real."""
            return None
        def is_alive(self):
            """Informa que o coletor está ativo."""
            return True

    class Bot:
        """Bot falso que registra execução sem conectar na Deriv."""
        def __init__(self, **_kwargs):
            """Ignora argumentos."""
            pass
        def run(self):
            """Registra a execução."""
            captured['bot_run'] = True

    def wait_symbols(**kwargs):
        """Confirma seleção de apenas um ativo por tendência."""
        calls.append(('wait_symbols', kwargs))
        return ['JD75']

    def set_symbol(symbol):
        """Atualiza o símbolo runtime simulado."""
        captured['symbol'] = symbol
        calls.append(('set_symbol', symbol))

    def fetch_history(_count):
        """Confirma que o histórico é buscado após a seleção do ativo."""
        calls.append(('history', captured['symbol']))
        return 1

    monkeypatch.setattr(pipeline.sys, 'argv', ['pipeline.py', '--demo', '--top-symbols', '5'])
    monkeypatch.setattr(pipeline, 'TICKS_CSV', str(tmp_path / 'ticks.csv'))
    monkeypatch.setattr(pipeline, 'AI_MODEL_PATH', str(tmp_path / 'model.pkl'))
    monkeypatch.setattr(pipeline, '_validate_deriv_token', lambda demo: True)
    monkeypatch.setattr(pipeline, '_wait_for_strong_training_symbols', wait_symbols)
    monkeypatch.setattr(pipeline, '_ensure_operable_symbol', lambda symbol: symbol)
    monkeypatch.setattr(pipeline, '_run_initial_symbol_tournament', Mock(side_effect=AssertionError('torneio não deve rodar')))
    monkeypatch.setattr(pipeline, '_set_runtime_symbol', set_symbol)
    monkeypatch.setattr(pipeline, '_runtime_symbol', lambda: captured['symbol'])
    monkeypatch.setattr(pipeline, '_prepare_training_start', lambda **_kwargs: None)
    monkeypatch.setattr(pipeline, '_banner', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline, '_ensure_ticks_header', lambda: None)
    monkeypatch.setattr(pipeline, '_CollectorThread', Collector)
    monkeypatch.setattr(pipeline, '_fetch_historical_ticks', fetch_history)
    monkeypatch.setattr(pipeline, '_count_ticks', lambda: 500)
    monkeypatch.setattr(pipeline, '_wait_for_ticks', lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, '_run_training', lambda **_kwargs: (True, {'eligible': False, 'rejection_reasons': ['x']}))
    monkeypatch.setattr(pipeline, '_should_promote_model', lambda current, challenger: (False, 'x'))
    monkeypatch.setattr(pipeline, '_load_latest_final_metrics', lambda: None)
    monkeypatch.setattr(pipeline, '_wait_for_compatible_model', lambda collector=None: True)
    monkeypatch.setattr(pipeline, '_uses_bearer_auth', lambda: False)
    monkeypatch.setattr(pipeline, 'RiskManager', lambda initial_balance: Mock())
    monkeypatch.setattr(pipeline, 'DerivBot', Bot)
    monkeypatch.setattr(pipeline, '_check_market_switch', lambda: None)
    monkeypatch.setattr(pipeline.threading, 'Thread', FakeThread)

    pipeline._main_session({'lock': Mock()})

    assert ('wait_symbols', {'no_scan': False, 'limit': pipeline._TOP_TRENDING_SYMBOLS}) in calls
    assert ('set_symbol', 'JD75') in calls
    assert ('history', 'JD75') in calls
    assert captured['bot_run'] is True


def test_pending_contract_without_model_starts_bootstrap_retrain(monkeypatch, tmp_path):
    """Garante que contrato pendente não impede bootstrap quando não há modelo compatível."""
    started_threads = []
    captured = {'symbol': 'R_100', 'bot_run': False, 'waited_model': False}

    class FakeThread:
        """Thread falsa que apenas registra as inicializações."""
        def __init__(self, *args, **kwargs):
            """Guarda os kwargs e o nome da thread."""
            self.kwargs = kwargs
            self.name = kwargs.get('name')
        def start(self):
            """Registra a thread iniciada."""
            started_threads.append(self)

    class Collector:
        """Coletor falso sempre ativo e sem erro fatal."""
        fatal_error = ''
        def __init__(self, symbol):
            """Guarda o símbolo recebido."""
            self.symbol = symbol
        def start(self):
            """Não inicia nada."""
            return None
        def stop(self):
            """Não para nada."""
            return None
        def is_alive(self):
            """Informa que o coletor está ativo."""
            return True

    class Bot:
        """Bot falso que apenas registra a execução."""
        def __init__(self, **_kwargs):
            """Ignora os argumentos."""
            pass
        def run(self):
            """Registra que o bot foi executado."""
            captured['bot_run'] = True

    state = tmp_path / 'state.json'
    state.write_text(json.dumps({'active_contract': {'symbol': 'R_100'}}))

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pipeline.sys, 'argv', ['pipeline.py', '--demo', '--top-symbols', '1'])
    monkeypatch.setattr(pipeline, 'TICKS_CSV', str(tmp_path / 'ticks.csv'))
    monkeypatch.setattr(pipeline, 'AI_MODEL_PATH', str(tmp_path / 'model.pkl'))
    monkeypatch.setattr(pipeline, '_BOT_STATUS_JSON', tmp_path / 'bot_status.json')
    monkeypatch.setattr(pipeline, '_validate_deriv_token', lambda demo: True)
    monkeypatch.setattr(pipeline, '_set_runtime_symbol', lambda symbol: captured.update(symbol=symbol))
    monkeypatch.setattr(pipeline, '_runtime_symbol', lambda: captured['symbol'])
    monkeypatch.setattr(pipeline, '_prepare_training_start', lambda **_kwargs: None)
    monkeypatch.setattr(pipeline, '_banner', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline, '_ensure_ticks_header', lambda: None)
    monkeypatch.setattr(pipeline, '_CollectorThread', Collector)
    monkeypatch.setattr(pipeline, '_fetch_historical_ticks', lambda _count: 0)
    monkeypatch.setattr(pipeline, '_count_ticks', lambda: 200)
    monkeypatch.setattr(pipeline, '_load_latest_final_metrics', lambda: None)
    monkeypatch.setattr(pipeline, '_wait_for_compatible_model', lambda collector=None: captured.update(waited_model=True) or True)
    monkeypatch.setattr(pipeline, '_uses_bearer_auth', lambda: False)
    monkeypatch.setattr(pipeline, 'RiskManager', lambda initial_balance: Mock())
    monkeypatch.setattr(pipeline, 'DerivBot', Bot)
    monkeypatch.setattr(pipeline, '_check_market_switch', lambda: None)
    monkeypatch.setattr(pipeline.threading, 'Thread', FakeThread)

    pipeline._main_session({'lock': Mock()})

    assert [thread.name for thread in started_threads] == ['RetrainBootstrap']
    assert started_threads[0].kwargs.get('kwargs') == {'bootstrap_until_ready': True}
    assert captured['bot_run'] is True
    assert captured['waited_model'] is False


def test_bootstrap_retrain_requests_market_switch_after_repeated_rejections(monkeypatch, tmp_path):
    """Garante que o retreino de bootstrap pede troca de mercado após rejeições repetidas do candidato."""
    calls = {"training": 0, "switch": []}
    event = threading.Event()

    monkeypatch.setattr(pipeline, "_shutdown", threading.Event())
    monkeypatch.setattr(pipeline, "_market_switch", event)
    monkeypatch.setattr(pipeline, "BOOTSTRAP_MAX_FAILED_PROMOTIONS", 3)
    monkeypatch.setattr(pipeline, "RETRAIN_NEW_LABELS_TRIGGER", 1)
    monkeypatch.setattr(pipeline, "TICKS_CSV", str(tmp_path / "ticks.csv"))
    monkeypatch.setattr(pipeline, "AI_MODEL_PATH", str(tmp_path / "model.pkl"))
    monkeypatch.setattr(pipeline, "_runtime_symbol", lambda: "R_100")
    monkeypatch.setattr(pipeline, "_count_ticks", lambda: 500)
    monkeypatch.setattr(pipeline, "_count_labelable_contracts_for_training", lambda: 9999)
    monkeypatch.setattr(pipeline, "minimum_training_rows", lambda: 100)
    monkeypatch.setattr(pipeline, "_load_latest_final_metrics", lambda: None)
    monkeypatch.setattr(pipeline, "_should_promote_model", lambda current, challenger: (False, "reprovado"))
    monkeypatch.setattr(pipeline, "_request_market_switch_for_unfavorable", lambda *args: (calls["switch"].append(args), event.set()))

    def fake_sleep(_seconds):
        """Encerra o pipeline quando a troca de mercado já foi sinalizada."""
        if event.is_set():
            pipeline._shutdown.set()

    def fake_run_training(**kwargs):
        """Conta o treino e devolve candidato inelegível."""
        calls["training"] += 1
        Path(kwargs['model_path']).write_text("candidate")
        return True, {"eligible": False, "rejection_reasons": ["reprovado"]}

    monkeypatch.setattr(pipeline.time, "sleep", fake_sleep)
    monkeypatch.setattr(pipeline, "_run_training", fake_run_training)

    pipeline._retrain_state.update(last_epoch=0, last_tick_count=0, last_label_count=0, bootstrap_failed_promotions=0)
    pipeline._retrain_loop(bootstrap_until_ready=True)

    assert calls["training"] == 3
    assert calls["switch"]
    assert calls["switch"][-1][:3] == ("R_100", "bootstrap_model_not_promoted", 3)


def test_active_model_requires_matching_approved_shadow_before_retrain_promotion(monkeypatch, tmp_path):
    """Garante que aprovação shadow de outro bundle/símbolo/timeframe não libera promoção."""
    import config
    model_path = tmp_path / 'model.pkl'
    shadow_path = tmp_path / 'shadow.json'
    model_path.write_bytes(b'approved model fixture')
    model_id = pipeline.fingerprint(model_path)
    metadata = {
        'approved': True,
        'symbol': 'R_100',
        'artifact_sha256': model_id,
    }
    (tmp_path / 'model.pkl.metadata.json').write_text(json.dumps(metadata))
    shadow = {
        'schema_version': 1,
        'symbol': 'R_100',
        'model_id': model_id,
        'timeframe_sec': config.CANDLE_TIMEFRAME_SEC,
        'duration_sec': 900,
        'approved_at_epoch': None,
    }
    shadow_path.write_text(json.dumps(shadow))
    monkeypatch.setattr(pipeline, 'AI_MODEL_PATH', str(model_path))
    monkeypatch.setattr(pipeline, '_SHADOW_STATE_JSON', shadow_path)
    monkeypatch.setattr(pipeline, '_runtime_symbol', lambda: 'R_100')

    assert not pipeline._active_model_shadow_approved()
    shadow['approved_at_epoch'] = 1_800_000_000
    shadow_path.write_text(json.dumps(shadow))
    assert pipeline._active_model_shadow_approved()
    shadow['timeframe_sec'] = 60
    shadow_path.write_text(json.dumps(shadow))
    assert not pipeline._active_model_shadow_approved()
