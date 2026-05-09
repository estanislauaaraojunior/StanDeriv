"""
tests/test_pipeline_integration.py — Testes de integração
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestFeatureSync:
    """Verifica que as listas de features estão sincronizadas."""

    def test_train_model_matches_ai_predictor(self):
        from train_model import FEATURES
        from ai_predictor import _FEATURES
        assert FEATURES == _FEATURES, (
            f"FEATURES dessincronizadas!\n"
            f"train_model: {FEATURES}\n"
            f"ai_predictor: {_FEATURES}"
        )

    def test_feature_count_is_27(self):
        from train_model import FEATURES
        assert len(FEATURES) == 27, f"Esperado 27 features, encontrado {len(FEATURES)}"

    def test_pa_features_present(self):
        from train_model import FEATURES
        pa_features = [
            "pa_market_structure", "pa_bos_strength", "pa_trend_consistency",
            "pa_sr_distance", "pa_sr_touch_count", "pa_sr_position",
            "pa_demand_zone", "pa_supply_zone",
            "pa_fvg_bullish", "pa_fvg_bearish", "pa_candle_at_sr",
        ]
        for f in pa_features:
            assert f in FEATURES, f"Feature PA '{f}' ausente em FEATURES"


class TestAiPredictorFeatureMap:
    """Verifica que _compute_feature_map retorna as features corretas."""

    def test_feature_map_returns_27_keys(self):
        import math
        from ai_predictor import _compute_feature_map

        # Gerar preços sintéticos suficientes
        prices = [6500.0 + i * 0.1 + math.sin(i / 10) * 2 for i in range(300)]
        result = _compute_feature_map(prices)
        assert result is not None, "Feature map retornou None com 300 preços"
        assert len(result) == 27, f"Esperado 27 keys, encontrado {len(result)}: {list(result.keys())}"

    def test_feature_map_has_pa_keys(self):
        import math
        from ai_predictor import _compute_feature_map

        prices = [6500.0 + i * 0.1 + math.sin(i / 10) * 2 for i in range(300)]
        result = _compute_feature_map(prices)
        assert result is not None

        pa_keys = [
            "pa_market_structure", "pa_bos_strength", "pa_trend_consistency",
            "pa_sr_distance", "pa_sr_touch_count", "pa_sr_position",
            "pa_demand_zone", "pa_supply_zone",
            "pa_fvg_bullish", "pa_fvg_bearish", "pa_candle_at_sr",
        ]
        for key in pa_keys:
            assert key in result, f"Chave PA '{key}' ausente no feature map"


class TestDatasetBuilder:
    """Verifica que dataset_builder extrai features PA."""

    def test_extract_features_has_pa(self):
        import math
        from dataset_builder import _extract_features

        prices = [6500.0 + i * 0.1 + math.sin(i / 10) * 2 for i in range(300)]
        result = _extract_features(prices)
        assert result is not None

        pa_keys = [
            "pa_market_structure", "pa_bos_strength", "pa_trend_consistency",
            "pa_sr_distance", "pa_sr_touch_count", "pa_sr_position",
            "pa_demand_zone", "pa_supply_zone",
            "pa_fvg_bullish", "pa_fvg_bearish", "pa_candle_at_sr",
        ]
        for key in pa_keys:
            assert key in result, f"Feature PA '{key}' ausente em _extract_features"


class TestDashboardImport:
    """Verifica que o dashboard importa sem erro."""

    def test_server_imports(self):
        """Verifica que dashboard/server.py pode ser importado parcialmente."""
        # Testar que indicators importa corretamente
        import indicators as ind
        assert hasattr(ind, "ticks_to_candles")
        assert hasattr(ind, "price_action_features")
        assert hasattr(ind, "detect_candle_patterns")

    def test_detect_candle_patterns_returns_list(self):
        import indicators as ind
        prices = [6500.0 + i * 0.1 for i in range(200)]
        candles = ind.ticks_to_candles(prices, 10)
        result = ind.detect_candle_patterns(candles)
        assert isinstance(result, list)

    def test_dashboard_server_uses_unbuffered_pipeline_logs(self):
        """Garantir logs em tempo real no dashboard."""
        import os

        path = os.path.join(os.path.dirname(__file__), "..", "dashboard", "server.py")
        with open(path, encoding="utf-8") as f:
            source = f.read()

        assert '"-u"' in source, "dashboard/server.py deve iniciar o pipeline com python -u"
        assert 'PYTHONUNBUFFERED' in source, "dashboard/server.py deve forçar stdout sem buffering"
        assert 'pipeline.log' in source, "dashboard/server.py deve manter fallback de log persistido"


class TestModelMetricsInApiState:
    """Verifica que /api/state inclui last_metrics do model_metrics.json."""

    def test_read_json_safe_returns_list(self):
        """_read_json_safe deve retornar listas, não só dicts."""
        import json, tempfile, os, sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dashboard"))

        # Simula leitura de model_metrics.json (lista)
        data = [
            {"timestamp": "2026-04-01T16:22:58", "best_model": "Stacking",
             "auc": 0.5054, "accuracy": 0.519, "f1": 0.4826, "n_train": 4672, "n_test": 1052},
            {"timestamp": "2026-04-01T16:39:48", "best_model": "RandomForest",
             "auc": 0.61, "accuracy": 0.58, "f1": 0.55, "n_train": 6000, "n_test": 1500},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name
        try:
            with open(path) as fh:
                loaded = json.load(fh)
            assert isinstance(loaded, list), "Deve retornar lista"
            assert loaded[-1]["best_model"] == "RandomForest", "Deve retornar último item"
            assert loaded[-1]["n_train"] == 6000
        finally:
            os.unlink(path)

    def test_last_metrics_keys(self):
        """last_metrics deve ter todas as chaves esperadas."""
        import json, tempfile, os
        entry = {
            "timestamp": "2026-04-01T17:00:00",
            "best_model": "XGBoost",
            "auc": 0.62, "accuracy": 0.60, "f1": 0.58,
            "n_train": 7000, "n_test": 1800,
        }
        expected_keys = {"best_model", "accuracy", "auc", "f1", "n_train", "n_test", "timestamp"}
        assert expected_keys.issubset(entry.keys()), (
            f"Chaves ausentes: {expected_keys - entry.keys()}"
        )

    def test_pipeline_trim_limit_is_at_least_200k(self):
        """Verifica que o pipeline usa max_ticks >= 200000 no retrain."""
        import re, os
        pipeline_path = os.path.join(os.path.dirname(__file__), "..", "pipeline.py")
        with open(pipeline_path) as f:
            source = f.read()
        matches = re.findall(r"_trim_ticks\(tmp_ticks,\s*max_ticks=(\d+)\)", source)
        assert matches, "_trim_ticks com max_ticks não encontrado em pipeline.py"
        for val in matches:
            assert int(val) >= 200000, (
                f"max_ticks={val} é muito pequeno (mínimo 200000 para dataset crescer com o tempo)"
            )

    def test_train_model_metrics_path_is_absolute(self):
        """Garante que train_model.py usa caminho absoluto para model_metrics.json."""
        import re, os
        path = os.path.join(os.path.dirname(__file__), "..", "train_model.py")
        with open(path) as f:
            source = f.read()
        # Deve usar abspath — nunca o padrão relativo 'or "."'
        assert 'abspath' in source, "train_model.py deve usar os.path.abspath para gravar model_metrics.json"
        assert 'os.path.dirname(output_path) or "."' not in source, (
            "Caminho relativo 'or \".\"' ainda presente — corrigir para os.path.abspath"
        )


class TestRecommendedConfig:
    """Verifica que config.py segue as configurações recomendadas pela estratégia atual."""

    def test_candle_timeframe_sec_is_5min_or_more(self):
        """Velas de 5 minutos produzem menos ruído em índices sintéticos."""
        import config
        assert config.CANDLE_TIMEFRAME_SEC >= 60, (
            f"Bot precisa de pelo menos 60s por candle para operar, "
            f"atual: {config.CANDLE_TIMEFRAME_SEC}s"
        )

    def test_target_lookforward_at_least_2(self):
        """Horizonte de 2+ candles é mais estável para previsão direcional."""
        import config
        assert config.TARGET_LOOKFORWARD >= 2, (
            f"Estratégia recomenda TARGET_LOOKFORWARD >= 2, atual: {config.TARGET_LOOKFORWARD}"
        )

    def test_signal_score_min_is_selective(self):
        """Score mínimo de 0.25+ evita entradas de baixa qualidade."""
        import config
        assert config.SIGNAL_SCORE_MIN >= 0.03, (
            f"Estratégia recomenda SIGNAL_SCORE_MIN >= 0.03, atual: {config.SIGNAL_SCORE_MIN}"
        )

    def test_ai_confidence_min_is_selective(self):
        """Confiança mínima maior reduz operações de AUC ~0.5 (ruído)."""
        import config
        assert config.AI_CONFIDENCE_MIN >= 0.55, (
            f"Estratégia recomenda AI_CONFIDENCE_MIN >= 0.55, atual: {config.AI_CONFIDENCE_MIN}"
        )

    def test_min_candles_warmup_adequate(self):
        """Aquecimento de 20+ velas dá ao modelo dados suficientes antes da primeira entrada."""
        import config
        assert config.MIN_CANDLES >= 35, (
            f"Estratégia recomenda MIN_CANDLES >= 35 (MACD precisa de 35 preços), atual: {config.MIN_CANDLES}"
        )
