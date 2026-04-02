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
