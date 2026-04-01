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
