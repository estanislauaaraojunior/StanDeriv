"""
dataset_builder.py — gera dataset.csv para treinamento da IA.

Lê ticks.csv (gerado pelo collector.py) e extrai features usando as mesmas
funções de indicators.py que o bot usa em tempo real — garantindo consistência
total entre treino e inferência.

Features extraídas por janela deslizante de 100 ticks:
  Via indicators.py : ema9, ema21, ema_cross, rsi, macd_hist, macd_line,
                      adx, bb_width, bb_position, momentum3
  Estatísticas raw   : return_1, return_3, return_5,
                       volatility_10, volatility_20, high_low_5

Target:
  1 → próximo tick é maior que o tick atual (preço vai subir)
  0 → próximo tick é menor ou igual (preço vai cair)

Uso:
    python dataset_builder.py
    python dataset_builder.py --input ticks.csv --output dataset.csv --window 100
"""

import argparse
import csv
import os
import sys

import pandas as pd

# O builder usa as mesmas funções dos indicadores do bot
import indicators as ind
from config import (
    EMA_FAST, EMA_SLOW,
    RSI_PERIOD,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    ADX_PERIOD,
    BB_PERIOD, BB_STD,
    TICKS_CSV, DATASET_CSV, SYMBOL,
    CANDIDATE_DURATIONS,
    CANDLE_SIZE, PA_SR_TOLERANCE, TARGET_NOISE_THRESHOLD,
    TARGET_LOOKFORWARD, CANDLE_TIMEFRAME_SEC,
)


def _resolve_active_symbol(ticks_df: "pd.DataFrame") -> str:
    """
    Retorna o símbolo ativo, por ordem de prioridade:
      1. Símbolo salvo em state.json (sessão atual do pipeline)
      2. Símbolo mais frequente no ticks.csv (maior volume de dados)
      3. SYMBOL de config.py (fallback)
    """
    # 1. state.json
    try:
        import json as _j
        _state_path = os.path.join(os.path.dirname(os.path.abspath(TICKS_CSV)), "state.json")
        if os.path.exists(_state_path):
            with open(_state_path) as _sf:
                _sym = _j.load(_sf).get("symbol", "")
            if _sym and "symbol" in ticks_df.columns:
                _count = (ticks_df["symbol"] == _sym).sum()
                if _count >= 50:
                    return _sym
    except Exception:
        pass
    # 2. Mais frequente no CSV
    if "symbol" in ticks_df.columns:
        try:
            return str(ticks_df["symbol"].value_counts().idxmax())
        except Exception:
            pass
    # 3. Fallback
    return SYMBOL
from feature_engine import compute_feature_map

# Tamanho mínimo da janela de preços para calcular todos os indicadores
# ADX é o mais exigente: period * 2 + 1 pontos
_MIN_WINDOW = ADX_PERIOD * 2 + 1  # ≈ 29 com ADX_PERIOD=14


def _extract_features(window: list) -> dict | None:
    """Delega para feature_engine.compute_feature_map (fonte única de features)."""
    return compute_feature_map(window)


def build_dataset(
    ticks_path: str,
    output_path: str,
    window_size: int = 100,
    lookforward: int = TARGET_LOOKFORWARD,
) -> int:
    """
    Constrói o dataset e salva em output_path.

    Args:
        lookforward: número de ticks à frente para calcular o target.
                     1 = próximo tick (padrão). N > 1 = média dos próximos N ticks.

    Retorna o número de linhas geradas.
    """
    if not os.path.exists(ticks_path):
        print(f"[ERRO] Arquivo de ticks não encontrado: '{ticks_path}'")
        print("       Execute collector.py primeiro para gerar o arquivo.")
        sys.exit(1)

    print(f"[DATASET] Lendo ticks de '{ticks_path}'...")

    # O collector.py salva com cabeçalho: epoch, datetime, symbol, price
    df = pd.read_csv(ticks_path)

    # Resolve o símbolo ativo: state.json > mais frequente no CSV > config.SYMBOL
    active_symbol = _resolve_active_symbol(df)

    # P6: Filtrar ticks pelo símbolo ativo para evitar contaminação de dados
    if "symbol" in df.columns:
        before = len(df)
        df = df[df["symbol"] == active_symbol]
        dropped = before - len(df)
        if dropped > 0:
            print(f"[DATASET] {dropped:,} ticks de outros símbolos descartados (ativo: {active_symbol}).")

    # Compatível também com formato antigo (sem cabeçalho, 2 colunas: epoch, price)
    if "price" in df.columns:
        prices_raw = df["price"].astype(float).tolist()
    elif df.shape[1] == 2:
        prices_raw = df.iloc[:, 1].astype(float).tolist()
    else:
        # Tenta a 4ª coluna (formato coletor atual: epoch, datetime, symbol, price)
        prices_raw = df.iloc[:, 3].astype(float).tolist()

    # Extrai epochs para agregar em candles de tempo
    epochs_raw = None
    if "epoch" in df.columns:
        try:
            epochs_raw = df["epoch"].astype(int).tolist()
        except Exception:
            epochs_raw = None

    # ── Agregar ticks em candles de CANDLE_TIMEFRAME_SEC segundos ────────────
    # Quando há dados misturados (ticks antigos + closes históricos recentes),
    # o avg_gap global pode ser enganoso. Verifica primeiro os dados mais recentes:
    # se os últimos ~500 pontos têm frequência de candles, usa-os diretamente.
    if epochs_raw is not None and len(epochs_raw) >= 2:
        # Ordena por epoch para garantir ordem cronológica
        if len(epochs_raw) > 1 and epochs_raw[-1] < epochs_raw[0]:
            sorted_pairs = sorted(zip(epochs_raw, prices_raw), key=lambda x: x[0])
            epochs_raw = [e for e, _ in sorted_pairs]
            prices_raw = [p for _, p in sorted_pairs]

        # Verifica avg_gap do segmento recente (todo o dataset — sem cap arbitrário)
        recent_slice = len(epochs_raw)
        recent_epochs = epochs_raw[-recent_slice:]
        recent_avg_gap = (recent_epochs[-1] - recent_epochs[0]) / max(len(recent_epochs) - 1, 1)

        if recent_avg_gap >= CANDLE_TIMEFRAME_SEC * 0.8:
            # Os dados recentes já estão em frequência de candles — usa-os diretamente.
            # Bug #11: o cap anterior era silencioso e fixo em 500 pontos, podendo descartar
            # dados valiosos. Agora usa todos os pontos disponíveis neste segmento.
            prices_all = prices_raw[-recent_slice:]
            print(
                f"[DATASET] Dados recentes em frequência de candles (~{recent_avg_gap:.0f}s/ponto), "
                f"usando {len(prices_all):,} pontos (de {len(prices_raw):,} disponíveis)."
            )
        else:
            # Agrega todos os ticks em candles de tempo
            ticks_list = [{"epoch": e, "price": p} for e, p in zip(epochs_raw, prices_raw)]
            candles_agg = ind.ticks_to_candles_by_time(ticks_list, CANDLE_TIMEFRAME_SEC)
            prices_all = [c["close"] for c in candles_agg]
            print(
                f"[DATASET] {len(prices_raw):,} ticks → {len(prices_all):,} candles "
                f"de {CANDLE_TIMEFRAME_SEC}s ({CANDLE_TIMEFRAME_SEC // 60} min)."
            )
    else:
        prices_all = prices_raw
        print(f"[DATASET] Sem epochs — usando {len(prices_all):,} preços diretamente.")

    n_ticks = len(prices_all)
    print(f"[DATASET] {n_ticks:,} pontos de preço para features.")

    min_needed = window_size + max(1, lookforward)
    if n_ticks < min_needed:
        print(
            f"[ERRO] Ticks insuficientes: {n_ticks} < {min_needed} necessários "
            f"(window={window_size})."
        )
        print("       Continue coletando com collector.py e tente novamente.")
        sys.exit(1)

    rows = []
    skipped = 0

    # Threshold adaptativo: auto-calibra para a volatilidade intrínseca do símbolo.
    # Evita filtrar todos os dados em símbolos de baixa volatilidade (ex: R_10)
    # onde o movimento típico por tick (≈0,002%) é menor que o limite padrão de 0,01%.
    _sample_rets = [
        abs((prices_all[j] - prices_all[j - 1]) / prices_all[j - 1])
        for j in range(1, min(1000, len(prices_all)))
        if prices_all[j - 1] > 0
    ]
    if _sample_rets:
        _sample_rets.sort()
        _median_abs_ret = _sample_rets[len(_sample_rets) // 2]
        # Se o threshold configurado for maior que a mediana dos retornos reais,
        # usa a mediana como teto — preserva filtragem relativa por símbolo
        _adaptive_noise = min(TARGET_NOISE_THRESHOLD, _median_abs_ret * 0.5)
    else:
        _adaptive_noise = TARGET_NOISE_THRESHOLD

    # Margem de lookahead: apenas o lookforward real (CANDIDATE_DURATIONS são minutos,
    # não contagem de candles — a duração ótima é calculada por volatilidade sem lookahead)
    _max_lookahead = max(1, lookforward)

    for i in range(window_size, n_ticks - _max_lookahead):
        window  = prices_all[i - window_size: i + 1]  # window_size+1 preços
        current = prices_all[i]

        # Target multi-tick: média dos próximos `lookforward` ticks
        future_prices = prices_all[i + 1: i + 1 + lookforward]
        if len(future_prices) < lookforward:
            continue
        future_mean = sum(future_prices) / len(future_prices)

        # Threshold dinâmico: volatilidade recente (máximo entre noise_threshold e vol/4)
        def _local_vol() -> float:
            s = window[-21:]
            if len(s) < 2:
                return 0.0
            rets = [(s[j] - s[j-1]) / s[j-1] for j in range(1, len(s)) if s[j-1] != 0]
            if not rets:
                return 0.0
            m = sum(rets) / len(rets)
            return (sum((r - m) ** 2 for r in rets) / len(rets)) ** 0.5

        threshold = max(_adaptive_noise, _local_vol() * 0.25)

        features = _extract_features(window)
        if features is None:
            skipped += 1
            continue

        # Target com threshold adaptativo
        delta_pct = (future_mean - current) / current if current != 0 else 0.0
        if abs(delta_pct) < threshold:
            skipped += 1
            continue
        target = 1 if delta_pct > 0 else 0
        features["target"] = target

        # ── Duração ótima (heurística por volatilidade — sem lookahead) ──
        def _vol_window(n: int) -> float:
            s = window[-n - 1:]
            if len(s) < 2:
                return 0.0
            rets = [(s[j] - s[j - 1]) / s[j - 1]
                    for j in range(1, len(s)) if s[j - 1] != 0]
            if not rets:
                return 0.0
            m = sum(rets) / len(rets)
            return (sum((r - m) ** 2 for r in rets) / len(rets)) ** 0.5

        vol = _vol_window(20)
        n_cands = len(CANDIDATE_DURATIONS)
        # Limiares calibrados para volatilidade de candles de tempo (maior que de ticks)
        if vol > 0.005:
            best_d = CANDIDATE_DURATIONS[0]          # alta vol → duração curta (5m)
        elif vol < 0.001:
            best_d = CANDIDATE_DURATIONS[-1]         # baixa vol → duração longa (30m)
        else:
            best_d = CANDIDATE_DURATIONS[n_cands // 2]  # moderada (15m)

        features["optimal_duration"] = best_d
        rows.append(features)

    if not rows:
        print("[ERRO] Nenhuma linha gerada. Verifique o arquivo de ticks.")
        sys.exit(1)

    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_path, index=False)

    print(f"[DATASET] Horizonte target: {lookforward} tick(s)")
    print(f"[DATASET] Linhas geradas  : {len(rows):,}")
    print(f"[DATASET] Linhas ignoradas: {skipped:,} (indicadores insuficientes)")
    print("[DATASET] Distribuição do target:")
    vc = result_df["target"].value_counts()
    print(f"           Sobe (1): {vc.get(1, 0):>6,} ({vc.get(1, 0)/len(rows)*100:.1f}%)")
    print(f"           Cai  (0): {vc.get(0, 0):>6,} ({vc.get(0, 0)/len(rows)*100:.1f}%)")
    print(f"[DATASET] Salvo em '{output_path}'")

    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gera dataset.csv para treinamento da IA a partir de ticks.csv.",
    )
    parser.add_argument("--input",  default=TICKS_CSV,   help=f"Arquivo de entrada (padrão: {TICKS_CSV})")
    parser.add_argument("--output", default=DATASET_CSV, help=f"Arquivo de saída (padrão: {DATASET_CSV})")
    parser.add_argument(
        "--window", type=int, default=100,
        help="Tamanho da janela deslizante de ticks para calcular features (padrão: 100)",
    )
    parser.add_argument(
        "--lookforward", type=int, default=TARGET_LOOKFORWARD,
        help=(
            "Número de ticks à frente para calcular o target "
            f"(padrão: {TARGET_LOOKFORWARD})"
        ),
    )
    args = parser.parse_args()

    build_dataset(args.input, args.output, args.window, args.lookforward)


if __name__ == "__main__":
    main()
