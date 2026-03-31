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
)

# Tamanho mínimo da janela de preços para calcular todos os indicadores
# ADX é o mais exigente: period * 2 + 1 pontos
_MIN_WINDOW = ADX_PERIOD * 2 + 1  # ≈ 29 com ADX_PERIOD=14


def _extract_features(window: list) -> dict | None:
    """
    Extrai todas as features de uma janela de preços.

    Retorna dict com as features ou None se dados insuficientes.
    """
    ema9  = ind.ema(window, EMA_FAST)
    ema21 = ind.ema(window, EMA_SLOW)
    rsi_v = ind.rsi(window, RSI_PERIOD)
    macd_ = ind.macd(window, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    adx_v = ind.adx(window, ADX_PERIOD)
    bb_   = ind.bollinger(window, BB_PERIOD, BB_STD)
    mom3  = ind.momentum(window, 3)

    # Qualquer indicador ausente → linha descartada
    if any(v is None for v in [ema9, ema21, rsi_v, macd_, adx_v, mom3]):
        return None

    macd_line, _macd_sig, macd_hist = macd_

    price = window[-1]

    # Distância relativa entre EMAs (normalizada pelo preço atual)
    ema_cross = (ema9 - ema21) / price if price != 0 else 0.0

    # Bollinger: largura e posição relativa do preço
    bb_width    = 0.0
    bb_position = 0.0
    if bb_ is not None:
        bb_upper, bb_mid, bb_lower = bb_
        band_range = bb_upper - bb_lower
        if band_range > 0:
            bb_width    = band_range / bb_mid if bb_mid != 0 else 0.0
            bb_position = (price - bb_lower) / band_range  # 0=lower, 1=upper

    # Retornos simples: (price_t - price_{t-N}) / price_{t-N}
    def _ret(n: int) -> float:
        if len(window) <= n:
            return 0.0
        p_prev = window[-1 - n]
        return (price - p_prev) / p_prev if p_prev != 0 else 0.0

    # Volatilidade: desvio padrão dos retornos tick-a-tick na janela N
    def _vol(n: int) -> float:
        slice_ = window[-n - 1:]
        if len(slice_) < 2:
            return 0.0
        rets = [(slice_[i] - slice_[i - 1]) / slice_[i - 1]
                for i in range(1, len(slice_))
                if slice_[i - 1] != 0]
        if not rets:
            return 0.0
        mean_ = sum(rets) / len(rets)
        var_  = sum((r - mean_) ** 2 for r in rets) / len(rets)
        return var_ ** 0.5

    # High-low spread dos últimos 5 ticks (normalizado)
    last5 = window[-5:]
    high5 = max(last5)
    low5  = min(last5)
    hl5   = (high5 - low5) / price if price != 0 else 0.0

    return {
        # Indicadores técnicos (mesmos do bot)
        "ema9":       ema9,
        "ema21":      ema21,
        "ema_cross":  ema_cross,
        "rsi":        rsi_v,
        "macd_line":  macd_line,
        "macd_hist":  macd_hist,
        "adx":        adx_v,
        "bb_width":   bb_width,
        "bb_position": bb_position,
        "momentum3":  mom3,
        # Estatísticas raw
        "return_1":     _ret(1),
        "return_3":     _ret(3),
        "return_5":     _ret(5),
        "volatility_10": _vol(10),
        "volatility_20": _vol(20),
        "high_low_5":   hl5,
    }


def build_dataset(
    ticks_path: str,
    output_path: str,
    window_size: int = 100,
) -> int:
    """
    Constrói o dataset e salva em output_path.

    Retorna o número de linhas geradas.
    """
    if not os.path.exists(ticks_path):
        print(f"[ERRO] Arquivo de ticks não encontrado: '{ticks_path}'")
        print("       Execute collector.py primeiro para gerar o arquivo.")
        sys.exit(1)

    print(f"[DATASET] Lendo ticks de '{ticks_path}'...")

    # O collector.py salva com cabeçalho: epoch, datetime, symbol, price
    df = pd.read_csv(ticks_path)

    # P6: Filtrar ticks pelo símbolo configurado para evitar contaminação de dados
    if "symbol" in df.columns:
        before = len(df)
        df = df[df["symbol"] == SYMBOL]
        dropped = before - len(df)
        if dropped > 0:
            print(f"[DATASET] {dropped:,} ticks de outros símbolos descartados (esperado: {SYMBOL}).")

    # Compatível também com formato antigo (sem cabeçalho, 2 colunas: epoch, price)
    if "price" in df.columns:
        prices_all = df["price"].astype(float).tolist()
    elif df.shape[1] == 2:
        prices_all = df.iloc[:, 1].astype(float).tolist()
    else:
        # Tenta a 4ª coluna (formato coletor atual: epoch, datetime, symbol, price)
        prices_all = df.iloc[:, 3].astype(float).tolist()

    n_ticks = len(prices_all)
    print(f"[DATASET] {n_ticks:,} ticks carregados.")

    min_needed = window_size + 1  # +1 para o próximo tick (target)
    if n_ticks < min_needed:
        print(
            f"[ERRO] Ticks insuficientes: {n_ticks} < {min_needed} necessários "
            f"(window={window_size})."
        )
        print("       Continue coletando com collector.py e tente novamente.")
        sys.exit(1)

    rows = []
    skipped = 0

    # Margem para o lookahead da duração ótima (ex: 10 ticks além do tick atual)
    _max_lookahead = max(CANDIDATE_DURATIONS)

    for i in range(window_size, n_ticks - _max_lookahead):
        window  = prices_all[i - window_size: i + 1]  # window_size+1 preços
        current = prices_all[i]
        next_p  = prices_all[i + 1]

        features = _extract_features(window)
        if features is None:
            skipped += 1
            continue

        target = 1 if next_p > current else 0
        features["target"] = target

        # ── Duração ótima ──────────────────────────────────────────────
        # Direção base do movimento: +1 sobe, -1 cai
        direction = 1 if next_p > current else -1

        best_d   = CANDIDATE_DURATIONS[0]
        best_abs = 0.0
        for d in CANDIDATE_DURATIONS:
            future_p = prices_all[i + d]
            delta    = (future_p - current) * direction   # positivo = movimento favorável
            if delta > best_abs:
                best_abs = delta
                best_d   = d

        features["optimal_duration"] = best_d
        rows.append(features)

    if not rows:
        print("[ERRO] Nenhuma linha gerada. Verifique o arquivo de ticks.")
        sys.exit(1)

    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_path, index=False)

    print(f"[DATASET] Linhas geradas  : {len(rows):,}")
    print(f"[DATASET] Linhas ignoradas: {skipped:,} (indicadores insuficientes)")
    print(f"[DATASET] Distribuição do target:")
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
    args = parser.parse_args()

    build_dataset(args.input, args.output, args.window)


if __name__ == "__main__":
    main()
