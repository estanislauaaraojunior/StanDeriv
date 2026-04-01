"""
train_model.py — treinamento do pipeline ML completo.

Modelos treinados:
  1. RandomForestClassifier  (scikit-learn)
  2. XGBClassifier           (xgboost)
  3. StackingClassifier      (RF + XGB como base, LogisticRegression como meta-learner)

O melhor modelo (por ROC-AUC no conjunto de teste) é salvo em model.pkl.
A separação treino/teste é estritamente temporal (sem shuffle) para evitar
data-leakage de séries temporais.

Uso:
    python train_model.py
    python train_model.py --dataset dataset.csv --output model.pkl --test-ratio 0.2
"""

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")  # suprimir avisos de convergência na linha de comando

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config import (
    DATASET_CSV, AI_MODEL_PATH, DURATION_MODEL_PATH, CANDIDATE_DURATIONS,
    TRANSFORMER_MODEL_PATH, USE_TRANSFORMER, TRANSFORMER_SEQ_LEN,
    TRANSFORMER_D_MODEL, TRANSFORMER_N_HEADS, TRANSFORMER_N_LAYERS,
    TRANSFORMER_DROPOUT, TRANSFORMER_EPOCHS, TRANSFORMER_BATCH_SIZE,
    TRANSFORMER_LR, TRANSFORMER_PATIENCE,
)

# Importa TFT (opcional — requer torch; fallback gracioso se não instalado)
try:
    from transformer_model import TFTPredictor
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# Features que o predictor em tempo real também irá calcular (deve estar em sincronia
# com dataset_builder.py e ai_predictor.py)
FEATURES = [
    "ema9", "ema21", "ema_cross",
    "rsi",
    "macd_line", "macd_hist",
    "adx",
    "bb_width", "bb_position",
    "momentum3",
    "return_1", "return_3", "return_5",
    "volatility_10", "volatility_20",
    "high_low_5",
    # Price Action features
    "pa_market_structure", "pa_bos_strength", "pa_trend_consistency",
    "pa_sr_distance", "pa_sr_touch_count", "pa_sr_position",
    "pa_demand_zone", "pa_supply_zone",
    "pa_fvg_bullish", "pa_fvg_bearish", "pa_candle_at_sr",
]


# ─────────────────────────────────────────────────────────────────
#  Construção dos modelos
# ─────────────────────────────────────────────────────────────────

def _build_rf() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])


def _build_xgb(scale_pos_weight: float = 1.0) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )),
    ])


def _build_stacking(scale_pos_weight: float = 1.0) -> StackingClassifier:
    """
    Stacking: RF e XGB como estimadores base, LogisticRegression como meta-learner.
    O StandardScaler do meta-learner é encapsulado em um Pipeline auxiliar.
    """
    base_estimators = [
        ("rf",  _build_rf()),
        ("xgb", _build_xgb(scale_pos_weight)),
    ]
    meta = Pipeline([
        ("scaler", StandardScaler()),
        ("meta",   LogisticRegression(max_iter=1000, random_state=42)),
    ])
    return StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta,
        cv=5,
        passthrough=False,
        n_jobs=-1,
    )


# ─────────────────────────────────────────────────────────────────
#  Avaliação
# ─────────────────────────────────────────────────────────────────

def _evaluate(name: str, model, X_test, y_test) -> dict:
    preds   = model.predict(X_test)
    proba   = model.predict_proba(X_test)[:, 1]
    acc     = accuracy_score(y_test, preds)
    prec    = precision_score(y_test, preds, zero_division=0)
    rec     = recall_score(y_test, preds, zero_division=0)
    f1      = f1_score(y_test, preds, zero_division=0)
    auc     = roc_auc_score(y_test, proba)

    print(f"\n  ─── {name} ───")
    print(f"  Acurácia  : {acc:.4f}")
    print(f"  Precisão  : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")

    return {"name": name, "model": model, "auc": auc, "acc": acc, "f1": f1}


# ─────────────────────────────────────────────────────────────────
#  TFT — Dataset sequencial, treinamento e avaliação
# ─────────────────────────────────────────────────────────────────

def _build_sequence_dataset(X: np.ndarray, seq_len: int) -> np.ndarray:
    """
    Converte array plano (N, n_features) em janelas temporais (N-seq_len, seq_len, n_features).

    Janela i: X[i : i+seq_len] → label y[i+seq_len].
    Os labels correspondentes são y[seq_len:] (alinhados ao tick seguinte da janela).
    """
    N = len(X)
    if N <= seq_len:
        return np.empty((0, seq_len, X.shape[1]), dtype=np.float32)
    seqs = np.stack([X[i : i + seq_len] for i in range(N - seq_len)], axis=0)
    return seqs.astype(np.float32)


def _train_tft(
    X_seq_train: np.ndarray,
    y_dir_train: np.ndarray,
    n_features:  int,
    y_dur_train: "Optional[np.ndarray]" = None,
) -> "TFTPredictor":
    """
    Instancia e treina um TFTPredictor com os hiperparâmetros de config.py.

    y_dur_train deve conter índices de classe (0..n_dur-1), não valores de duração.
    """
    tft = TFTPredictor(
        n_features    = n_features,
        n_dur_classes = len(CANDIDATE_DURATIONS),
        seq_len       = TRANSFORMER_SEQ_LEN,
        d_model       = TRANSFORMER_D_MODEL,
        n_heads       = TRANSFORMER_N_HEADS,
        n_layers      = TRANSFORMER_N_LAYERS,
        dropout       = TRANSFORMER_DROPOUT,
        epochs        = TRANSFORMER_EPOCHS,
        batch_size    = TRANSFORMER_BATCH_SIZE,
        lr            = TRANSFORMER_LR,
        patience      = TRANSFORMER_PATIENCE,
    )
    tft.fit(X_seq_train, y_dir_train, y_dur=y_dur_train)
    return tft


def _evaluate_tft(name: str, tft: "TFTPredictor", X_seq: np.ndarray, y: np.ndarray) -> dict:
    """Avalia um TFTPredictor e imprime métricas comparáveis aos modelos clássicos."""
    try:
        proba  = tft.predict_proba(X_seq)          # (N, 2)
        preds  = (proba[:, 1] >= 0.5).astype(int)
        proba1 = proba[:, 1]

        acc  = accuracy_score(y, preds)
        prec = precision_score(y, preds, zero_division=0)
        rec  = recall_score(y, preds, zero_division=0)
        f1   = f1_score(y, preds, zero_division=0)
        auc  = roc_auc_score(y, proba1) if len(np.unique(y)) > 1 else 0.5

        print(f"\n  ─── {name} ───")
        print(f"  Acurácia  : {acc:.4f}")
        print(f"  Precisão  : {prec:.4f}")
        print(f"  Recall    : {rec:.4f}")
        print(f"  F1        : {f1:.4f}")
        print(f"  ROC-AUC   : {auc:.4f}")

        return {"name": name, "model": tft, "auc": auc, "acc": acc, "f1": f1}
    except Exception as exc:
        print(f"\n  ─── {name} ─── [ERRO na avaliação: {exc}]")
        return {"name": name, "model": tft, "auc": 0.0, "acc": 0.0, "f1": 0.0}


# ─────────────────────────────────────────────────────────────────
#  Importância de features (apenas para RandomForest)
# ─────────────────────────────────────────────────────────────────

def _print_feature_importance(rf_pipeline: Pipeline) -> None:
    rf_clf = rf_pipeline.named_steps["clf"]
    importances = rf_clf.feature_importances_
    pairs = sorted(zip(FEATURES, importances), key=lambda x: x[1], reverse=True)
    print("\n  Top features (RandomForest):")
    for feat, imp in pairs[:8]:
        bar = "█" * int(imp * 40)
        print(f"    {feat:<20} {imp:.4f}  {bar}")


# ─────────────────────────────────────────────────────────────────
#  Treinamento principal
# ─────────────────────────────────────────────────────────────────

def train(dataset_path: str, output_path: str, test_ratio: float, gap_ratio: float = 0.02) -> None:
    if not os.path.exists(dataset_path):
        print(f"[ERRO] Dataset não encontrado: '{dataset_path}'")
        print("       Execute dataset_builder.py primeiro.")
        sys.exit(1)

    print(f"[TREINO] Lendo dataset de '{dataset_path}'...")
    df = pd.read_csv(dataset_path)

    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f"[ERRO] Colunas ausentes no dataset: {missing}")
        print("       Regenere o dataset com dataset_builder.py.")
        sys.exit(1)

    X = df[FEATURES].values
    y = df["target"].values

    n = len(X)
    split = int(n * (1 - test_ratio))

    # P11: Gap temporal entre treino e teste para evitar vazamento de dados
    gap = max(50, int(n * gap_ratio))
    X_train, X_test = X[:split], X[split + gap:]
    y_train, y_test = y[:split], y[split + gap:]

    # Extrair rótulos de duração antecipadamente (usado pelo TFT e modelo de duração clássico)
    has_dur     = "optimal_duration" in df.columns
    y_dur_arr   = df["optimal_duration"].values if has_dur else None
    y_dur_train = y_dur_arr[:split] if has_dur else None
    y_dur_test  = y_dur_arr[split + gap:] if has_dur else None

    print(f"[TREINO] Total de amostras : {n:,}")
    print(f"[TREINO] Treino            : {split:,} ({(1-test_ratio)*100:.0f}%)")
    print(f"[TREINO] Gap temporal      : {gap:,} amostras")
    print(f"[TREINO] Teste             : {len(X_test):,} ({len(X_test)/n*100:.1f}%)")

    # Peso para compensar desequilíbrio de classes
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    spw   = n_neg / n_pos if n_pos > 0 else 1.0
    print(f"[TREINO] Classe 1 (sobe): {n_pos:,} | Classe 0 (cai): {n_neg:,} | SPW={spw:.2f}")

    print("\n[TREINO] Treinando modelos...")

    # 1. RandomForest
    print("  → RandomForest...", end=" ", flush=True)
    rf = _build_rf()
    rf.fit(X_train, y_train)
    print("OK")

    # 2. XGBoost
    print("  → XGBoost...", end=" ", flush=True)
    xgb = _build_xgb(scale_pos_weight=spw)
    xgb.fit(X_train, y_train)
    print("OK")

    # 3. Stacking (treina base+meta internamente com CV)
    print("  → Stacking (RF+XGB+LR meta) — pode demorar...", end=" ", flush=True)
    stk = _build_stacking(scale_pos_weight=spw)
    stk.fit(X_train, y_train)
    print("OK")

    # Avaliação
    print("\n[TREINO] Resultados no conjunto de teste:")
    results = [
        _evaluate("RandomForest", rf,  X_test, y_test),
        _evaluate("XGBoost",      xgb, X_test, y_test),
        _evaluate("Stacking",     stk, X_test, y_test),
    ]

    # Importância de features (RF)
    _print_feature_importance(rf)

    # ── Temporal Fusion Transformer (TFT) ────────────────────────────────
    if USE_TRANSFORMER and _TORCH_AVAILABLE:
        seq_len = TRANSFORMER_SEQ_LEN
        print(f"\n[TREINO] Treinando Temporal Fusion Transformer (seq_len={seq_len}, d_model={TRANSFORMER_D_MODEL})...")

        X_seq_train = _build_sequence_dataset(X_train, seq_len)
        X_seq_test  = _build_sequence_dataset(X_test,  seq_len)
        y_seq_train = y_train[seq_len:]   # label alinhado ao fim de cada janela
        y_seq_test  = y_test[seq_len:]

        if len(X_seq_train) < 10:
            print(f"[TREINO] Dataset insuficiente para TFT ({len(X_seq_train)} amostras de treino) — pulando.")
            print("         Colete mais ticks com collector.py e re-execute.")
        else:
            # Converte valores de duração para índices de classe (0..n_dur-1)
            y_dur_seq = None
            if has_dur and y_dur_train is not None:
                dur_to_idx = {d: i for i, d in enumerate(CANDIDATE_DURATIONS)}
                y_dur_seq  = np.array(
                    [dur_to_idx.get(int(d), 0) for d in y_dur_train[seq_len:]],
                    dtype=np.int64,
                )

            tft        = _train_tft(X_seq_train, y_seq_train, len(FEATURES), y_dur_seq)
            tft_result = _evaluate_tft("TFT (Transformer)", tft, X_seq_test, y_seq_test)

            tft_payload = {
                "model":       tft,
                "features":    FEATURES,
                "name":        "TFT",
                "seq_len":     seq_len,
                "dur_classes": CANDIDATE_DURATIONS,
            }
            joblib.dump(tft_payload, TRANSFORMER_MODEL_PATH)
            print(f"[TREINO] TFT salvo em '{TRANSFORMER_MODEL_PATH}'")

            # Comparação final de todas as arquiteturas
            all_aucs = [
                (r["name"], r["auc"]) for r in results
            ] + [(tft_result["name"], tft_result["auc"])]
            print("\n[TREINO] Comparativo ROC-AUC:")
            for mname, mauc in sorted(all_aucs, key=lambda t: t[1], reverse=True):
                bar = "█" * int(mauc * 40)
                print(f"  {mname:<22} {mauc:.4f}  {bar}")

    elif USE_TRANSFORMER and not _TORCH_AVAILABLE:
        print("\n[TREINO] USE_TRANSFORMER=True mas PyTorch não está instalado.")
        print("         Execute: pip install torch --index-url https://download.pytorch.org/whl/cpu")

    # Melhor modelo por ROC-AUC (apenas entre os modelos clássicos salvo em model.pkl)
    best = max(results, key=lambda r: r["auc"])
    print(f"\n[TREINO] ✔  Melhor modelo: {best['name']} (AUC={best['auc']:.4f})")

    # Salvar
    payload = {
        "model":    best["model"],
        "features": FEATURES,
        "name":     best["name"],
    }
    joblib.dump(payload, output_path)
    print(f"[TREINO] Modelo salvo em '{output_path}'")

    # ── Modelo de duração dinâmica (RF clássico — complementa o TFT) ────────
    if has_dur and y_dur_train is not None:
        print("\n[TREINO] Treinando modelo de duração dinâmica (RF)...")

        dur_model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )),
        ])
        dur_model.fit(X_train, y_dur_train)

        dur_preds = dur_model.predict(X_test)
        dur_acc   = accuracy_score(y_dur_test, dur_preds)
        print(f"[TREINO] Modelo de duração (RF) — Acurácia: {dur_acc:.4f}")
        print(f"[TREINO] Durações candidatas: {CANDIDATE_DURATIONS}")

        dur_payload = {
            "model":     dur_model,
            "features":  FEATURES,
            "durations": CANDIDATE_DURATIONS,
        }
        joblib.dump(dur_payload, DURATION_MODEL_PATH)
        print(f"[TREINO] Modelo de duração (RF) salvo em '{DURATION_MODEL_PATH}'")
    else:
        print("[TREINO] Coluna 'optimal_duration' não encontrada — regenere o dataset antes de retreinar.")

    # Interpretação da acurácia para o usuário
    acc = best["acc"]
    if acc >= 0.60:
        tier = "Excelente para mercado financeiro"
    elif acc >= 0.55:
        tier = "Bom — acima do aleatório com margem útil"
    elif acc >= 0.52:
        tier = "Razoável — mercado muito ruidoso"
    else:
        tier = "Abaixo do esperado — colete mais ticks e retreine"

    print(f"\n  Acurácia final: {acc:.2%} → {tier}")
    print("\n  Próximo passo: python bot.py --demo")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Treina RandomForest + XGBoost + Stacking e salva o melhor modelo.",
    )
    parser.add_argument("--dataset",    default=DATASET_CSV,   help=f"Dataset de entrada (padrão: {DATASET_CSV})")
    parser.add_argument("--output",     default=AI_MODEL_PATH, help=f"Arquivo de saída do modelo (padrão: {AI_MODEL_PATH})")
    parser.add_argument("--test-ratio", type=float, default=0.2, metavar="RATIO", help="Proporção do conjunto de teste, ex: 0.2 (padrão)")
    parser.add_argument("--gap-ratio",  type=float, default=0.02, metavar="RATIO", help="Gap temporal entre treino e teste como fração do dataset (padrão: 0.02)")
    args = parser.parse_args()

    train(args.dataset, args.output, args.test_ratio, args.gap_ratio)


if __name__ == "__main__":
    main()
