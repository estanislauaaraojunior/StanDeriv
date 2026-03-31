"""
transformer_model.py — Temporal Fusion Transformer (TFT) para séries de ticks.

Arquitetura nível hedge fund com:
  - Variable Selection Network (VSN): aprende importância de cada feature por timestep
  - Gated Residual Network (GRN): regularização e aprendizado não-linear com gating
  - Positional Encoding aprendível: mais eficiente para janelas fixas
  - Transformer Encoder: 2 camadas, 4 cabeças de atenção causal (Pre-LN)
  - MLP Head multi-task: direção (BUY/SELL) + duração dinâmica em paralelo

Referência: Lim et al. (2021) "Temporal Fusion Transformers for Interpretable
Multi-horizon Time Series Forecasting", IJF 37(4).

Interface pública:
    from transformer_model import TFTPredictor

    tft = TFTPredictor(n_features=16, seq_len=50)
    tft.fit(X_seq_train, y_dir_train, y_dur=y_dur_idx_train)
    proba = tft.predict_proba(X_seq_test)    # (N, 2) — col 0=DOWN, col 1=UP
    dur   = tft.predict_duration(X_seq_test) # (N,)   — índices de CANDIDATE_DURATIONS
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ─────────────────────────────────────────────────────────────────
#  Blocos arquiteturais
# ─────────────────────────────────────────────────────────────────

class GatedResidualNetwork(nn.Module):
    """
    GRN conforme Lim et al. (2021) — núcleo do TFT.

    Aplica transformação não-linear gated com skip connection:
        gate  = sigmoid(W_gate · x)
        h     = dropout(W2 · ELU(W1 · x))
        out   = LayerNorm(gate * h + x)
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.fc1  = nn.Linear(d_model, d_model)
        self.fc2  = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.elu(self.fc1(x))
        h = self.drop(self.fc2(h))
        g = torch.sigmoid(self.gate(x))
        return self.norm(g * h + x)


class VariableSelectionNetwork(nn.Module):
    """
    VSN: aprende um peso softmax por feature para cada passo temporal.

    Emite attention weights interpretáveis (shape batch, seq_len, n_features),
    permitindo análise de quais features o modelo prioriza em cada momento.
    """

    def __init__(self, n_features: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        # Projeta cada feature escalar para d_model independentemente
        self.feature_projs = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(n_features)
        ])
        # Selector: contexto flatten → peso softmax por feature
        self.selector = nn.Sequential(
            nn.Linear(n_features * d_model, n_features),
            nn.Softmax(dim=-1),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            selected: (batch, seq_len, d_model) — features ponderadas
            weights:  (batch, seq_len, n_features) — para interpretabilidade
        """
        B, T, F_ = x.shape

        # Projeta individualmente cada feature: lista de (B, T, d_model)
        projs  = [self.feature_projs[i](x[..., i : i + 1]) for i in range(F_)]
        proj   = torch.stack(projs, dim=2)                     # (B, T, F_, d_model)

        # Calcula pesos via contexto achatado
        flat    = proj.flatten(2)                               # (B, T, F_*d_model)
        weights = self.selector(flat)                           # (B, T, F_)

        # Combinação ponderada das features projetadas
        selected = (proj * weights.unsqueeze(-1)).sum(dim=2)   # (B, T, d_model)
        return self.drop(selected), weights


class TFTModel(nn.Module):
    """
    Temporal Fusion Transformer completo.

    Input:  (batch, seq_len, n_features)  — float32
    Output: (logits_dir, logits_dur)
              logits_dir: (batch, 2)         — DOWN / UP
              logits_dur: (batch, n_dur)     — duração candidata (multi-task)
    """

    def __init__(
        self,
        n_features:    int,
        n_dur_classes: int,
        seq_len:       int   = 50,
        d_model:       int   = 64,
        n_heads:       int   = 4,
        n_layers:      int   = 2,
        dropout:       float = 0.15,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        # 1. Variable Selection Network
        self.vsn = VariableSelectionNetwork(n_features, d_model, dropout)

        # 2. Positional encoding aprendível (eficiente para janela de tamanho fixo)
        self.pos_emb = nn.Embedding(seq_len, d_model)

        # 3. GRN pré-encoder (estabiliza representações antes da atenção)
        self.pre_grn = GatedResidualNetwork(d_model, dropout)

        # 4. Transformer Encoder com atenção causal e Pre-Layer Norm
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # Pre-LN: mais estável que Post-LN em séries ruidosas
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,   # necessário com norm_first=True (Pre-LN)
        )

        # 5. GRN pós-encoder (refinamento após captura de dependências temporais)
        self.post_grn = GatedResidualNetwork(d_model, dropout)

        # 6. Temporal attention pooling aprendível
        #    (em vez de média, aprende quais timesteps são mais informativos)
        self.temporal_attn = nn.Linear(d_model, 1)

        # 7. Heads MLP multi-task — compartilham representação, mas aprendem
        #    objetivos distintos: direção (classificação binária) e duração (multinomial)
        self.head_dir = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2),
        )
        self.head_dur = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_dur_classes),
        )

    def _make_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Máscara causal: posições futuras recebem -inf (ignoradas na atenção)."""
        return torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        device  = x.device

        # VSN: seleção ponderada de features → (B, T, d_model)
        h, _ = self.vsn(x)

        # Positional encoding (broadcast sobre batch)
        h = h + self.pos_emb(torch.arange(T, device=device))

        # GRN pré-encoder
        h = self.pre_grn(h)

        # Transformer Encoder com máscara causal
        causal_mask = self._make_causal_mask(T, device)
        h = self.encoder(h, mask=causal_mask)

        # GRN pós-encoder
        h = self.post_grn(h)

        # Temporal attention pooling: (B, T, 1) → weighted sum → (B, d_model)
        attn_w = F.softmax(self.temporal_attn(h), dim=1)   # (B, T, 1)
        ctx    = (h * attn_w).sum(dim=1)                    # (B, d_model)

        return self.head_dir(ctx), self.head_dur(ctx)


# ─────────────────────────────────────────────────────────────────
#  Wrapper sklearn-compatible
# ─────────────────────────────────────────────────────────────────

class TFTPredictor:
    """
    Wrapper treinável e serializável do TFTModel.

    Interface compatível com joblib (salva state_dict, não o módulo):
        fit(X_seq, y_dir, y_dur=None)  → self
        predict_proba(X_seq)           → np.ndarray (N, 2)  [P_DOWN, P_UP]
        predict_duration(X_seq)        → np.ndarray (N,)    índices → CANDIDATE_DURATIONS
    """

    def __init__(
        self,
        n_features:    int,
        n_dur_classes: int   = 4,
        seq_len:       int   = 50,
        d_model:       int   = 64,
        n_heads:       int   = 4,
        n_layers:      int   = 2,
        dropout:       float = 0.15,
        epochs:        int   = 80,
        batch_size:    int   = 128,
        lr:            float = 3e-4,
        patience:      int   = 10,
    ):
        self._arch = dict(
            n_features=n_features,
            n_dur_classes=n_dur_classes,
            seq_len=seq_len,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )
        self._train_cfg = dict(
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            patience=patience,
        )
        self._state_dict = None   # preenchido após fit; base da serialização
        self._model      = None   # instância em memória (lazy, não serializado)

    # ─────────────────────────────────
    #  Treinamento
    # ─────────────────────────────────

    def fit(
        self,
        X_seq:     np.ndarray,
        y_dir:     np.ndarray,
        y_dur:     Optional[np.ndarray] = None,
        val_split: float = 0.1,
    ) -> "TFTPredictor":
        """
        Treina o TFT com Early Stopping e Cosine LR com warmup.

        Args:
            X_seq:     (N, seq_len, n_features) — float32
            y_dir:     (N,)  — 0=cai / 1=sobe
            y_dur:     (N,)  — índices de duração 0..n_dur_classes-1 (opcional)
            val_split: fração do fim de X_seq usada para validação (temporal)
        """
        epochs     = self._train_cfg["epochs"]
        batch_size = self._train_cfg["batch_size"]
        lr         = self._train_cfg["lr"]
        patience   = self._train_cfg["patience"]
        device     = torch.device("cpu")

        model = TFTModel(**self._arch).to(device)
        self._model = model

        n       = len(X_seq)
        n_val   = max(int(n * val_split), 1)
        n_train = n - n_val

        Xt = torch.tensor(X_seq[:n_train], dtype=torch.float32)
        Xv = torch.tensor(X_seq[n_train:], dtype=torch.float32)
        yt = torch.tensor(y_dir[:n_train], dtype=torch.long)
        yv = torch.tensor(y_dir[n_train:], dtype=torch.long)

        has_dur = y_dur is not None
        if has_dur:
            ydt = torch.tensor(y_dur[:n_train], dtype=torch.long)
            ydv = torch.tensor(y_dur[n_train:], dtype=torch.long)
            ds_train = TensorDataset(Xt, yt, ydt)
            ds_val   = TensorDataset(Xv, yv, ydv)
        else:
            ds_train = TensorDataset(Xt, yt)
            ds_val   = TensorDataset(Xv, yv)

        dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,  drop_last=False)
        dl_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False, drop_last=False)

        # AdamW + Cosine LR com warmup linear de 5 épocas
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        warmup    = min(5, epochs)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=max(epochs - warmup, 1), eta_min=lr * 0.01
                ),
            ],
            milestones=[warmup],
        )

        # Label smoothing: reduz overconfidence em dados ruidosos de mercado
        ce_dir = nn.CrossEntropyLoss(label_smoothing=0.10)
        ce_dur = nn.CrossEntropyLoss(label_smoothing=0.05)

        best_val_loss = float("inf")
        best_state    = None
        no_improve    = 0

        for epoch in range(1, epochs + 1):
            # ── Treino ──────────────────────────────────────────
            model.train()
            for batch in dl_train:
                optimizer.zero_grad()
                if has_dur:
                    xb, yb_d, yb_dur = [t.to(device) for t in batch]
                else:
                    xb, yb_d = [t.to(device) for t in batch]
                lg_dir, lg_dur = model(xb)
                loss = ce_dir(lg_dir, yb_d)
                if has_dur:
                    loss = loss + 0.3 * ce_dur(lg_dur, yb_dur)   # peso menor p/ duração
                loss.backward()
                # Gradient clipping: evita explosão de gradientes em séries longas
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()

            # ── Validação ────────────────────────────────────────
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in dl_val:
                    if has_dur:
                        xb, yb_d, yb_dur = [t.to(device) for t in batch]
                    else:
                        xb, yb_d = [t.to(device) for t in batch]
                    lg_dir, lg_dur = model(xb)
                    v = ce_dir(lg_dir, yb_d)
                    if has_dur:
                        v = v + 0.3 * ce_dur(lg_dur, yb_dur)
                    val_loss += v.item()

            val_loss /= max(len(dl_val), 1)

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state    = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve    = 0
            else:
                no_improve += 1

            if epoch % 10 == 0 or epoch == 1:
                lr_now = optimizer.param_groups[0]["lr"]
                print(
                    f"    [TFT] época {epoch:3d}/{epochs}"
                    f" | val_loss={val_loss:.4f}"
                    f" | lr={lr_now:.2e}"
                    f" | patience={no_improve}/{patience}"
                )

            if no_improve >= patience:
                print(f"    [TFT] Early stopping na época {epoch}.")
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        # Serializa apenas o state_dict (leve, sem referências ao grafo de computação)
        self._state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        return self

    # ─────────────────────────────────
    #  Inferência
    # ─────────────────────────────────

    def _ensure_model(self) -> TFTModel:
        """Reconstrói o nn.Module a partir do state_dict se necessário (lazy)."""
        if self._model is None:
            m = TFTModel(**self._arch)
            if self._state_dict is not None:
                m.load_state_dict(self._state_dict)
            m.eval()
            self._model = m
        return self._model

    def predict_proba(self, X_seq: np.ndarray) -> np.ndarray:
        """
        Retorna probabilidades (N, 2).
        Coluna 0 = P(DOWN), Coluna 1 = P(UP).
        """
        model = self._ensure_model()
        model.eval()
        with torch.no_grad():
            X_t           = torch.tensor(X_seq, dtype=torch.float32)
            logits_dir, _ = model(X_t)
            proba         = F.softmax(logits_dir, dim=-1).numpy()
        return proba

    def predict_duration(self, X_seq: np.ndarray) -> np.ndarray:
        """
        Retorna índices de duração (N,) para indexar em CANDIDATE_DURATIONS.
        Ex: [0] → CANDIDATE_DURATIONS[0], [2] → CANDIDATE_DURATIONS[2].
        """
        model = self._ensure_model()
        model.eval()
        with torch.no_grad():
            X_t           = torch.tensor(X_seq, dtype=torch.float32)
            _, logits_dur = model(X_t)
            return logits_dur.argmax(dim=-1).numpy()

    # ─────────────────────────────────
    #  Serialização (joblib-safe)
    # ─────────────────────────────────

    def __getstate__(self):
        """Exclui _model (nn.Module) da serialização — usa apenas _state_dict."""
        state = self.__dict__.copy()
        state["_model"] = None
        return state

    def __setstate__(self, state):
        """Restaura todos os atributos; _model é recriado via _ensure_model() na próxima chamada."""
        self.__dict__.update(state)
        self._model = None
