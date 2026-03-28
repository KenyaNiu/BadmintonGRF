"""
ST-GCN + Temporal Transformer baseline for GRF estimation.

Architecture:
  Input  (B, T, 119)  flat features
  Reshape → (B, T, 17, 7)  per-joint: pos_xy + vel_xy + acc_xy + score
  Spatial GCN ×2  7→32→64  (COCO-17 skeleton graph)
  Mean pool over joints → (B, T, 64)
  Linear projection → (B, T, hidden_dim)
  Temporal Transformer ×num_tf_layers  (pre-norm, batch_first)
  MLP Head → (B, T, 1)  Fz prediction

Parameter count: ~280K (hidden_dim=128, num_tf_layers=2)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# COCO-17 skeleton edges (0-indexed)
# ---------------------------------------------------------------------------
COCO_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # head
    (5, 7), (7, 9), (6, 8), (8, 10),          # arms
    (5, 6), (5, 11), (6, 12), (11, 12),       # torso
    (11, 13), (13, 15), (12, 14), (14, 16),   # legs
]
NUM_JOINTS = 17

# Input feature layout (must match build_features in impact_dataset.py)
# pos   (T, 34) = joints 0..16, xy coords, indices  0..33
# vel   (T, 34)                                      34..67
# acc   (T, 34)                                      68..101
# score (T, 17)                                      102..118
FEAT_PER_JOINT = 7   # 2+2+2+1


def build_adjacency(edges: list, num_nodes: int = NUM_JOINTS) -> np.ndarray:
    """Symmetrically-normalised adjacency with self-loops: D^{-1/2} A D^{-1/2}."""
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    A += np.eye(num_nodes, dtype=np.float32)           # self-loop
    deg = A.sum(axis=1)
    D_inv_sqrt = np.diag(np.where(deg > 0, deg ** -0.5, 0.0))
    return (D_inv_sqrt @ A @ D_inv_sqrt).astype(np.float32)


# ---------------------------------------------------------------------------
# Spatial graph convolution block
# ---------------------------------------------------------------------------

class GraphConv(nn.Module):
    """Single graph convolution: A-weighted neighbourhood aggregation + linear."""

    def __init__(self, in_channels: int, out_channels: int, A: np.ndarray):
        super().__init__()
        self.register_buffer("A", torch.from_numpy(A))   # (N, N)
        self.fc = nn.Linear(in_channels, out_channels, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, N, C_in)
        # aggregate neighbours: (B, T, N, C_in)
        x_agg = torch.einsum("nm,btnc->btnc", self.A, x)
        return F.gelu(self.fc(x_agg))                    # (B, T, N, C_out)


class SpatialGCNBlock(nn.Module):
    """Two-layer GCN block with residual connection + LayerNorm."""

    def __init__(self, in_c: int, out_c: int, A: np.ndarray, dropout: float = 0.1):
        super().__init__()
        self.gcn1 = GraphConv(in_c, out_c, A)
        self.gcn2 = GraphConv(out_c, out_c, A)
        self.norm = nn.LayerNorm(out_c)
        self.drop = nn.Dropout(dropout)
        self.shortcut = nn.Linear(in_c, out_c, bias=False) if in_c != out_c else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.gcn1(x)
        out = self.drop(out)
        out = self.gcn2(out)
        return self.norm(out + residual)


# ---------------------------------------------------------------------------
# Positional encoding  (sinusoidal, fixed — no parameters to overfit)
# ---------------------------------------------------------------------------

class SinusoidalPE(nn.Module):
    """Standard sinusoidal positional encoding, added to (B, T, d_model)."""

    def __init__(self, d_model: int, max_len: int = 256, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        return self.drop(x + self.pe[:, :x.size(1)])


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class STGCNTransformer(nn.Module):
    """
    Spatial-Temporal GCN + Transformer for single-camera GRF (Fz) estimation.

    Args:
        hidden_dim:     Transformer d_model (and projection dim after GCN pool).
        gcn_channels:   Intermediate GCN channel width [32, 64 by default].
        num_tf_layers:  Number of Transformer encoder layers.
        num_heads:      Attention heads (hidden_dim must be divisible by num_heads).
        dropout:        Dropout rate applied in GCN, Transformer, and head.
    """

    def __init__(
        self,
        hidden_dim:    int = 128,
        gcn_channels:  tuple = (32, 64),
        num_tf_layers: int = 2,
        num_heads:     int = 4,
        dropout:       float = 0.2,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        A = build_adjacency(COCO_EDGES)
        c1, c2 = gcn_channels

        # Spatial GCN: 7 → c1 → c2
        self.spatial = nn.Sequential(
            SpatialGCNBlock(FEAT_PER_JOINT, c1, A, dropout=dropout * 0.5),
            SpatialGCNBlock(c1, c2, A, dropout=dropout * 0.5),
        )

        # Project mean-pooled spatial features to Transformer d_model
        self.proj = nn.Sequential(
            nn.Linear(c2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # Positional encoding — CRITICAL: Transformer has no implicit ordering
        self.pos_enc = SinusoidalPE(hidden_dim, max_len=256, dropout=dropout * 0.5)

        # Temporal Transformer (pre-norm for stability on small datasets)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # pre-LN: more stable, better for small data
        )
        self.temporal_tf = nn.TransformerEncoder(
            enc_layer,
            num_layers=num_tf_layers,
            enable_nested_tensor=False,
        )

        # Output head
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

    # ------------------------------------------------------------------

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _reshape_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape flat (B, T, 119) → per-joint (B, T, 17, 7).

        Feature layout:
          pos   [  0: 34] → (17, 2)
          vel   [ 34: 68] → (17, 2)
          acc   [ 68:102] → (17, 2)
          score [102:119] → (17, 1)
        """
        B, T, _ = x.shape
        pos   = x[:, :, :34].reshape(B, T, NUM_JOINTS, 2)
        vel   = x[:, :, 34:68].reshape(B, T, NUM_JOINTS, 2)
        acc   = x[:, :, 68:102].reshape(B, T, NUM_JOINTS, 2)
        score = x[:, :, 102:].unsqueeze(-1)                  # (B, T, 17, 1)
        return torch.cat([pos, vel, acc, score], dim=-1)     # (B, T, 17, 7)

    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, 119) normalised features from BadmintonDataset.
            src_key_padding_mask: (B, T) bool tensor, True = padded position to ignore.
        Returns:
            out: (B, T)  Fz predictions.
        """
        # 1. Per-joint feature tensor
        x_joint = self._reshape_input(x)          # (B, T, 17, 7)

        # 2. Spatial GCN: model skeleton topology
        x_spatial = self.spatial(x_joint)         # (B, T, 17, 64)

        # 3. Aggregate joints → frame-level token
        x_pooled = x_spatial.mean(dim=2)          # (B, T, 64)
        x_proj   = self.proj(x_pooled)            # (B, T, hidden)

        # 4. Positional encoding — gives Transformer temporal order awareness
        x_pe = self.pos_enc(x_proj)               # (B, T, hidden)

        # 5. Temporal Transformer with padding mask
        x_tf = self.temporal_tf(
            x_pe,
            src_key_padding_mask=src_key_padding_mask,
        )                                          # (B, T, hidden)

        # 6. Per-frame Fz regression
        out = self.head(x_tf).squeeze(-1)         # (B, T)
        return out

    # ------------------------------------------------------------------

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = STGCNTransformer(hidden_dim=128, num_tf_layers=2, num_heads=4, dropout=0.2)
    print(f"Parameters: {model.count_parameters():,}")

    B, T = 4, 120
    x = torch.randn(B, T, 119)
    out = model(x)
    print(f"Input:  {x.shape}  →  Output: {out.shape}")   # expect (4, 120)
    assert out.shape == (B, T), f"Unexpected output shape: {out.shape}"
    print("Sanity check passed.")