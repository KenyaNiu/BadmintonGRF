"""
Registered baseline methods — ids, families, default hyperparameters.

Training entry: ``python -m baseline train``. Imports: ``from baseline.registry import METHODS, get_spec, …``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

Family = Literal["flat", "stgcn"]


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    short_name: str
    family: Family
    description: str
    defaults: Dict[str, Any]
    paper_tag: Optional[str] = None


METHODS: Dict[str, MethodSpec] = {
    "tcn_bilstm": MethodSpec(
        method_id="tcn_bilstm",
        short_name="TCN+BiLSTM",
        family="flat",
        description="Temporal conv + bidirectional LSTM head.",
        defaults={
            "lr": 1e-3,
            "batch_size": 512,
            "hidden_dim": 128,
            "tcn_channels": 128,
            "num_layers": 2,
            "dropout": 0.3,
        },
        paper_tag="E1",
    ),
    "gru_bigru": MethodSpec(
        method_id="gru_bigru",
        short_name="TCN+BiGRU",
        family="flat",
        description="Temporal conv + bidirectional GRU.",
        defaults={
            "lr": 1e-3,
            "batch_size": 512,
            "hidden_dim": 128,
            "tcn_channels": 128,
            "num_layers": 2,
            "dropout": 0.3,
        },
    ),
    "tcn_mlp": MethodSpec(
        method_id="tcn_mlp",
        short_name="TCN+MLP",
        family="flat",
        description="Stacked temporal convolutions + frame-wise MLP (no RNN).",
        defaults={
            "lr": 1e-3,
            "batch_size": 512,
            "tcn_channels": 128,
            "tcn_blocks": 4,
            "dropout": 0.25,
        },
    ),
    "seq_transformer": MethodSpec(
        method_id="seq_transformer",
        short_name="Seq-Transformer",
        family="flat",
        description="Linear projection + sinusoidal PE + Transformer encoder (no skeleton graph).",
        defaults={
            "lr": 5e-4,
            "batch_size": 256,
            "hidden_dim": 128,
            "tf_layers": 3,
            "num_heads": 4,
            "dropout": 0.2,
        },
    ),
    "dlinear": MethodSpec(
        method_id="dlinear",
        short_name="DLinear",
        family="flat",
        description="Moving-average trend vs seasonal + dual linear heads (AAAI 2023 DLinear family).",
        defaults={
            "lr": 1e-3,
            "batch_size": 512,
            "hidden_dim": 128,
            "dropout": 0.1,
            "kernel_size": 25,
        },
    ),
    "patch_tst": MethodSpec(
        method_id="patch_tst",
        short_name="PatchTST",
        family="flat",
        description="Time patching + Transformer on patch tokens (ICLR 2023 PatchTST family).",
        defaults={
            "lr": 5e-4,
            "batch_size": 256,
            "hidden_dim": 128,
            "patch_len": 16,
            "tf_layers": 2,
            "num_heads": 4,
            "dropout": 0.2,
            "max_patch_positions": 2048,
        },
    ),
    "ms_tcn": MethodSpec(
        method_id="ms_tcn",
        short_name="MS-TCN",
        family="flat",
        description="Parallel dilated temporal convolutions + fusion (multi-scale TCN).",
        defaults={
            "lr": 1e-3,
            "batch_size": 512,
            "hidden_dim": 128,
            "dropout": 0.25,
            "dilations": [1, 2, 4, 8],
        },
    ),
    "tsmixer_grf": MethodSpec(
        method_id="tsmixer_grf",
        short_name="TSMixer",
        family="flat",
        description="MLP-Mixer style token-mixing + channel-mixing baseline (fast and strong for multivariate time series).",
        defaults={
            "lr": 8e-4,
            "batch_size": 512,
            "hidden_dim": 256,
            "num_layers": 6,
            "dropout": 0.15,
            "max_len": 512,
            "weight_decay": 1e-4,
        },
    ),
    "patch_tst_xl": MethodSpec(
        method_id="patch_tst_xl",
        short_name="PatchTST-XL",
        family="flat",
        description="Large PatchTST: d_model=256, 4 layers, patch_len=8, wide FFN (strong time-series baseline).",
        defaults={
            "lr": 3e-4,
            "batch_size": 128,
            "hidden_dim": 256,
            "patch_len": 8,
            "tf_layers": 4,
            "num_heads": 8,
            "dropout": 0.12,
            "dim_ff": 1024,
            "max_patch_positions": 4096,
            "weight_decay": 1e-4,
        },
    ),
    "stgcn_transformer": MethodSpec(
        method_id="stgcn_transformer",
        short_name="ST-GCN+Transformer",
        family="stgcn",
        description="Spatial GCN on COCO-17 + temporal Transformer.",
        defaults={
            "lr": 5e-4,
            "batch_size": 256,
            "hidden_dim": 128,
            "gcn_ch1": 32,
            "gcn_ch2": 64,
            "tf_layers": 2,
            "num_heads": 4,
            "dropout": 0.2,
            "weight_decay": 1e-4,
        },
        paper_tag="E4",
    ),
}


def list_method_ids() -> List[str]:
    return sorted(METHODS.keys())


def get_spec(method_id: str) -> MethodSpec:
    if method_id not in METHODS:
        raise KeyError(
            f"Unknown method {method_id!r}. Choose one of: {', '.join(list_method_ids())}"
        )
    return METHODS[method_id]


def build_flat_model(method_id: str, args: Any, input_dim: int, output_dim: int) -> Any:
    from baseline.models.dlinear import DLinearGRF
    from baseline.models.gru import GRUBaseline
    from baseline.models.ms_tcn import MultiScaleTCNGRF
    from baseline.models.patch_tst import PatchTSTGRF
    from baseline.models.tcn_lstm import LSTMBaseline
    from baseline.models.tcn_mlp import TCNMLPBaseline
    from baseline.models.tsmixer_grf import TSMixerGRF
    from baseline.models.transformer_seq import SeqTransformer

    if method_id == "tcn_bilstm":
        return LSTMBaseline(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=output_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            bidirectional=not getattr(args, "unidirectional", False),
            tcn_channels=args.tcn_channels,
        )
    if method_id == "gru_bigru":
        return GRUBaseline(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=output_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            bidirectional=not getattr(args, "unidirectional", False),
            tcn_channels=args.tcn_channels,
        )
    if method_id == "tcn_mlp":
        return TCNMLPBaseline(
            input_dim=input_dim,
            output_dim=output_dim,
            tcn_channels=args.tcn_channels,
            num_blocks=getattr(args, "tcn_blocks", 4),
            dropout=args.dropout,
        )
    if method_id == "seq_transformer":
        return SeqTransformer(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.tf_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
        )
    if method_id == "dlinear":
        return DLinearGRF(
            input_dim=input_dim,
            output_dim=output_dim,
            kernel_size=int(getattr(args, "kernel_size", 25)),
            dropout=float(args.dropout),
        )
    if method_id == "patch_tst":
        _pdim_ff = getattr(args, "dim_ff", None)
        if _pdim_ff is None:
            _pdim_ff = int(args.hidden_dim) * 2
        return PatchTSTGRF(
            input_dim=input_dim,
            output_dim=output_dim,
            patch_len=int(getattr(args, "patch_len", 16)),
            d_model=int(args.hidden_dim),
            num_layers=int(args.tf_layers),
            num_heads=int(args.num_heads),
            dropout=float(args.dropout),
            max_patch_positions=int(getattr(args, "max_patch_positions", 2048)),
            dim_ff=int(_pdim_ff),
        )
    if method_id == "patch_tst_xl":
        dff = getattr(args, "dim_ff", None)
        if dff is None:
            dff = int(args.hidden_dim) * 4
        return PatchTSTGRF(
            input_dim=input_dim,
            output_dim=output_dim,
            patch_len=int(getattr(args, "patch_len", 8)),
            d_model=int(args.hidden_dim),
            num_layers=int(args.tf_layers),
            num_heads=int(args.num_heads),
            dropout=float(args.dropout),
            max_patch_positions=int(getattr(args, "max_patch_positions", 4096)),
            dim_ff=int(dff),
        )
    if method_id == "tsmixer_grf":
        return TSMixerGRF(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=int(getattr(args, "hidden_dim", 256)),
            num_layers=int(getattr(args, "num_layers", 6)),
            dropout=float(getattr(args, "dropout", 0.15)),
            max_len=int(getattr(args, "max_len", 512)),
        )
    if method_id == "ms_tcn":
        dil = getattr(args, "dilations", (1, 2, 4, 8))
        if isinstance(dil, list):
            dil = tuple(int(x) for x in dil)
        return MultiScaleTCNGRF(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=int(args.hidden_dim),
            dilations=dil,
            dropout=float(args.dropout),
        )
    raise ValueError(f"Not a flat method: {method_id}")


def apply_method_defaults(method_id: str, args: Any) -> None:
    spec = get_spec(method_id)
    for k, v in spec.defaults.items():
        if not hasattr(args, k) or getattr(args, k) is None:
            setattr(args, k, v)
