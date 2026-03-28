"""LOSO training for ST-GCN + temporal Transformer (skeleton graph + Transformer)."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from baseline.registry import apply_method_defaults, get_spec
from baseline.impact_dataset import BadmintonDataset
from baseline.models.stgcn_transformer import STGCNTransformer
from baseline.training.dataloader_utils import (
    build_loader_kwargs,
    configure_cuda_training,
    default_num_workers,
    effective_multiprocessing_start_method,
)

log = logging.getLogger("baseline.training.loso_stgcn")


def contact_weighted_mse_stgcn(
    pred: torch.Tensor,
    target: torch.Tensor,
    ev_idx: torch.Tensor,
    alpha: float = 10.0,
    half_win: int = 25,
) -> torch.Tensor:
    T = pred.shape[1]
    weights = torch.ones_like(target)
    for b, ev in enumerate(ev_idx):
        lo = max(0, ev.item() - half_win)
        hi = min(T, ev.item() + half_win + 1)
        weights[b, lo:hi] = alpha
    return (weights * (pred - target) ** 2).mean()


def compute_metrics_stgcn(
    preds: Union[np.ndarray, List[np.ndarray]],
    targets: Union[np.ndarray, List[np.ndarray]],
    ev_idxs: np.ndarray,
) -> Dict[str, float]:
    if isinstance(preds, list):
        flat_pred = np.concatenate(preds) if preds else np.array([], dtype=np.float32)
        flat_target = np.concatenate(targets) if targets else np.array([], dtype=np.float32)
    else:
        flat_pred = preds.flatten()
        flat_target = targets.flatten()
    ss_res = ((flat_target - flat_pred) ** 2).sum()
    ss_tot = ((flat_target - flat_target.mean()) ** 2).sum()
    r2 = 1.0 - ss_res / (ss_tot + 1e-8)
    rmse = float(np.sqrt(((flat_pred - flat_target) ** 2).mean()))
    peak_errs, peak_timings = [], []
    for p, t, ev in zip(preds, targets, ev_idxs):
        win = slice(max(0, ev - 30), min(len(t), ev + 30))
        peak_true = t[win].max()
        idx_pred = p[win].argmax()
        idx_true = t[win].argmax()
        peak_errs.append(abs(p[win].max() - peak_true))
        peak_timings.append(abs(int(idx_pred) - int(idx_true)))
    out = {
        "r2": float(r2),
        "rmse": rmse,
        "peak_err": float(np.mean(peak_errs)),
        "peak_timing": float(np.mean(peak_timings)),
        "n_samples": float(len(preds)),
        # Aliases for unified tables
        "r2_fz": float(r2),
        "rmse_fz": rmse,
        "peak_err_bw": float(np.mean(peak_errs)),
        "peak_timing_fr": float(np.mean(peak_timings)),
    }
    return out


def pad_collate(batch):
    if isinstance(batch[0], dict):
        feats = [b["pose"].numpy() for b in batch]
        targets = [b["target"].numpy() for b in batch]
        ev_idxs = [int(b["ev_idx"]) for b in batch]
    else:
        feats, targets, ev_idxs = zip(*batch)
    max_t = max(f.shape[0] for f in feats)

    def pad(arr, val=0.0):
        pad_len = max_t - arr.shape[0]
        if pad_len == 0:
            return arr
        return np.concatenate([arr, np.full((pad_len, *arr.shape[1:]), val)], axis=0)

    feats_p = torch.tensor(np.stack([pad(f) for f in feats]), dtype=torch.float32)
    targets_p = torch.tensor(np.stack([pad(t) for t in targets]), dtype=torch.float32)
    ev_t = torch.tensor(np.array(ev_idxs), dtype=torch.long)
    if targets_p.ndim == 3 and targets_p.shape[-1] == 1:
        targets_p = targets_p.squeeze(-1)
    pad_mask = torch.zeros(len(feats), max_t, dtype=torch.bool)
    for i, f in enumerate(feats):
        if f.shape[0] < max_t:
            pad_mask[i, f.shape[0] :] = True
    return feats_p, targets_p, ev_t, pad_mask


def train_one_epoch(
    model, loader, optimiser, device, alpha=10.0, half_win=25, clip_norm=1.0
) -> float:
    model.train()
    total_loss, n = 0.0, 0
    for feats, targets, ev_idxs, pad_mask in loader:
        feats = feats.to(device)
        targets = targets.to(device)
        ev_idxs = ev_idxs.to(device)
        pad_mask = pad_mask.to(device)
        optimiser.zero_grad()
        preds = model(feats, src_key_padding_mask=pad_mask)
        loss = contact_weighted_mse_stgcn(preds, targets, ev_idxs, alpha, half_win)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimiser.step()
        total_loss += loss.item() * feats.size(0)
        n += feats.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate_stgcn(model, loader, device, alpha=10.0, half_win=25) -> Tuple[float, Dict]:
    model.eval()
    total_loss, n = 0.0, 0
    # Per-sample trimmed series — pad_collate pads each batch to its own max T, so we cannot
    # np.concatenate batches on the time axis; strip padding with pad_mask first.
    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    all_evs: List[int] = []
    for feats, targets, ev_idxs, pad_mask in loader:
        feats = feats.to(device)
        targets = targets.to(device)
        ev_idxs = ev_idxs.to(device)
        pad_mask = pad_mask.to(device)
        preds = model(feats, src_key_padding_mask=pad_mask)
        loss = contact_weighted_mse_stgcn(preds, targets, ev_idxs, alpha, half_win)
        total_loss += loss.item() * feats.size(0)
        n += feats.size(0)
        pm = pad_mask.cpu()
        pr = preds.cpu().numpy()
        tg = targets.cpu().numpy()
        evc = ev_idxs.cpu().numpy()
        for b in range(feats.size(0)):
            n_valid = int((~pm[b]).sum().item())
            pb = np.asarray(pr[b, :n_valid], dtype=np.float32).ravel()
            tb = np.asarray(tg[b, :n_valid], dtype=np.float32).ravel()
            all_preds.append(pb)
            all_targets.append(tb)
            all_evs.append(int(evc[b]))
    val_loss = total_loss / max(n, 1)
    evs_np = np.asarray(all_evs, dtype=np.int64)
    metrics = compute_metrics_stgcn(all_preds, all_targets, evs_np)
    return val_loss, metrics


def train_fold_stgcn(
    fold_name: str,
    train_paths: List[str],
    test_paths: List[str],
    args: Any,
    device: torch.device,
    out_dir: Path,
) -> Dict:
    log.info("Fold %s | train=%d test=%d", fold_name, len(train_paths), len(test_paths))
    train_ds = BadmintonDataset(train_paths, fz_only=args.fz_only)
    test_ds = BadmintonDataset(test_paths, fz_only=args.fz_only)
    configure_cuda_training(device)
    _nw = getattr(args, "num_workers", None)
    nw = int(default_num_workers() if _nw is None else _nw)
    pin_mem = (not getattr(args, "no_pin_memory", False)) and (device.type == "cuda")
    kw = build_loader_kwargs(
        collate_fn=pad_collate,
        num_workers=nw,
        pin_memory=pin_mem,
        prefetch_factor=getattr(args, "prefetch_factor", None) if nw > 0 else None,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        **kw,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        **kw,
    )
    fold_dir = out_dir / fold_name
    fold_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = fold_dir / "best_model.pth"
    fold_done = (fold_dir / "fold_complete.json").exists()

    if getattr(args, "resume_folds", False) and best_ckpt.exists() and fold_done:
        log.info(
            "  resume fold %s: skip train, load %s → eval on test loader",
            fold_name,
            best_ckpt,
        )
        model = STGCNTransformer(
            hidden_dim=args.hidden_dim,
            gcn_channels=(args.gcn_ch1, args.gcn_ch2),
            num_tf_layers=args.tf_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
        ).to(device)
        try:
            model.load_state_dict(
                torch.load(best_ckpt, map_location=device, weights_only=True)
            )
        except Exception as e:
            log.warning("  resume load failed (%s); training from scratch", e)
        else:
            _, metrics = evaluate_stgcn(
                model, test_loader, device, args.loss_alpha, args.loss_half_win
            )
            log.info(
                "  fold %s (resume): r2=%.3f rmse=%.4f peak_err=%.4f",
                fold_name,
                metrics.get("r2", float("nan")),
                metrics.get("rmse", float("nan")),
                metrics.get("peak_err", float("nan")),
            )
            return metrics

    log.info(
        "  DataLoader: num_workers=%d mp_start=%s prefetch=%s pin_memory=%s",
        nw,
        effective_multiprocessing_start_method(),
        kw.get("prefetch_factor", "-"),
        pin_mem,
    )
    model = STGCNTransformer(
        hidden_dim=args.hidden_dim,
        gcn_channels=(args.gcn_ch1, args.gcn_ch2),
        num_tf_layers=args.tf_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.epochs, eta_min=args.lr * 0.05
    )
    best_val_loss = float("inf")
    patience_cnt = 0
    best_metrics: Dict = {}
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimiser, device, args.loss_alpha, args.loss_half_win
        )
        val_loss, metrics = evaluate_stgcn(
            model, test_loader, device, args.loss_alpha, args.loss_half_win
        )
        scheduler.step()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = metrics
            patience_cnt = 0
            torch.save(model.state_dict(), fold_dir / "best_model.pth")
        else:
            patience_cnt += 1
        if epoch % 50 == 0 or epoch == 1:
            log.info(
                "  ep %d/%d train=%.4f val=%.4f r2=%.3f peak_err=%.3f lr=%.2e",
                epoch,
                args.epochs,
                train_loss,
                val_loss,
                metrics["r2"],
                metrics["peak_err"],
                scheduler.get_last_lr()[0],
            )
        if patience_cnt >= args.patience:
            log.info("  early stop epoch=%d", epoch)
            break
    (fold_dir / "fold_complete.json").write_text(
        json.dumps({"finished": True, "fold": fold_name}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return best_metrics


def run_stgcn_loso(args: Any) -> Dict[str, Any]:
    method_id = args.method
    get_spec(method_id)
    apply_method_defaults(method_id, args)
    if getattr(args, "device", None) in (None, ""):
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if not hasattr(args, "loss_alpha"):
        args.loss_alpha = 10.0
    if not hasattr(args, "loss_half_win"):
        args.loss_half_win = 25

    device = torch.device(args.device)
    with open(args.loso_splits, encoding="utf-8") as f:
        splits = json.load(f)
    if getattr(args, "folds", None):
        splits = {k: v for k, v in splits.items() if k in args.folds}

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.run_dir) if args.run_dir else Path("runs") / f"{ts}_{method_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output: %s", out_dir)

    cfg = json.loads(json.dumps(vars(args), default=str))
    cfg["method_id"] = method_id
    cfg["family"] = "stgcn"
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    t0 = time.time()
    per_fold: Dict[str, Dict] = {}
    for fold_name, fold_data in sorted(splits.items()):
        per_fold[fold_name] = train_fold_stgcn(
            fold_name,
            fold_data["train"],
            fold_data["test"],
            args,
            device,
            out_dir,
        )

    elapsed = time.time() - t0
    r2s = [v["r2"] for v in per_fold.values()]
    rmses = [v["rmse"] for v in per_fold.values()]
    peak_errs = [v["peak_err"] for v in per_fold.values()]
    peak_times = [v["peak_timing"] for v in per_fold.values()]

    summary = {
        "experiment": method_id,
        "method": method_id,
        "family": "stgcn",
        "timestamp": ts,
        "elapsed_sec": round(elapsed, 1),
        "mean": {
            "r2": round(float(np.mean(r2s)), 4),
            "r2_std": round(float(np.std(r2s)), 4),
            "r2_fz": round(float(np.mean(r2s)), 4),
            "rmse": round(float(np.mean(rmses)), 4),
            "rmse_fz": round(float(np.mean(rmses)), 4),
            "peak_err": round(float(np.mean(peak_errs)), 4),
            "peak_err_bw": round(float(np.mean(peak_errs)), 4),
            "peak_timing": round(float(np.mean(peak_times)), 2),
            "peak_timing_fr": round(float(np.mean(peak_times)), 2),
        },
        "per_fold": {k: {m: round(float(v), 4) for m, v in mv.items()} for k, mv in per_fold.items()},
        "args": vars(args),
    }

    log.info(
        "LOSO mean: r2=%.3f±%.3f rmse=%.4f peak_err=%.4f peak_timing=%.1f (%.1f min)",
        summary["mean"]["r2"],
        summary["mean"]["r2_std"],
        summary["mean"]["rmse"],
        summary["mean"]["peak_err"],
        summary["mean"]["peak_timing"],
        elapsed / 60,
    )

    if args.save_report:
        summ_path = out_dir / "summary.json"
        summ_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        log.info("Wrote %s", summ_path)
        try:
            from baseline.tasks.canonical import write_summary_canonical

            cp = write_summary_canonical(summary_path=summ_path, experiment=method_id)
            log.info("Wrote %s", cp)
        except Exception as e:
            log.warning("Could not write summary_canonical.json: %s", e)
    return summary
