"""
Multi-view late fusion (confidence-weighted averaging) for *any* single-view run.

Reads ``config.json`` inside ``--base_run_dir`` to recover ``family`` + ``method_id``,
then loads per-fold checkpoints and fuses predictions across cameras for the same impact.

Usage::

    python -m baseline fuse \\
        --loso_splits data/reports/loso_splits_10p.json \\
        --base_run_dir runs/my_tcn_bilstm_run \\
        --save_report

For ST-GCN runs, uses impact-level alignment (same as the legacy E5 script).
For flat runs (TCN+LSTM/GRU/Transformer/TCN-MLP), uses per-trial grouping (legacy E2).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from baseline.registry import build_flat_model, get_spec
from baseline.impact_dataset import INPUT_DIM, build_features
from baseline.models.stgcn_transformer import STGCNTransformer

log = logging.getLogger("late_fusion")


# ---------------------------------------------------------------------------
# Flat models: group by trial (same convention as legacy e2_late_fusion)
# ---------------------------------------------------------------------------

def group_by_trial(paths: List[str]) -> Dict[str, Dict[int, str]]:
    groups: Dict[str, Dict[int, str]] = defaultdict(dict)
    for p in paths:
        name = Path(p).stem
        m = re.match(r"(.+)_cam(\d+)_(.+)", name)
        if m:
            prefix, cam, suffix = m.group(1), int(m.group(2)), m.group(3)
            trial_key = f"{prefix}_{suffix}"
            groups[trial_key][cam] = p
    return dict(groups)


@torch.no_grad()
def late_fusion_flat(
    model: torch.nn.Module,
    trial_groups: Dict[str, Dict[int, str]],
    cameras: List[int],
    fz_only: bool,
    device: torch.device,
) -> Dict[str, float]:
    from baseline.training.metrics import compute_metrics, compute_peak_metrics

    all_preds, all_targets, all_ev = [], [], []
    n_fused, n_single = 0, 0
    for _trial_key, cam_paths in trial_groups.items():
        available = [c for c in cameras if c in cam_paths]
        if not available:
            continue
        cam_preds, cam_weights = [], []
        target_ref, ev_ref = None, None
        for cam in available:
            try:
                d = np.load(cam_paths[cam], allow_pickle=True)
            except Exception:
                continue
            kps = d["keypoints_norm"].astype(np.float32)
            sc = d["scores"].astype(np.float32)
            grf = d["grf_normalized"].astype(np.float32)
            ev = int(d["ev_idx"])
            pose = torch.from_numpy(build_features(kps, sc)).unsqueeze(0).to(device)
            target = grf[:, 2:3] if fz_only else grf
            with torch.no_grad():
                pred = model(pose).squeeze(0).detach().cpu().numpy()
            weight = float(sc.mean())
            cam_preds.append(pred)
            cam_weights.append(weight)
            if target_ref is None:
                target_ref = target
                ev_ref = ev
        if not cam_preds or target_ref is None:
            continue
        w = np.array(cam_weights, dtype=np.float64)
        w = w / (w.sum() + 1e-8)
        fused = sum(wi * pi for wi, pi in zip(w, cam_preds))
        all_preds.append(fused)
        all_targets.append(target_ref)
        all_ev.append(ev_ref)
        n_fused += 1
        n_single += len(cam_preds)
    if not all_preds:
        return {}
    metrics = compute_metrics(all_preds, all_targets)
    metrics.update(compute_peak_metrics(all_preds, all_targets, all_ev))
    metrics["n_test"] = float(len(all_preds))
    metrics["avg_cams"] = n_single / max(n_fused, 1)
    metrics["r2"] = metrics.get("r2_fz", float("nan"))
    metrics["rmse"] = metrics.get("rmse_fz", float("nan"))
    metrics["peak_err"] = metrics.get("peak_err_bw", float("nan"))
    metrics["peak_timing"] = metrics.get("peak_timing_fr", float("nan"))
    return metrics


def _cfg_to_args(cfg: Dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(**{k: v for k, v in cfg.items() if not k.startswith("_")})


def load_flat_model_for_fold(
    base_run: Path,
    fold: str,
    cfg: Dict[str, Any],
    device: torch.device,
) -> Optional[torch.nn.Module]:
    ckpt = base_run / f"fold_{fold}" / "best_model.pth"
    if not ckpt.exists():
        return None
    method_id = cfg.get("method_id") or cfg.get("method")
    if not method_id:
        log.error("config.json missing method_id")
        return None
    args = _cfg_to_args(cfg)
    out_dim = 1 if cfg.get("fz_only", True) else 3
    model = build_flat_model(method_id, args, INPUT_DIM, out_dim).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    return model


def run_late_fusion_flat(args: argparse.Namespace, cfg: Dict[str, Any]) -> Dict[str, Any]:
    base = Path(args.base_run_dir)
    device = torch.device(args.device)
    splits = json.loads(Path(args.loso_splits).read_text(encoding="utf-8"))
    cameras = args.cameras or [1, 2, 3, 4, 5, 6, 7, 8]
    per_fold: Dict[str, Dict] = {}
    for test_sub, fold in splits.items():
        log.info("Fusion fold %s", test_sub)
        model = load_flat_model_for_fold(base, test_sub, cfg, device)
        if model is None:
            log.warning("Missing checkpoint for %s", test_sub)
            continue
        trial_groups = group_by_trial(fold["test"])
        m = late_fusion_flat(model, trial_groups, cameras, args.fz_only, device)
        if m:
            per_fold[test_sub] = m
    return _summarize("late_fusion_flat", per_fold, base, args)


# ---------------------------------------------------------------------------
# ST-GCN: impact-level fusion (legacy e5)
# ---------------------------------------------------------------------------

def load_stgcn_for_fold(
    base_run: Path,
    fold: str,
    cfg: Dict[str, Any],
    device: torch.device,
) -> Optional[STGCNTransformer]:
    ckpt = base_run / fold / "best_model.pth"
    if not ckpt.exists():
        return None
    m = STGCNTransformer(
        hidden_dim=cfg.get("hidden_dim", 128),
        gcn_channels=(cfg.get("gcn_ch1", 32), cfg.get("gcn_ch2", 64)),
        num_tf_layers=cfg.get("tf_layers", 2),
        num_heads=cfg.get("num_heads", 4),
        dropout=0.0,
    ).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    m.eval()
    return m


def group_by_impact(paths: List[str]) -> Dict[Tuple[str, str], Dict[int, str]]:
    groups: Dict[Tuple[str, str], Dict[int, str]] = defaultdict(dict)
    pat = re.compile(r"^(.+)_cam(\d+)_impact_(\d+)\.npz$")
    for p in paths:
        m = pat.match(Path(p).name)
        if m:
            groups[(m.group(1), m.group(3))][int(m.group(2))] = p
    return dict(groups)


def infer_one_stgcn(
    path: str,
    model: STGCNTransformer,
    device: torch.device,
    fz_only: bool,
) -> Optional[Dict]:
    from baseline.impact_dataset import BadmintonDataset

    ds = BadmintonDataset([path], fz_only=fz_only)
    if len(ds) == 0:
        return None
    item = ds[0]
    pose = item["pose"].unsqueeze(0).to(device)
    target = item["target"].squeeze(-1).numpy()
    ev_idx = int(item["ev_idx"])
    with torch.no_grad():
        pred = model(pose).squeeze(0).detach().cpu().numpy()
    path0 = ds.paths[0]
    try:
        with np.load(path0, mmap_mode="r", allow_pickle=False) as d:
            scores = np.asarray(d["scores"], dtype=np.float32)
    except Exception:
        with np.load(path0, allow_pickle=True) as d:
            scores = np.asarray(d["scores"], dtype=np.float32)
    conf = float(scores[:, 11:].mean())
    return {"pred": pred, "target": target, "ev_idx": ev_idx, "conf": conf}


def fuse_event(cam_results: List[Dict]) -> Optional[Dict]:
    if not cam_results:
        return None
    ev_idxs = np.array([r["ev_idx"] for r in cam_results])
    ref_ev = int(np.median(ev_idxs))
    ref_len = int(np.median([len(r["pred"]) for r in cam_results]))
    aligned, confs = [], []
    for r in cam_results:
        shift = r["ev_idx"] - ref_ev
        pred = r["pred"].copy()
        if shift > 0:
            pred = np.concatenate([np.zeros(shift, dtype=np.float32), pred])
        elif shift < 0:
            pred = pred[-shift:]
        if len(pred) > ref_len:
            pred = pred[:ref_len]
        elif len(pred) < ref_len:
            pred = np.concatenate([pred, np.zeros(ref_len - len(pred), dtype=np.float32)])
        aligned.append(pred)
        confs.append(r["conf"])
    confs = np.array(confs, dtype=np.float32)
    weights = confs / (confs.sum() + 1e-8)
    fused = (np.stack(aligned, axis=0) * weights[:, None]).sum(axis=0)
    best = int(np.argmin(np.abs(ev_idxs - ref_ev)))
    tgt = cam_results[best]["target"]
    if len(tgt) > ref_len:
        tgt = tgt[:ref_len]
    elif len(tgt) < ref_len:
        tgt = np.concatenate([tgt, np.zeros(ref_len - len(tgt), dtype=np.float32)])
    return {"pred": fused, "target": tgt, "ev_idx": ref_ev, "n_cams": len(cam_results)}


def metrics_from_events(events: List[Dict]) -> Dict[str, float]:
    all_pred = np.concatenate([e["pred"] for e in events])
    all_target = np.concatenate([e["target"] for e in events])
    r2 = float(
        1.0
        - ((all_target - all_pred) ** 2).sum()
        / (((all_target - all_target.mean()) ** 2).sum() + 1e-8)
    )
    rmse = float(np.sqrt(((all_pred - all_target) ** 2).mean()))
    peak_errs, peak_timings = [], []
    for e in events:
        p, t, ev = e["pred"], e["target"], e["ev_idx"]
        lo, hi = max(0, ev - 30), min(len(t), ev + 30)
        peak_errs.append(float(abs(p[lo:hi].max() - t[lo:hi].max())))
        peak_timings.append(float(abs(int(p[lo:hi].argmax()) - int(t[lo:hi].argmax()))))
    return {
        "r2": r2,
        "rmse": rmse,
        "peak_err": float(np.mean(peak_errs)),
        "peak_timing": float(np.mean(peak_timings)),
        "r2_fz": r2,
        "rmse_fz": rmse,
        "peak_err_bw": float(np.mean(peak_errs)),
        "peak_timing_fr": float(np.mean(peak_timings)),
        "avg_cams": float(np.mean([e["n_cams"] for e in events])),
    }


def run_fold_stgcn_fusion(
    test_paths: List[str],
    model: STGCNTransformer,
    fz_only: bool,
    device: torch.device,
) -> Dict[str, float]:
    groups = group_by_impact(test_paths)
    events = []
    for (_trial, _idx), cam_paths in sorted(groups.items()):
        cam_results = []
        for _cam_id, path in sorted(cam_paths.items()):
            res = infer_one_stgcn(path, model, device, fz_only)
            if res is not None:
                cam_results.append(res)
        ev = fuse_event(cam_results)
        if ev is not None:
            events.append(ev)
    return metrics_from_events(events) if events else {}


def run_late_fusion_stgcn(args: argparse.Namespace, cfg: Dict[str, Any]) -> Dict[str, Any]:
    base = Path(args.base_run_dir)
    device = torch.device(args.device)
    splits = json.loads(Path(args.loso_splits).read_text(encoding="utf-8"))
    per_fold: Dict[str, Dict] = {}
    for fold_name, fold_data in sorted(splits.items()):
        model = load_stgcn_for_fold(base, fold_name, cfg, device)
        if model is None:
            log.warning("Missing ST-GCN checkpoint: %s", base / fold_name)
            continue
        per_fold[fold_name] = run_fold_stgcn_fusion(
            fold_data["test"], model, args.fz_only, device
        )
    return _summarize("late_fusion_stgcn", per_fold, base, args)


def _summarize(
    experiment: str,
    per_fold: Dict[str, Dict],
    base: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    if not per_fold:
        log.error("No folds completed")
        return {}
    keys = [k for k in next(iter(per_fold.values())).keys() if isinstance(next(iter(per_fold.values()))[k], (float, int))]
    mean = {k: float(np.nanmean([r[k] for r in per_fold.values() if k in r])) for k in keys}
    std = {k: float(np.nanstd([r[k] for r in per_fold.values() if k in r])) for k in keys}
    out = {
        "experiment": experiment,
        "base_run_dir": str(base),
        "mean": {k: round(v, 4) for k, v in mean.items()},
        "std": {k: round(v, 4) for k, v in std.items()},
        "per_fold": per_fold,
    }
    log.info("Late fusion mean r2=%.4f rmse=%.4f", mean.get("r2_fz", mean.get("r2", float("nan"))), mean.get("rmse_fz", mean.get("rmse", float("nan"))))
    if args.save_report:
        out_dir = base / "late_fusion"
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / "summary.json"
        p.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        log.info("Wrote %s", p)
        try:
            from baseline.tasks.canonical import write_summary_canonical

            cfg_path = base / "config.json"
            cfg_run = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
            mid = cfg_run.get("method_id") or cfg_run.get("method") or "unknown"
            cp = write_summary_canonical(summary_path=p, experiment=f"{mid}_late_fusion")
            log.info("Wrote %s", cp)
        except Exception as e:
            log.warning("Could not write late_fusion summary_canonical.json: %s", e)
    return out


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BadmintonGRF late fusion")
    p.add_argument("--loso_splits", required=True)
    p.add_argument("--base_run_dir", required=True, help="Directory that contains config.json + fold_* checkpoints.")
    p.add_argument("--cameras", nargs="+", type=int, default=None)
    p.add_argument("--fz_only", action="store_true", default=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_report", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args(argv)
    base = Path(args.base_run_dir)
    cfg_path = base / "config.json"
    if not cfg_path.exists():
        raise SystemExit(f"Missing {cfg_path}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    family = cfg.get("family")
    if not family:
        # Infer from method
        mid = cfg.get("method_id") or cfg.get("method", "")
        try:
            family = get_spec(mid).family
        except Exception:
            family = "flat"
    if family == "stgcn":
        run_late_fusion_stgcn(args, cfg)
    else:
        run_late_fusion_flat(args, cfg)


if __name__ == "__main__":
    main()
