"""Evaluate a LatCH checkpoint on the shared 5 % holdout — report val_point_mae.

Lets us compare souped / averaged heads against their ingredients without re-training.

Usage:
  sat-venv/bin/python scripts/eval_latch.py \\
      --ckpt latch_weights/latch_rms_energy_bass_soup_v1_best.pt \\
      [--feature rms_energy_bass] \\
      [--n-val-batches 50]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset, DataLoader


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--feature", default=None,
                   help="Override feature_name (defaults to checkpoint metadata).")
    p.add_argument("--holdout-frac", type=float, default=0.05)
    p.add_argument("--holdout-seed", type=int, default=12345)
    p.add_argument("--subset-frac", type=float, default=1.0)
    p.add_argument("--subset-seed", type=int, default=0)
    p.add_argument("--latent-dir", default="/run/media/kim/Lehto/latents")
    p.add_argument("--db-path", default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-val-batches", type=int, default=200,
                   help="Cap on validation batches for speed; <=0 means full val.")
    p.add_argument("--objective", default=None,
                   help="Forward-noising schedule (default: from checkpoint metadata).")
    return p.parse_args()


def main():
    args = parse_args()

    # ROCm env BEFORE torch
    import os
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stable_audio_tools"))
    import rocm_env
    rocm_env.apply_profile("inference")

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from latch_dataset import LatCHDataset
    from stable_audio_tools.models.latch import load_latch_from_checkpoint
    from stable_audio_tools.training.temporal_loss import val_diagnostic_metrics

    raw = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict) or "state_dict" not in raw:
        print(f"FATAL: {args.ckpt}: bare or unrecognised state dict", file=sys.stderr)
        sys.exit(1)
    meta = raw
    feature = args.feature or meta.get("feature_name")
    objective = args.objective or meta.get("noise_schedule") or "rectified_flow"
    standardize = bool(meta.get("standardized"))
    std_mean = float(meta.get("std_mean") or 0.0)
    std_std = float(meta.get("std_std") or 1.0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    head = load_latch_from_checkpoint(str(args.ckpt), device=device)
    head.eval()

    # Same holdout discipline as train_latch.py — shared 5 % fixed across runs.
    clamp_min, clamp_max = (-60.0, 0.0) if str(feature).startswith("rms_energy") else (None, None)
    dataset = LatCHDataset(
        args.latent_dir, info_dir=None,
        target_feature=feature,
        db_path=args.db_path,
        clamp_min=clamp_min, clamp_max=clamp_max,
        smooth_kind=meta.get("smooth_kind", "none"), smooth_width=meta.get("smooth_width", 0.0),
        subset_frac=args.subset_frac, subset_seed=args.subset_seed,
        holdout_frac=args.holdout_frac, holdout_seed=args.holdout_seed,
        target_source="db", npz_root=None,
    )
    if getattr(dataset, "holdout_indices", None):
        val_ds = Subset(dataset, dataset.holdout_indices)
    else:
        val_ds = dataset
    print(f"Eval: feature={feature}  n_val={len(val_ds)}  objective={objective}  "
          f"standardize={standardize}")

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            drop_last=False, num_workers=2)

    # Same forward-noising as train_latch.validate() (rectified_flow only here for brevity)
    def forward_noise_schedule(t, obj):
        if obj == "rectified_flow":
            return (1.0 - t), t
        raise ValueError(f"unhandled objective {obj!r} in eval; extend if you need it")

    def unstd_t(pred):
        return pred if not standardize else pred * std_std + std_mean

    val_gen = torch.Generator(device=device).manual_seed(1234)
    accum = {"val_point_mae": 0.0, "val_deriv_corr": 0.0, "val_multiscale_mae": 0.0, "n": 0}
    seen = 0
    with torch.no_grad():
        for v_lat, v_tgt in val_loader:
            if args.n_val_batches > 0 and seen >= args.n_val_batches:
                break
            v_lat = v_lat.to(device); v_tgt = v_tgt.to(device)
            B = v_lat.size(0)
            tv = torch.rand((B,), device=device, generator=val_gen)
            a, s = forward_noise_schedule(tv, objective)
            z = a.view(B, 1, 1) * v_lat + s.view(B, 1, 1) * torch.randn(
                v_lat.shape, device=device, generator=val_gen)
            with torch.cuda.amp.autocast():
                preds = head(z, tv)
            preds_raw = unstd_t(preds.float())
            diag = val_diagnostic_metrics(preds_raw, v_tgt)
            for k in ("val_point_mae", "val_deriv_corr", "val_multiscale_mae"):
                accum[k] += float(diag[k]) * B
            accum["n"] += B
            seen += 1

    n = max(1, accum["n"])
    print()
    print(f"val_point_mae      = {accum['val_point_mae'] / n:.4f}")
    print(f"val_deriv_corr     = {accum['val_deriv_corr'] / n:.4f}")
    print(f"val_multiscale_mae = {accum['val_multiscale_mae'] / n:.4f}")
    print(f"(averaged over {accum['n']} samples across {seen} batches)")


if __name__ == "__main__":
    main()
