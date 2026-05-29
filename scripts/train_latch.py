import os
import sys

# Apply the AMD/ROCm env BEFORE importing torch. Prefer the `amd:` section of the
# --config YAML (so a run is self-contained); otherwise fall back to rocm_env.yaml's
# `training` profile. (The package/inference path applies rocm_env.yaml on import.)
def _early_arg(flag):
    for i, a in enumerate(sys.argv):
        if a == flag and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None

def _apply_amd_env():
    cfg_path = _early_arg("--config")
    amd = None
    if cfg_path and os.path.exists(cfg_path):
        import yaml
        amd = (yaml.safe_load(open(cfg_path)) or {}).get("amd")
    if amd:
        for k, v in amd.items():
            os.environ.setdefault(k, str(v))
        for k in ("TRITON_CACHE_DIR", "MIOPEN_CUSTOM_CACHE_DIR", "MIOPEN_USER_DB_PATH"):
            p = os.environ.get(k)
            if p:
                os.makedirs(p, exist_ok=True)
        tf = os.environ.get("PYTORCH_TUNABLEOP_FILENAME")
        if tf and os.path.dirname(tf):
            os.makedirs(os.path.dirname(tf), exist_ok=True)
        print(f"[amd] applied {len(amd)} env vars from {cfg_path} (amd: section)")
    else:
        import importlib.util as _ilu
        from pathlib import Path as _Path
        _re = _Path(__file__).resolve().parent.parent / "stable_audio_tools" / "rocm_env.py"
        _spec = _ilu.spec_from_file_location("_sat_rocm_env", _re)
        _m = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_m)
        _m.apply_profile("training")

_apply_amd_env()

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import math

# Use the scripts path to import local modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from latch_dataset import LatCHDataset, _is_ts_feature as _is_ts_feature_name
from latch_model import LatCH

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def forward_noise_schedule(t, objective):
    """Forward (LatCH-F) noising schedule for timestep t in [0, 1].

    MUST match the diffusion model the head will guide, otherwise the head is
    queried on out-of-distribution latents at sampling time. t=0 is clean,
    t=1 is pure noise in both conventions. Returns (alpha_t, sigma_t).

      rectified_flow / rf_denoiser :  z_t = (1-t)·z0 + t·noise   (linear)
      v                            :  z_t = cos(π/2 t)·z0 + sin(π/2 t)·noise  (VP)
    """
    if objective in ("rectified_flow", "rf_denoiser"):
        alpha_t = 1.0 - t
        sigma_t = t
    elif objective == "v":
        alpha_t = torch.cos(math.pi / 2 * t)
        sigma_t = torch.sin(math.pi / 2 * t)
    else:
        raise ValueError(f"Unknown diffusion objective: {objective!r}")
    return alpha_t, sigma_t

def _sample_target_stats(dataset, n: int = 200):
    """Estimate target distribution stats from up to ``n`` random items.

    Returns a dict with float scalars: mean, std, min, max (over channels and frames).
    """
    import random
    import numpy as np
    idxs = list(range(len(dataset)))
    random.shuffle(idxs)
    samples = []
    for i in idxs[:n]:
        try:
            _, target = dataset[i]
            samples.append(target.numpy())
        except Exception:
            continue
    if not samples:
        return {}
    arr = np.concatenate([s.reshape(-1) for s in samples])
    return {
        "mean": float(arr.mean()),
        "std":  float(arr.std()),
        "min":  float(arr.min()),
        "max":  float(arr.max()),
        "p1":   float(np.percentile(arr, 1)),
        "p99":  float(np.percentile(arr, 99)),
        "n_samples": len(samples),
    }


def _slider_spec(stats: dict, clamp=(None, None)) -> dict:
    """UI target-slider bounds + scale.

    Scale is chosen from the distribution (log only for strictly-positive features
    spanning a wide dynamic range, e.g. spectral_flatness/kurtosis; dB/signed stay
    linear). Bounds: a **clamped** feature (e.g. rms dB clamped to [-60,0]) uses the
    clamp as the slider range so guidance can over-drive to the loud/bright extreme,
    not just to p99 (the outliers are on the quiet end, already clamped). Unclamped
    features use robust p1/p99 so outlier frames don't wreck the range.
    """
    if not stats:
        return {}
    p_lo = stats.get("p1", stats.get("min"))
    p_hi = stats.get("p99", stats.get("max"))
    scale = "linear"
    if p_lo is not None and p_hi is not None and p_lo > 0 and p_hi / p_lo > 30:
        scale = "log"
    cmin, cmax = clamp
    lo = cmin if cmin is not None else p_lo
    hi = cmax if cmax is not None else p_hi
    return {"slider_min": lo, "slider_max": hi, "slider_scale": scale}


def _default_kind_for(feature_name: str) -> str:
    """Pick a reasonable default target kind for the UI given the feature."""
    if "beat" in feature_name or "onset" in feature_name or "downbeat" in feature_name:
        return "beat_grid"
    if "rms" in feature_name or "energy" in feature_name:
        return "ramp_up"
    return "constant"


def train(
    latent_dir="/run/media/kim/Lehto/latents",
    info_dir="/run/media/kim/Mantu/ai-music/Goa_Separated_crops",
    target_feature="rms_energy_bass",
    batch_size=8,
    epochs=10,
    lr=1e-4,
    save_dir="latch_weights",
    db_path=None,
    objective="rectified_flow",
    val_frac=0.02,
    val_seed=0,
    tag="",
    num_workers=4,
    use_tensorboard=False,
    use_wandb=False,
    wandb_project="latch",
    test_cfg=None,
    standardize=False,
    smooth_kind="none",
    smooth_width=0.0,
    subset_frac=1.0,
    subset_seed=0,
    holdout_frac=0.0,
    holdout_seed=12345,
    target_source="db",
    npz_root=None,
    preload="none",
    dim=256,
    depth=6,
    num_heads=8,
    optimizer_name="adamw",
    scheduler="none",
    save_best_only=False,
    seed=0,
    compile_model=False,
    compile_mode="default",
    t_injection="film",
    # FusionOpt + TemporalShapeLoss (see docs/superpowers/specs/2026-05-29-fusion-optimiser-design.md)
    loss_name="smoothl1",
    mona_alpha=0.2,
    lambda_deriv=1.0,
    lambda_multi=0.5,
    curriculum_steps=0,
    reset_optimizer=False,
    hot_dtype="fp32",
):
    os.makedirs(save_dir, exist_ok=True)
    # Seed the training RNG (per-step noise t/randn + shuffle/randperm) so runs are
    # reproducible and seed-variance is measurable. (GPU GEMM atomics aren't bit-exact,
    # so same-seed runs are very close but not necessarily identical.)
    if seed is not None:
        import random as _random
        _random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"Seed: {seed}")
    # Checkpoint name stem; --tag adds a version label (e.g. v2) into the filename
    # so it's distinguishable in latch_weights/ and the UI dropdown.
    stem = f"latch_{target_feature}" + (f"_{tag}" if tag else "")
    print(f"Forward-noising schedule: objective={objective} "
          f"(MUST match the diffusion model this head will guide)")

    # dB-floor clamp for rms_energy_* targets: near-silent crops read down to ~-97 dB
    # and dominate the loss. Clamp to [-60, 0] (sub-60 dB is inaudible). Applied in the
    # dataset so train/val targets AND feature_stats are all consistent.
    clamp_min, clamp_max = (-60.0, 0.0) if target_feature.startswith("rms_energy") else (None, None)
    if clamp_min is not None:
        print(f"Target clamp: [{clamp_min}, {clamp_max}] dB (rms feature)")

    # Init Dataset
    dataset = LatCHDataset(
        latent_dir, info_dir,
        target_feature=target_feature,
        db_path=db_path,
        clamp_min=clamp_min, clamp_max=clamp_max,
        smooth_kind=smooth_kind, smooth_width=smooth_width,
        subset_frac=subset_frac, subset_seed=subset_seed,
        holdout_frac=holdout_frac, holdout_seed=holdout_seed,
        target_source=target_source, npz_root=npz_root,
    )
    if len(dataset) == 0:
        print("No valid latent-INFO pairs found. Make sure external drives are mounted.")
        return

    # Peek at one sample to determine out_channels (1 for scalars/1-D ts, 12 for hpcp_ts, etc.)
    sample_latent, sample_target = dataset[0]
    out_channels = sample_target.shape[0]  # (C, T) → C
    print(f"Target '{target_feature}': out_channels={out_channels}, seq_len={sample_target.shape[1]}")

    # Train/val split. With a shared holdout (holdout_frac>0) the val set is a FIXED
    # subset of the full population (identical across subset/full runs) — so val_median
    # is directly comparable for ablations. Otherwise: the usual seeded random split.
    if getattr(dataset, "holdout_indices", None):
        from torch.utils.data import Subset
        train_ds = Subset(dataset, dataset.train_indices)
        val_ds = Subset(dataset, dataset.holdout_indices)
        n_train, n_val = len(train_ds), len(val_ds)
        print(f"Shared-holdout split: {len(train_ds)} train / {len(val_ds)} val "
              f"(holdout_frac={holdout_frac}, seed={holdout_seed}, subset_frac={subset_frac}) "
              f"— val is identical across runs, comparable for ablation")
    else:
        n_val = max(1, int(val_frac * len(dataset)))
        n_train = len(dataset) - n_val
        split_gen = torch.Generator().manual_seed(val_seed)
        train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val], generator=split_gen)
        print(f"Split: {n_train} train / {n_val} val (val_frac={val_frac})")

    # Compute target stats from the TRAIN split only (don't peek at val)
    print("Sampling target stats from train split (up to 200 items)...")
    feature_stats = _sample_target_stats(train_ds, n=200)
    print(f"Target stats: {feature_stats}")
    target_kind_default = _default_kind_for(dataset.bare_feature)
    smoothing_on = smooth_kind not in (None, "none") and smooth_width > 0
    if smoothing_on:
        # A smoothed marker train is a continuous density envelope, not a beat grid.
        target_kind_default = "constant"
        print(f"Smoothing: kind={smooth_kind}, width={smooth_width} → continuous density "
              f"(regression target, kind=constant)")
    print(f"Default target kind for inference: {target_kind_default}")
    slider = _slider_spec(feature_stats, clamp=(clamp_min, clamp_max))  # UI bounds + linear/log scale
    print(f"Slider spec: {slider}")

    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
                        num_workers=num_workers, persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False,
                            num_workers=max(1, num_workers // 2), persistent_workers=num_workers > 0)
    print(f"DataLoader workers: {num_workers} (train) / {max(1, num_workers // 2)} (val)")

    # --- optional one-time preload: materialize (latent, target) into a contiguous
    # tensor so every epoch is GPU-bound (no per-step DB decompress / external-drive
    # reads). `gpu` keeps it in VRAM (fastest; subset must fit); `ram` keeps it in
    # host RAM and streams batches. Pays a single read+decompress pass up front.
    def _preload_tensors(subset, where, tag):
        dl = DataLoader(subset, batch_size=512, shuffle=False,
                        num_workers=num_workers, persistent_workers=False)
        lats, tgts = [], []
        for lat, tgt in tqdm(dl, desc=f"preload[{tag}]"):
            lats.append(lat); tgts.append(tgt)
        L = torch.cat(lats, 0); T = torch.cat(tgts, 0)
        need_gb = (L.numel() + T.numel()) * 4 / 1e9
        if where == "gpu" and torch.cuda.is_available():
            free = torch.cuda.mem_get_info()[0]
            if (L.numel() + T.numel()) * 4 < free * 0.7:
                L, T = L.to(device), T.to(device)
                print(f"preload[{tag}]: {tuple(L.shape)} on GPU ({need_gb:.2f} GB)")
                return L, T
            print(f"preload[{tag}]: {need_gb:.2f} GB > 70% of free VRAM "
                  f"({free/1e9:.2f} GB) → falling back to RAM")
        print(f"preload[{tag}]: {tuple(L.shape)} in RAM ({need_gb:.2f} GB)")
        return L, T

    pre_train = pre_val = None
    if preload and preload != "none":
        print(f"Preloading dataset into {preload.upper()} (one-time pass)...")
        pre_train = _preload_tensors(train_ds, preload, "train")
        pre_val = _preload_tensors(val_ds, preload, "val")

    # Init Model — out_channels adapts to the target feature shape
    model = LatCH(in_channels=64, out_channels=out_channels, dim=dim, depth=depth,
                  num_heads=num_heads, t_injection=t_injection).to(device)
    raw_model = model  # uncompiled handle — used for state_dict() to keep checkpoints prefix-free
    print(f"LatCH head: dim={dim}, depth={depth}, num_heads={num_heads}, t_injection={t_injection} "
          f"({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")
    if compile_model:
        model = torch.compile(model, mode=compile_mode)
        print(f"torch.compile: mode={compile_mode} (1st-iter warmup will spike)")
    
    # Optimizer (bracketing experiment): adamw | lion | prodigy | adam8bit | schedulefree | fusion
    _is_sf = optimizer_name in ("schedulefree", "fusion")
    _is_fusion = optimizer_name == "fusion"
    if optimizer_name == "lion":
        from lion_pytorch import Lion
        optimizer = Lion(model.parameters(), lr=lr, weight_decay=1e-2)
    elif optimizer_name == "prodigy":
        from prodigyopt import Prodigy
        # LR-free: it adapts d; lr is a base multiplier (use 1.0) and wants a scheduler.
        optimizer = Prodigy(model.parameters(), lr=lr, weight_decay=0.0,
                            safeguard_warmup=True, use_bias_correction=True)
    elif optimizer_name == "adam8bit":
        from bitsandbytes.optim import AdamW8bit
        optimizer = AdamW8bit(model.parameters(), lr=lr)
    elif optimizer_name == "schedulefree":
        from schedulefree import AdamWScheduleFree
        # No external LR schedule by design; needs .train()/.eval() toggling (handled below).
        warmup = max(100, int(0.05 * epochs * max(1, n_train // batch_size)))
        optimizer = AdamWScheduleFree(model.parameters(), lr=lr, warmup_steps=warmup)
    elif optimizer_name == "fusion":
        # FusionOpt: bifurcated Muon+MONA+KL-Shampoo (spectral) + ScheduleFree-AdamW (scalar)
        # with shared Polyak step size. Spec §3.
        from stable_audio_tools.training.fusion_opt import FusionOpt
        from stable_audio_tools.training.fusion_groups import (
            build_fusion_param_groups, summarise_groups,
        )
        warmup = max(100, int(0.05 * epochs * max(1, n_train // batch_size)))
        groups = build_fusion_param_groups(raw_model, spectral_wd=0.01, scalar_wd=0.0)
        optimizer = FusionOpt(
            groups,
            lr=lr,
            mona_alpha=mona_alpha,
            warmup_steps=warmup,
            hot_dtype=hot_dtype,
        )
        print("FusionOpt groups:")
        print(summarise_groups(groups))
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    sched = None
    if scheduler == "cosine" and not _is_sf:
        sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if scheduler == "cosine" and _is_fusion:
        raise ValueError("--scheduler cosine is incompatible with --optimizer fusion "
                         "(Polyak step is schedule-free)")
    print(f"Optimizer: {optimizer_name} (lr={lr}); "
          f"scheduler: {'none (schedule-free)' if _is_sf else scheduler}")
    
    # Decide loss based on feature type (use dataset.bare_feature for matching)
    bare = dataset.bare_feature
    if "hpcp" in bare or "chroma" in bare:
        def cosine_loss(preds, targets):
            p_norm = preds / (preds.norm(dim=1, keepdim=True) + 1e-8)
            t_norm = targets / (targets.norm(dim=1, keepdim=True) + 1e-8)
            return (1.0 - (p_norm * t_norm).sum(dim=1)).mean()
        criterion = cosine_loss
        loss_type = "cosine"
    elif ("beat" in bare or "onset" in bare) and not smoothing_on:
        def sparse_weighted_bce(preds, targets, threshold=0.2):
            loss = nn.BCEWithLogitsLoss(reduction='none')(preds, targets)
            mask = targets > threshold
            loss_active   = loss[mask].mean()  if mask.sum()    > 0 else preds.new_tensor(0.0)
            loss_inactive = loss[~mask].mean() if (~mask).sum() > 0 else preds.new_tensor(0.0)
            return (loss_active + loss_inactive) / 2
        criterion = sparse_weighted_bce
        loss_type = "bce_logits"  # head emits logits; inference guidance must match
    else:
        # Robust regression loss (Huber/SmoothL1). Optionally standardize the target to
        # zero-mean/unit-std so low-variance features (flatness, tonic_strength, …) can't
        # collapse to predicting the mean; un-standardized at inference via metadata.
        std = float(feature_stats.get("std") or 1.0)
        if standardize:
            std_mean = float(feature_stats.get("mean") or 0.0)
            std_std = max(std, 1e-6)
            huber_beta = 1.0  # standardized units
            print(f"Standardized targets: (x-{std_mean:.4f})/{std_std:.4f}; SmoothL1 beta=1.0")
        else:
            huber_beta = max(1e-3, round(0.5 * std, 4))
            print(f"Robust loss: SmoothL1 (Huber) beta={huber_beta}")
        if loss_name == "temporal":
            # TemporalShapeLoss: L_point + lambda_d * L_deriv + lambda_m * L_multi
            # See docs/superpowers/specs/2026-05-29-fusion-optimiser-design.md §4
            from stable_audio_tools.training.temporal_loss import TemporalShapeLoss
            criterion = TemporalShapeLoss(
                huber_beta=huber_beta,
                lambda_deriv=lambda_deriv,
                lambda_multi=lambda_multi,
                point_loss="auto",
                curriculum_steps=curriculum_steps,
            )
            loss_type = "temporal_shape"
            print(f"Temporal-shape loss: lambda_deriv={lambda_deriv}, lambda_multi={lambda_multi}, "
                  f"curriculum_steps={curriculum_steps}")
        else:
            criterion = nn.SmoothL1Loss(beta=huber_beta)
            loss_type = "smooth_l1"

    huber_beta = locals().get("huber_beta")  # None unless regression branch set it
    std_mean = locals().get("std_mean")      # None unless standardize (regression only)
    std_std = locals().get("std_std")

    def _std_t(tgt):
        """Standardize a target tensor (identity unless --standardize on a regression head)."""
        return tgt if std_mean is None else (tgt - std_mean) / std_std

    def _unstd_t(pred):
        """Inverse of _std_t — pred in raw feature units (for diagnostic MAE).
        Identity when standardize is off."""
        return pred if std_mean is None else pred * std_std + std_mean

    def validate():
        """Held-out loss with FIXED noise/timesteps so it's comparable across epochs.

        Returns (val_mean, val_median, diag) where diag is a dict of cross-loss
        comparable metrics in RAW feature units:
          - val_point_mae:      mean |pred_raw - target_raw|
          - val_deriv_corr:     Pearson(diff(pred_raw), diff(target_raw))
          - val_multiscale_mae: avg over scales of |pool(pred_raw) - pool(target_raw)|
        """
        from stable_audio_tools.training.temporal_loss import val_diagnostic_metrics
        model.eval()
        if _is_sf:
            optimizer.eval()    # schedule-free: switch to AVERAGED weights for val + checkpoint
                                # (left in eval; the next epoch's optimizer.train() resumes)
        vg = torch.Generator(device=device).manual_seed(1234)
        losses = []
        diag_accum = {"val_point_mae": 0.0, "val_deriv_corr": 0.0,
                      "val_multiscale_mae": 0.0, "_n": 0}
        def _vbatches():
            if pre_val is not None:
                Lv, Tv = pre_val
                for i in range(0, Lv.shape[0], batch_size):
                    yield Lv[i:i + batch_size], Tv[i:i + batch_size]
            else:
                yield from val_loader
        with torch.no_grad():
            for v_lat, v_tgt in _vbatches():
                v_lat = v_lat.to(device); v_tgt = v_tgt.to(device)
                Bv = v_lat.size(0)
                tv = torch.rand((Bv,), device=device, generator=vg)
                a, s = forward_noise_schedule(tv, objective)
                z = a.view(Bv, 1, 1) * v_lat + s.view(Bv, 1, 1) * torch.randn(
                    v_lat.shape, device=device, generator=vg)
                with torch.cuda.amp.autocast():
                    preds = model(z, tv)
                    losses.append(criterion(preds, _std_t(v_tgt)).item())
                # Raw-unit diagnostics for cross-loss comparison
                try:
                    diag = val_diagnostic_metrics(_unstd_t(preds), v_tgt)
                    diag_accum["val_point_mae"] += float(diag["val_point_mae"]) * Bv
                    diag_accum["val_deriv_corr"] += float(diag["val_deriv_corr"]) * Bv
                    diag_accum["val_multiscale_mae"] += float(diag["val_multiscale_mae"]) * Bv
                    diag_accum["_n"] += Bv
                except Exception:
                    pass  # diagnostics are best-effort; never block validation
        model.train()
        if not losses:
            return float("nan"), float("nan"), {}
        n = max(1, diag_accum["_n"])
        diag = {
            "val_point_mae":      diag_accum["val_point_mae"]      / n,
            "val_deriv_corr":     diag_accum["val_deriv_corr"]     / n,
            "val_multiscale_mae": diag_accum["val_multiscale_mae"] / n,
        }
        return float(np.mean(losses)), float(np.median(losses)), diag

    print(f"Starting training on feature: {target_feature} ({n_train} train items).")

    best_val_median = float("inf")

    # --- optional experiment logging (TensorBoard / W&B, opt-in) ---
    run_cfg = {
        "feature": target_feature, "objective": objective, "loss_type": loss_type,
        "huber_beta": huber_beta, "target_clamp": [clamp_min, clamp_max],
        "lr": lr, "batch_size": batch_size, "epochs": epochs, "val_frac": val_frac,
        "out_channels": out_channels, "tag": tag,
        "smooth_kind": smooth_kind, "smooth_width": smooth_width, "subset_frac": subset_frac,
        "holdout_frac": holdout_frac, "preload": preload,
        "dim": dim, "depth": depth, "num_heads": num_heads,
        "optimizer": optimizer_name, "scheduler": scheduler,
        **{f"stat_{k}": v for k, v in feature_stats.items()},
    }
    tb = None
    if use_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(os.path.join("runs", stem))
        print(f"TensorBoard -> runs/{stem}  (view: tensorboard --logdir runs)")
    wb = None
    if use_wandb:
        try:
            import wandb
            wb = wandb.init(project=wandb_project, name=stem, config=run_cfg)
            print(f"W&B -> project '{wandb_project}', run '{stem}'")
        except Exception as e:
            print(f"[wandb] disabled: {e}")

    def log_scalars(d, step):
        if tb is not None:
            for k, v in d.items():
                tb.add_scalar(k, v, step)
        if wb is not None:
            wb.log(d, step=step)

    # --- optional periodic test inference: generate audio with the in-training head ---
    # Enabled when test_cfg provides the base diffusion model (model_config + ckpt_path).
    # Uses the just-saved epoch checkpoint as the LatCH guide so metadata/criterion match.
    _base = {}
    def run_test_inference(epoch, ckpt_path, step):
        if not test_cfg or not test_cfg.get("model_config") or not test_cfg.get("ckpt_path"):
            return
        if (epoch + 1) % int(test_cfg.get("every_epochs", 1)) != 0:
            return
        try:
            import json as _json, soundfile as _sf
            from stable_audio_tools.models.factory import create_model_from_config
            from stable_audio_tools.models.utils import load_ckpt_state_dict
            from stable_audio_tools.inference.generation import generate_diffusion_cond
            if "model" not in _base:
                with open(test_cfg["model_config"]) as f:
                    mc = _json.load(f)
                bm = create_model_from_config(mc)
                bm.load_state_dict(load_ckpt_state_dict(test_cfg["ckpt_path"]))
                _base.update(model=bm.half().to(device).eval(),
                             sr=mc["sample_rate"], ss=mc["sample_size"])
                print(f"  [test inference: base model loaded from {test_cfg['ckpt_path']}]")
            bm, sr, ss = _base["model"], _base["sr"], _base["ss"]
            win = test_cfg.get("window", [0.5, 1.0])
            n_steps = int(test_cfg.get("steps", 50))
            # target_value may be a scalar OR a bracket list -> one clip per value
            tv = test_cfg.get("target_value")
            if tv is None or tv == "auto":
                # auto-bracket from this head's own slider range -> one config fits any feature
                lo = slider.get("slider_min", feature_stats.get("min", 0.0))
                hi = slider.get("slider_max", feature_stats.get("max", 1.0))
                values = [lo + (hi - lo) * f for f in (0.25, 0.5, 0.85)]
            else:
                values = tv if isinstance(tv, (list, tuple)) else [tv]
            audio_logs = {}
            for val in values:
                audio = generate_diffusion_cond(
                    bm, steps=n_steps, cfg_scale=float(test_cfg.get("cfg_scale", 7.0)),
                    conditioning=[{"prompt": test_cfg.get("prompt", ""),
                                   "seconds_total": round(ss / sr)}],
                    sample_size=ss, seed=int(test_cfg.get("seed", 42)),
                    device=device, sigma_max=1.0,
                    latch_configs=[{
                        "model_path": ckpt_path,
                        "kind": test_cfg.get("kind", target_kind_default),
                        "value": float(val),
                        "weight": float(test_cfg.get("weight", 1.0)),
                        "start_pct": float(win[0]), "end_pct": float(win[1]),
                    }],
                    latch_hparams={
                        "rho": float(test_cfg.get("rho", test_cfg.get("gain", 5.0))),
                        "mu": float(test_cfg.get("mu", test_cfg.get("gain", 5.0))),
                        "gamma": float(test_cfg.get("gamma", 0.3)),
                        "n_iter": int(test_cfg.get("n_iter", 4)),
                        "log_norms": False,
                    },
                )
                wav = audio.squeeze(0).cpu().float().numpy()  # [C, N]
                vtag = f"{float(val):g}".replace("-", "m").replace(".", "p")
                out_wav = os.path.join(save_dir, f"{stem}_test_ep{epoch+1}_tgt{vtag}.wav")
                _sf.write(out_wav, wav.T, sr)
                print(f"  ↳ test inference ({n_steps} steps, target={val}) → {out_wav}")
                if wb is not None:
                    import wandb
                    audio_logs[f"test/audio_tgt{vtag}"] = wandb.Audio(
                        wav.mean(0), sample_rate=sr, caption=f"ep{epoch+1} target={val}")
            if wb is not None and audio_logs:
                import wandb  # noqa: F401
                wb.log(audio_logs, step=step)
        except Exception as e:
            print(f"  [test inference skipped: {e}]")

    global_step = 0
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(epochs):
        model.train()
        if _is_sf:
            optimizer.train()   # schedule-free: live weights for training (resumes after each eval)
        total_loss = 0.0

        if pre_train is not None:
            Lt, Tt = pre_train
            Nt = Lt.shape[0]
            perm = torch.randperm(Nt, device=Lt.device)
            n_batches = (Nt + batch_size - 1) // batch_size
            def _train_batches():
                for i in range(0, Nt, batch_size):
                    b = perm[i:i + batch_size]
                    yield Lt[b], Tt[b]
            pbar = tqdm(_train_batches(), total=n_batches, desc=f"Epoch {epoch+1}/{epochs}")
        else:
            n_batches = len(loader)
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        for latents, targets in pbar:
            latents = latents.to(device) # [B, 64, T]
            targets = targets.to(device) # [B, 1, T]
            
            B = latents.size(0)
            
            # Sample timesteps
            t = torch.rand((B,), device=device) # t in [0, 1]
            
            # Forward simulated noise (LatCH-F) — schedule must match the model
            alpha_t, sigma_t = forward_noise_schedule(t, objective)
            alpha_t = alpha_t.view(B, 1, 1)
            sigma_t = sigma_t.view(B, 1, 1)
            
            noise = torch.randn_like(latents)
            z_t = alpha_t * latents + sigma_t * noise
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                # Predict controls directly from noisy latent `z_t`
                preds = model(z_t, t)
                loss = criterion(preds, _std_t(targets))

            scaler.scale(loss).backward()
            if _is_fusion:
                # FusionOpt needs the loss for its Polyak step size (stays on-device)
                optimizer.set_loss(loss)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            global_step += 1
            pbar.set_postfix({'loss': loss.item()})
            if (tb is not None or wb is not None) and global_step % 50 == 0:
                step_log = {"train/loss_step": loss.item()}
                # Component breakdown if TemporalShapeLoss is in use
                comps = getattr(criterion, "last_components", None)
                if comps:
                    for k, v in comps.items():
                        step_log[f"train/{k}"] = float(v)
                log_scalars(step_log, global_step)
            
        avg_loss = total_loss / n_batches
        val_mean, val_median, val_diag = validate()
        print(f"Epoch {epoch+1}: train_avg={avg_loss:.4f}  "
              f"val_mean={val_mean:.4f}  val_median={val_median:.4f}  "
              f"val_point_mae={val_diag.get('val_point_mae', float('nan')):.4f}  "
              f"val_deriv_corr={val_diag.get('val_deriv_corr', float('nan')):.4f}")
        ep_log = {
            "train/avg": avg_loss, "val/mean": val_mean,
            "val/median": val_median, "epoch": epoch + 1,
        }
        for k, v in val_diag.items():
            ep_log[f"val/{k}"] = v
        if _is_fusion:
            ep_log.update(optimizer.diagnostic_summary())
        log_scalars(ep_log, global_step)

        checkpoint = {
            "state_dict": raw_model.state_dict(),
            "feature_name": target_feature,
            "feature_stats": feature_stats,
            "target_kind_default": target_kind_default,
            "noise_schedule": objective,  # forward-noising convention the head expects
            "loss_type": loss_type,       # mse | smooth_l1 | bce_logits | cosine | temporal_shape
            "huber_beta": huber_beta,     # SmoothL1 beta (None unless loss_type==smooth_l1)
            "target_clamp": [clamp_min, clamp_max],
            "standardized": std_mean is not None,  # head predicts (x-mean)/std space
            "std_mean": std_mean,         # un-standardize at inference: x = pred*std + mean
            "std_std": std_std,
            "smooth_kind": smooth_kind,   # envelope smoothing applied to the marker target
            "smooth_width": smooth_width,
            "subset_frac": subset_frac,
            "holdout_frac": holdout_frac, # shared fixed holdout used as val (0 = none)
            "dim": dim, "depth": depth, "num_heads": num_heads,
            "t_injection": t_injection,   # 'concat' (legacy, T=257) | 'film' (T=256, FA-aligned)
            "optimizer": optimizer_name, "scheduler": scheduler, "lr": lr, "seed": seed,
            "loss_name": loss_name,       # smoothl1 | temporal
            "fusion_config": {
                "mona_alpha": mona_alpha, "lambda_deriv": lambda_deriv,
                "lambda_multi": lambda_multi, "curriculum_steps": curriculum_steps,
            } if _is_fusion else None,
            **slider,                     # slider_min / slider_max (p1/p99) + slider_scale (linear|log)
            "val_mean": val_mean,
            "val_median": val_median,
            **{f"val_{k}": v for k, v in val_diag.items()},  # raw-unit diagnostics
        }
        # FusionOpt: the SF averaged iterate x_t is the deployable model; the live z_t
        # is in raw_model.state_dict() above and saved alongside for resume.
        if _is_fusion:
            checkpoint["averaged_state_dict"] = optimizer.average_state_dict()
        ep_ckpt = os.path.join(save_dir, f"{stem}_ep{epoch+1}.pt")
        if not save_best_only:            # keep only _best.pt during sweeps (disk)
            torch.save(checkpoint, ep_ckpt)
        if sched is not None:
            sched.step()

        # Keep a best-by-val-median copy (the outlier-robust convergence signal)
        if val_median == val_median and val_median < best_val_median:  # not NaN and improved
            best_val_median = val_median
            torch.save(checkpoint, os.path.join(save_dir, f"{stem}_best.pt"))
            print(f"  ↳ new best (val_median={val_median:.4f}) → {stem}_best.pt")
            log_scalars({"best/val_median": best_val_median}, global_step)

        run_test_inference(epoch, ep_ckpt, global_step)

    # When --save-best-only, the per-epoch ep_ckpt was skipped; save the final epoch's
    # state as _last.pt so we keep BOTH best (early stop) and last (full-training) heads.
    if save_best_only:
        last_path = os.path.join(save_dir, f"{stem}_last.pt")
        torch.save(checkpoint, last_path)
        print(f"  ↳ saved final epoch → {stem}_last.pt")

    if tb is not None:
        tb.close()
    if wb is not None:
        wb.finish()

if __name__ == "__main__":
    import argparse
    import yaml
    parser = argparse.ArgumentParser(
        description="Train a LatCH head. Settings can come from a --config YAML; "
                    "explicit CLI flags override the YAML."
    )
    parser.add_argument("--config", type=str, default=None,
                        help="YAML with training + test-inference settings (see latch_train.yaml). "
                             "CLI flags override matching YAML keys.")
    # Overridable settings default to None so we can tell whether the user set them
    # on the CLI (CLI wins) vs. falling back to the YAML, then to the built-in default.
    parser.add_argument("--feature", type=str, default=None,
                        help="bare feature name, e.g. rms_energy_bass, spectral_flatness, hpcp")
    parser.add_argument("--epochs",     type=int,   default=None)
    parser.add_argument("--batch-size", type=int,   default=None)
    parser.add_argument("--lr",         type=float, default=None)
    parser.add_argument("--latent-dir", type=str,   default=None)
    parser.add_argument("--db-path",    type=str,   default=None)
    parser.add_argument("--target-source", type=str, default=None,
                        choices=["db", "whole_track"],
                        help="'db' = legacy per-crop TimeseriesDB; 'whole_track' = slice "
                             "the crop window from a whole-track .TIMESERIES.npz (per-stem "
                             "onset envelopes, raw madmom soft activations, etc.)")
    parser.add_argument("--npz-root", type=str, default=None,
                        help="Whole-track npz dir (default /run/media/kim/Lehto/timeseries)")
    parser.add_argument("--save-dir",   type=str,   default=None)
    parser.add_argument("--val-frac",   type=float, default=None)
    parser.add_argument("--num-workers", type=int,  default=None)
    parser.add_argument("--tag",        type=str,   default=None,
                        help="version label in the checkpoint filename, e.g. v2")
    parser.add_argument("--objective",  type=str,   default=None,
                        choices=["rectified_flow", "rf_denoiser", "v"],
                        help="forward-noising schedule; MUST match the diffusion model "
                             "(Small = rectified_flow)")
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--standardize", action="store_true",
                        help="train regression heads on zero-mean/unit-std targets "
                             "(helps low-variance features avoid mean-collapse)")
    parser.add_argument("--smooth-kind", type=str, default=None,
                        choices=["none", "gaussian", "linear", "lowpass", "beat_weighted"],
                        help="smooth a sparse marker feature (beat/onset) into a continuous "
                             "density envelope; routes it to a regression loss")
    parser.add_argument("--smooth-width", type=float, default=None,
                        help="smoothing width in latent frames (~46 ms each), e.g. 3")
    parser.add_argument("--subset-frac", type=float, default=None,
                        help="deterministically train on a fraction of the data (e.g. 0.3) — "
                             "for quick smoothing-kind bake-offs; seeded so it's reproducible")
    parser.add_argument("--subset-seed", type=int, default=None)
    parser.add_argument("--holdout-frac", type=float, default=None,
                        help="reserve a fixed holdout (e.g. 0.05) used as the val set; "
                             "IDENTICAL across runs (separate seed), so val_median is "
                             "comparable for subset-vs-full ablations")
    parser.add_argument("--holdout-seed", type=int, default=None)
    parser.add_argument("--preload", type=str, default=None,
                        choices=["none", "ram", "gpu"],
                        help="preload (latent,target) into RAM or VRAM once so epochs are "
                             "GPU-bound (no per-step DB/disk). gpu requires the set to fit VRAM")
    parser.add_argument("--dim", type=int, default=None,
                        help="head width (256-aligned: 256/512/768/1024/1536). Bigger = more "
                             "capacity but ~linearly slower (see latch_shape_bench.py)")
    parser.add_argument("--depth", type=int, default=None, help="head transformer depth")
    parser.add_argument("--num-heads", type=int, default=None, help="attention heads")
    parser.add_argument("--optimizer", type=str, default=None,
                        choices=["adamw", "lion", "prodigy", "adam8bit", "schedulefree", "fusion"],
                        help="adamw | lion (~3-10x lower LR) | prodigy (LR-free, lr=1.0) | "
                             "adam8bit (bnb, low-mem) | schedulefree (no LR schedule; .train()/.eval() handled) | "
                             "fusion (Muon+MONA+KL-Shampoo+SF+; spec docs/superpowers/specs/2026-05-29-fusion-optimiser-design.md)")
    parser.add_argument("--loss", type=str, default=None,
                        choices=["smoothl1", "temporal"],
                        help="regression loss: smoothl1 (default) | temporal "
                             "(SmoothL1 + derivative + multi-scale; spec §4). "
                             "Ignored for hpcp/beat/onset features (use existing branch).")
    parser.add_argument("--mona-alpha", type=float, default=None,
                        help="FusionOpt MONA curvature-injection strength (default 0.2). "
                             "The seed-variance lever; higher = stronger deflection from sharp minima.")
    parser.add_argument("--lambda-deriv", type=float, default=None,
                        help="TemporalShapeLoss: weight on the derivative term (default 1.0)")
    parser.add_argument("--lambda-multi", type=float, default=None,
                        help="TemporalShapeLoss: weight on the multi-scale L1 term (default 0.5)")
    parser.add_argument("--curriculum-steps", type=int, default=None,
                        help="TemporalShapeLoss: linear warmup of lambda_deriv, lambda_multi "
                             "from 0 to default over this many steps (default 0 = off)")
    parser.add_argument("--reset-optimizer", action="store_true",
                        help="when resuming FusionOpt from a legacy AdamW checkpoint, drop "
                             "the AdamW optimizer state and re-init z_t = x_t = current weights")
    parser.add_argument("--hot-dtype", type=str, default=None,
                        choices=["fp32", "fp16"],
                        help="FusionOpt spectral hot-path dtype: fp32 (default, safer) or "
                             "fp16 (faster on RDNA4; ~3-5x NS5 speedup; profile shows NS5 "
                             "is 42%% of step time, see LATCH_RESULTS.txt §20).")
    parser.add_argument("--scheduler", type=str, default=None, choices=["none", "cosine"],
                        help="LR schedule: none (constant) | cosine (anneal over --epochs)")
    parser.add_argument("--save-best-only", action="store_true",
                        help="only keep <stem>_best.pt (no per-epoch checkpoints) — for sweeps")
    parser.add_argument("--seed", type=int, default=None,
                        help="training RNG seed (reproducible runs; vary it for seed-variance)")
    parser.add_argument("--compile", dest="compile_model", action="store_true",
                        help="wrap the head with torch.compile (Inductor); 1st iter slow, rest faster")
    parser.add_argument("--compile-mode", type=str, default="default",
                        choices=["default", "reduce-overhead", "max-autotune"],
                        help="torch.compile mode (default | reduce-overhead | max-autotune)")
    parser.add_argument("--t-injection", type=str, default=None,
                        choices=["concat", "film", "adaln_zero"],
                        help="timestep injection: 'film' (default, T=256, FA-aligned) | "
                             "'adaln_zero' (DiT-style per-block, +~50%% params, stronger conditioning) | "
                             "'concat' (legacy, T=257, prepended timestep token)")
    args = parser.parse_args()

    ycfg = {}
    if args.config:
        with open(args.config) as f:
            ycfg = yaml.safe_load(f) or {}
        print(f"Loaded config from {args.config}")

    def pick(cli, key, default):
        return cli if cli is not None else ycfg.get(key, default)

    train(
        latent_dir=pick(args.latent_dir, "latent_dir", "/run/media/kim/Lehto/latents"),
        target_feature=pick(args.feature, "feature", "rms_energy_bass"),
        epochs=pick(args.epochs, "epochs", 10),
        batch_size=pick(args.batch_size, "batch_size", 8),
        lr=pick(args.lr, "lr", 1e-4),
        db_path=pick(args.db_path, "db_path", None),
        save_dir=pick(args.save_dir, "save_dir", "latch_weights"),
        objective=pick(args.objective, "objective", "rectified_flow"),
        val_frac=pick(args.val_frac, "val_frac", 0.02),
        tag=pick(args.tag, "tag", ""),
        num_workers=pick(args.num_workers, "num_workers", 4),
        use_tensorboard=args.tensorboard or bool(ycfg.get("tensorboard", False)),
        standardize=args.standardize or bool(ycfg.get("standardize", False)),
        use_wandb=args.wandb or bool(ycfg.get("wandb", False)),
        wandb_project=pick(args.wandb_project, "wandb_project", "latch"),
        test_cfg=ycfg.get("test"),
        smooth_kind=pick(args.smooth_kind, "smooth_kind", "none"),
        smooth_width=pick(args.smooth_width, "smooth_width", 0.0),
        subset_frac=pick(args.subset_frac, "subset_frac", 1.0),
        subset_seed=pick(args.subset_seed, "subset_seed", 0),
        holdout_frac=pick(args.holdout_frac, "holdout_frac", 0.0),
        holdout_seed=pick(args.holdout_seed, "holdout_seed", 12345),
        target_source=pick(args.target_source, "target_source", "db"),
        npz_root=pick(args.npz_root, "npz_root", None),
        preload=pick(args.preload, "preload", "none"),
        dim=pick(args.dim, "dim", 256),
        depth=pick(args.depth, "depth", 6),
        num_heads=pick(args.num_heads, "num_heads", 8),
        optimizer_name=pick(args.optimizer, "optimizer", "adamw"),
        scheduler=pick(args.scheduler, "scheduler", "none"),
        save_best_only=args.save_best_only or bool(ycfg.get("save_best_only", False)),
        seed=pick(args.seed, "seed", 0),
        compile_model=args.compile_model or bool(ycfg.get("compile", False)),
        compile_mode=pick(args.compile_mode, "compile_mode", "default"),
        t_injection=pick(args.t_injection, "t_injection", "film"),
        # FusionOpt + TemporalShapeLoss
        loss_name=pick(args.loss, "loss", "smoothl1"),
        mona_alpha=pick(args.mona_alpha, "mona_alpha", 0.2),
        lambda_deriv=pick(args.lambda_deriv, "lambda_deriv", 1.0),
        lambda_multi=pick(args.lambda_multi, "lambda_multi", 0.5),
        curriculum_steps=pick(args.curriculum_steps, "curriculum_steps", 0),
        reset_optimizer=args.reset_optimizer or bool(ycfg.get("reset_optimizer", False)),
        hot_dtype=pick(args.hot_dtype, "hot_dtype", "fp32"),
    )
