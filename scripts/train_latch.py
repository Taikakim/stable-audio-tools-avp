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


def _slider_spec(stats: dict) -> dict:
    """Robust UI target-slider bounds + scale, from the sampled distribution.

    Uses p1/p99 (not min/max) so outlier frames don't wreck the auto-range, and
    picks a logarithmic scale only for strictly-positive features spanning a wide
    dynamic range (e.g. spectral_flatness, spectral_kurtosis). dB / signed /
    zero-touching features stay linear. Consumed by the gradio target slider.
    """
    if not stats:
        return {}
    lo = stats.get("p1", stats.get("min"))
    hi = stats.get("p99", stats.get("max"))
    scale = "linear"
    if lo is not None and hi is not None and lo > 0 and hi / lo > 30:
        scale = "log"
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
):
    os.makedirs(save_dir, exist_ok=True)
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
    )
    if len(dataset) == 0:
        print("No valid latent-INFO pairs found. Make sure external drives are mounted.")
        return

    # Peek at one sample to determine out_channels (1 for scalars/1-D ts, 12 for hpcp_ts, etc.)
    sample_latent, sample_target = dataset[0]
    out_channels = sample_target.shape[0]  # (C, T) → C
    print(f"Target '{target_feature}': out_channels={out_channels}, seq_len={sample_target.shape[1]}")

    # Deterministic train/val split (seeded so val never leaks into train across runs)
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
    print(f"Default target kind for inference: {target_kind_default}")
    slider = _slider_spec(feature_stats)   # robust UI bounds + linear/log scale
    print(f"Slider spec: {slider}")

    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
                        num_workers=num_workers, persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False,
                            num_workers=max(1, num_workers // 2), persistent_workers=num_workers > 0)
    print(f"DataLoader workers: {num_workers} (train) / {max(1, num_workers // 2)} (val)")

    # Init Model — out_channels adapts to the target feature shape
    model = LatCH(in_channels=64, out_channels=out_channels, dim=256, depth=6, num_heads=8).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    # Decide loss based on feature type (use dataset.bare_feature for matching)
    bare = dataset.bare_feature
    if "hpcp" in bare or "chroma" in bare:
        def cosine_loss(preds, targets):
            p_norm = preds / (preds.norm(dim=1, keepdim=True) + 1e-8)
            t_norm = targets / (targets.norm(dim=1, keepdim=True) + 1e-8)
            return (1.0 - (p_norm * t_norm).sum(dim=1)).mean()
        criterion = cosine_loss
        loss_type = "cosine"
    elif "beat" in bare or "onset" in bare:
        def sparse_weighted_bce(preds, targets, threshold=0.2):
            loss = nn.BCEWithLogitsLoss(reduction='none')(preds, targets)
            mask = targets > threshold
            loss_active   = loss[mask].mean()  if mask.sum()    > 0 else preds.new_tensor(0.0)
            loss_inactive = loss[~mask].mean() if (~mask).sum() > 0 else preds.new_tensor(0.0)
            return (loss_active + loss_inactive) / 2
        criterion = sparse_weighted_bce
        loss_type = "bce_logits"  # head emits logits; inference guidance must match
    else:
        # Robust regression loss (Huber/SmoothL1): quadratic for small errors, linear
        # beyond beta, so outliers (e.g. clamped silent crops) don't dominate. beta is
        # auto-scaled to the feature (half its std) so it's in the right units.
        std = float(feature_stats.get("std") or 1.0)
        huber_beta = max(1e-3, round(0.5 * std, 4))
        criterion = nn.SmoothL1Loss(beta=huber_beta)
        loss_type = "smooth_l1"
        print(f"Robust loss: SmoothL1 (Huber) beta={huber_beta}")

    huber_beta = locals().get("huber_beta")  # None unless regression branch set it

    def validate():
        """Held-out loss with FIXED noise/timesteps so it's comparable across epochs."""
        model.eval()
        vg = torch.Generator(device=device).manual_seed(1234)
        losses = []
        with torch.no_grad():
            for v_lat, v_tgt in val_loader:
                v_lat = v_lat.to(device); v_tgt = v_tgt.to(device)
                Bv = v_lat.size(0)
                tv = torch.rand((Bv,), device=device, generator=vg)
                a, s = forward_noise_schedule(tv, objective)
                z = a.view(Bv, 1, 1) * v_lat + s.view(Bv, 1, 1) * torch.randn(
                    v_lat.shape, device=device, generator=vg)
                with torch.cuda.amp.autocast():
                    losses.append(criterion(model(z, tv), v_tgt).item())
        model.train()
        if not losses:
            return float("nan"), float("nan")
        return float(np.mean(losses)), float(np.median(losses))

    print(f"Starting training on feature: {target_feature} ({n_train} train items).")

    best_val_median = float("inf")

    # --- optional experiment logging (TensorBoard / W&B, opt-in) ---
    run_cfg = {
        "feature": target_feature, "objective": objective, "loss_type": loss_type,
        "huber_beta": huber_beta, "target_clamp": [clamp_min, clamp_max],
        "lr": lr, "batch_size": batch_size, "epochs": epochs, "val_frac": val_frac,
        "out_channels": out_channels, "tag": tag,
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
            tv = test_cfg.get("target_value", feature_stats.get("max", 1.0))
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
        total_loss = 0.0
        
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
                loss = criterion(preds, targets)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            global_step += 1
            pbar.set_postfix({'loss': loss.item()})
            if (tb is not None or wb is not None) and global_step % 50 == 0:
                log_scalars({"train/loss_step": loss.item()}, global_step)
            
        avg_loss = total_loss / len(loader)
        val_mean, val_median = validate()
        print(f"Epoch {epoch+1}: train_avg={avg_loss:.4f}  "
              f"val_mean={val_mean:.4f}  val_median={val_median:.4f}")
        log_scalars({"train/avg": avg_loss, "val/mean": val_mean,
                     "val/median": val_median, "epoch": epoch + 1}, global_step)

        checkpoint = {
            "state_dict": model.state_dict(),
            "feature_name": target_feature,
            "feature_stats": feature_stats,
            "target_kind_default": target_kind_default,
            "noise_schedule": objective,  # forward-noising convention the head expects
            "loss_type": loss_type,       # mse | smooth_l1 | bce_logits | cosine
            "huber_beta": huber_beta,     # SmoothL1 beta (None unless loss_type==smooth_l1)
            "target_clamp": [clamp_min, clamp_max],
            **slider,                     # slider_min / slider_max (p1/p99) + slider_scale (linear|log)
            "val_mean": val_mean,
            "val_median": val_median,
        }
        ep_ckpt = os.path.join(save_dir, f"{stem}_ep{epoch+1}.pt")
        torch.save(checkpoint, ep_ckpt)

        # Keep a best-by-val-median copy (the outlier-robust convergence signal)
        if val_median == val_median and val_median < best_val_median:  # not NaN and improved
            best_val_median = val_median
            torch.save(checkpoint, os.path.join(save_dir, f"{stem}_best.pt"))
            print(f"  ↳ new best (val_median={val_median:.4f}) → {stem}_best.pt")
            log_scalars({"best/val_median": best_val_median}, global_step)

        run_test_inference(epoch, ep_ckpt, global_step)

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
        use_wandb=args.wandb or bool(ycfg.get("wandb", False)),
        wandb_project=pick(args.wandb_project, "wandb_project", "latch"),
        test_cfg=ycfg.get("test"),
    )
