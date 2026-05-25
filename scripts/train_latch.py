import os

# Apply the ROCm "training" profile from rocm_env.yaml BEFORE importing torch.
# Loaded standalone via importlib so the stable_audio_tools package __init__
# (which imports torch) does NOT run first — values live only in rocm_env.yaml.
import importlib.util as _ilu
from pathlib import Path as _Path
_re_path = _Path(__file__).resolve().parent.parent / "stable_audio_tools" / "rocm_env.py"
_spec = _ilu.spec_from_file_location("_sat_rocm_env", _re_path)
_rocm_env = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_rocm_env)
_rocm_env.apply_profile("training")

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
        "n_samples": len(samples),
    }


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
):
    os.makedirs(save_dir, exist_ok=True)
    print(f"Forward-noising schedule: objective={objective} "
          f"(MUST match the diffusion model this head will guide)")

    # Init Dataset
    dataset = LatCHDataset(
        latent_dir, info_dir,
        target_feature=target_feature,
        db_path=db_path,
    )
    if len(dataset) == 0:
        print("No valid latent-INFO pairs found. Make sure external drives are mounted.")
        return

    # Peek at one sample to determine out_channels (1 for scalars/1-D ts, 12 for hpcp_ts, etc.)
    sample_latent, sample_target = dataset[0]
    out_channels = sample_target.shape[0]  # (C, T) → C
    print(f"Target '{target_feature}': out_channels={out_channels}, seq_len={sample_target.shape[1]}")

    # Compute target distribution stats for inference UI
    print("Sampling target stats from dataset (up to 200 items)...")
    feature_stats = _sample_target_stats(dataset, n=200)
    print(f"Target stats: {feature_stats}")
    target_kind_default = _default_kind_for(dataset.bare_feature)
    print(f"Default target kind for inference: {target_kind_default}")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                        num_workers=4, persistent_workers=True)

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
        criterion = nn.MSELoss()
        loss_type = "mse"
        
    print(f"Starting training on feature: {target_feature} with {len(dataset)} items.")
    
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
            pbar.set_postfix({'loss': loss.item()})
            
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
        
        checkpoint = {
            "state_dict": model.state_dict(),
            "feature_name": target_feature,
            "feature_stats": feature_stats,
            "target_kind_default": target_kind_default,
            "noise_schedule": objective,  # forward-noising convention the head expects
            "loss_type": loss_type,       # mse | bce_logits | cosine
        }
        torch.save(checkpoint, os.path.join(save_dir, f"latch_{target_feature}_ep{epoch+1}.pt"))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Train a LatCH head on a time-series MIR feature."
    )
    parser.add_argument(
        "--feature", type=str, default="rms_energy_bass",
        help=("Target feature name (bare, without _ts suffix). "
              "Examples: spectral_flatness, rms_energy_bass, beat_activations, hpcp, tonic")
    )
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--batch-size", type=int,   default=8)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument(
        "--latent-dir", type=str, default="/run/media/kim/Lehto/latents",
        help="Root directory of per-track .npy latent files"
    )
    parser.add_argument(
        "--db-path", type=str, default=None,
        help="Override path to timeseries.db (default: /home/kim/Projects/mir/data/timeseries.db)"
    )
    parser.add_argument("--save-dir", type=str, default="latch_weights")
    parser.add_argument(
        "--objective", type=str, default="rectified_flow",
        choices=["rectified_flow", "rf_denoiser", "v"],
        help=("Forward-noising schedule; MUST match the diffusion model this head "
              "will guide. Small base model = rectified_flow.")
    )
    args = parser.parse_args()

    train(
        latent_dir=args.latent_dir,
        target_feature=args.feature,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        db_path=args.db_path,
        save_dir=args.save_dir,
        objective=args.objective,
    )
