#!/usr/bin/env python3
"""Screen a LatCH head for guidance usefulness BEFORE wiring its slider.

A head can only guide if its output actually responds to the latent. Low-variance /
weakly-encoded features (e.g. tonic, tonic_strength) train to ~constant predictors
whose gradient w.r.t. the latent is ~0 — so no amount of gain/weight moves the
output. This measures, on real latents:
  - pred_std_across_clips : how much the head's prediction varies between clips
  - ||dpred/dz|| / ||z||   : input sensitivity (the guidance gradient scale)

Rule of thumb (vs the working rms heads, which score ~1.5): a head with
sensitivity < ~0.1 and tiny pred_std is effectively dead-for-guidance.

Usage:
    python scripts/latch_head_sensitivity.py latch_weights/latch_X_v2_best.pt [more.pt ...]
    python scripts/latch_head_sensitivity.py            # screens all latch_weights/*_best.pt
"""

import glob
import os
import sys

import numpy as np
import torch

from stable_audio_tools.models.latch import load_latch_from_checkpoint

LATENT_DIR = "/run/media/kim/Lehto/latents"
STEMS = ("_bass", "_drums", "_other", "_vocals")


def _real_latents(n=8):
    out = []
    for td in sorted(glob.glob(LATENT_DIR + "/*")):
        if not os.path.isdir(td):
            continue
        for npy in sorted(glob.glob(td + "/*.npy")):
            if any(npy.endswith(s + ".npy") for s in STEMS):
                continue
            out.append(npy)
            if len(out) >= n:
                return out
    return out


def screen(paths, t_val=0.3, noise=0.3):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    z0 = torch.stack([torch.from_numpy(np.load(p).astype(np.float32)) for p in _real_latents()]).to(dev)
    t = torch.full((z0.shape[0],), t_val, device=dev)
    print(f"{'head':40} {'pred~':>8} {'pred_std':>9} {'sensitivity':>12}  verdict")
    for p in paths:
        head = load_latch_from_checkpoint(p, device=dev)
        zt = (z0 + noise * torch.randn_like(z0)).detach().requires_grad_(True)
        pred = head(zt.float(), t)
        pmean = pred.mean().item()
        pstd = pred.mean(dim=(1, 2)).std().item()
        g = torch.autograd.grad(pred.sum(), zt)[0]
        sens = g.norm().item() / (zt.norm().item() + 1e-9)
        verdict = "OK" if sens > 0.3 else ("WEAK" if sens > 0.1 else "DEAD (skip — won't guide)")
        print(f"{os.path.basename(p):40} {pmean:>8.3f} {pstd:>9.4f} {sens:>12.4e}  {verdict}")


if __name__ == "__main__":
    args = sys.argv[1:] or sorted(glob.glob("latch_weights/*_best.pt"))
    screen(args)
