#!/usr/bin/env python3
"""Bracketed LatCH sweep for the Small (rectified_flow) model.

Generates one clip per (window, gain) combination with a FIXED seed/prompt/target,
plus a LatCH-off baseline. For every clip it reports:
  - peak / rms          (blow-up / disintegration detector)
  - dL2_vs_baseline     (L2 of (audio-baseline)/||baseline||; 0 = no audible effect)

Same seed => identical initial noise => differences are purely the guidance.

Run:
    source rocm_env.sh   # optional; sets TUNING=0 read-only
    sat-venv/bin/python scripts/latch_sweep.py
"""

import argparse
import json
import os

import numpy as np
import soundfile as sf
import torch

from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict
from stable_audio_tools.inference.generation import generate_diffusion_cond

MODEL_CONFIG = "models/checkpoints/small/base_model_config.json"
CKPT = "models/checkpoints/small/base_model.ckpt"
LATCH = "latch_weights/latch_tonic_ep10.pt"

# (start_pct, end_pct): where guidance is active, as step-index fractions.
WINDOWS = [(0.0, 0.2), (0.5, 1.0), (0.7, 1.0), (0.8, 1.0)]
# rho == mu == gain.
GAINS = [0.03, 0.3, 1.0, 3.0, 10.0]


def gen(model, sample_size, seed, prompt, seconds, steps, cfg_scale, device,
        latch_configs=None, latch_hparams=None):
    audio = generate_diffusion_cond(
        model,
        steps=steps,
        cfg_scale=cfg_scale,
        conditioning=[{"prompt": prompt, "seconds_total": seconds}],
        sample_size=sample_size,
        seed=seed,
        device=device,
        sigma_max=1.0,
        latch_configs=latch_configs,
        latch_hparams=latch_hparams,
    )
    return audio.squeeze(0).cpu().float().numpy()  # [C, N]


def stats(x, baseline):
    peak = float(np.abs(x).max())
    rms = float(np.sqrt((x ** 2).mean()))
    if baseline is None:
        dl2 = 0.0
    else:
        dl2 = float(np.linalg.norm(x - baseline) / (np.linalg.norm(baseline) + 1e-12))
    return peak, rms, dl2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="latch_sweep_out")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=7.0)
    ap.add_argument("--prompt", default="goa trance, hypnotic, driving")
    ap.add_argument("--value", type=float, default=10.78, help="constant target value")
    ap.add_argument("--quick", action="store_true",
                    help="tiny bracket: baseline + 2 combos, to validate end-to-end")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(MODEL_CONFIG) as f:
        model_config = json.load(f)
    model = create_model_from_config(model_config)
    model.load_state_dict(load_ckpt_state_dict(CKPT))
    model = model.half().to(device)

    sr = model_config["sample_rate"]
    sample_size = model_config["sample_size"]
    seconds = round(sample_size / sr)
    print(f"Model loaded. sr={sr} sample_size={sample_size} (~{seconds}s)  device={device}")

    windows, gains = WINDOWS, GAINS
    if args.quick:
        windows, gains = [(0.7, 1.0)], [0.03, 10.0]

    # baseline (LatCH off)
    print("\n[baseline] LatCH off")
    base = gen(model, sample_size, args.seed, args.prompt, seconds,
               args.steps, args.cfg, device)
    sf.write(os.path.join(args.out, "baseline.wav"), base.T, sr)

    rows = []
    base_peak, base_rms, _ = stats(base, None)
    rows.append(("baseline", "-", base_peak, base_rms, 0.0))

    for (s0, s1) in windows:
        for g in gains:
            tag = f"w{s0:.2f}-{s1:.2f}_g{g:g}"
            print(f"\n[{tag}] window=[{s0},{s1}] rho=mu={g}")
            try:
                audio = gen(
                    model, sample_size, args.seed, args.prompt, seconds,
                    args.steps, args.cfg, device,
                    latch_configs=[{
                        "model_path": LATCH, "kind": "constant", "value": args.value,
                        "weight": 1.0, "start_pct": s0, "end_pct": s1,
                    }],
                    latch_hparams={"rho": g, "mu": g, "gamma": 0.3,
                                   "n_iter": 4, "log_norms": False},
                )
                sf.write(os.path.join(args.out, f"{tag}.wav"), audio.T, sr)
                p, r, d = stats(audio, base)
                rows.append((f"[{s0},{s1}]", f"{g:g}", p, r, d))
            except Exception as e:
                print(f"  FAILED: {e}")
                rows.append((f"[{s0},{s1}]", f"{g:g}", float("nan"),
                             float("nan"), float("nan")))

    print("\n\n===== SWEEP SUMMARY =====")
    print(f"(baseline: peak={base_peak:.3f} rms={base_rms:.3f}; "
          f"dL2 vs that baseline. disintegration judged vs baseline rms/peak)")
    print(f"{'window':>14} {'gain':>6} {'peak':>9} {'rms':>9} {'dL2_vs_base':>12}  note")
    for w, g, p, r, d in rows:
        note = ""
        if w == "baseline":
            note = "reference"
        elif r > 1.4 * base_rms or p > 1.4 * base_peak:
            note = "possible disintegration (energy blow-up)"
        elif d < 0.02:
            note = "barely audible"
        elif d < 0.08:
            note = "subtle"
        elif d < 0.20:
            note = "clear effect"
        else:
            note = "strong / madness?"
        print(f"{w:>14} {g:>6} {p:>9.4f} {r:>9.4f} {d:>12.4e}  {note}")
    print(f"\nWAVs written to {args.out}/")


if __name__ == "__main__":
    main()
