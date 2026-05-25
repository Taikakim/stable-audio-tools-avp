#!/usr/bin/env python3
"""Probe the σ/α/s_t trajectory that sample_euler_latch_guided builds.

LatCH guidance strength at step i is scaled by s_t = α_i / Σα. This script
reconstructs the *exact* schedule the sampler computes (RF sigmoid path or the
v-objective cos/sin path, with optional DistributionShift) so we can read off,
empirically, which steps carry usable guidance power — instead of guessing.

CPU-only, no model weights required. The schedule depends only on:
  objective (rectified_flow | v), steps, sigma_max, seq_len, dist_shift.

Usage:
    python scripts/latch_trajectory_probe.py                       # default sweep
    python scripts/latch_trajectory_probe.py --objective v --steps 100 --seq-len 256
"""

import argparse
import math

import torch

from stable_audio_tools.inference.sampling import DistributionShift, get_alphas_sigmas


def build_schedule(objective, steps, sigma_max, seq_len, dist_shift, use_sine):
    """Return (sigma_arr, alpha_arr) per step, replicating sample_euler_latch_guided."""
    is_rf = objective in ("rectified_flow", "rf_denoiser")
    if is_rf:
        _sm = min(sigma_max, 1.0)
        lsnr_max = math.log(((1 - _sm) / _sm) + 1e-6) if _sm < 1 else -6
        t = torch.sigmoid(-torch.linspace(lsnr_max, 2, steps + 1))
        t[0], t[-1] = _sm, 0
        alphas_arr, sigmas_arr = 1.0 - t, t
    else:
        t = torch.linspace(sigma_max, 0, steps + 1)
        if dist_shift is not None:
            t = dist_shift.time_shift(t, seq_len)
        alphas_arr, sigmas_arr = get_alphas_sigmas(t)
    # sampler uses indices 0..steps-1 (zip(t[:-1], t[1:]))
    return sigmas_arr[:-1], alphas_arr[:-1]


def summarize(label, sigma, alpha):
    s_t = alpha / alpha.sum()
    cum = torch.cumsum(s_t, dim=0)
    n = len(sigma)

    print(f"\n=== {label} ===")
    print(f"{'i':>4} {'pct':>6} {'sigma':>9} {'alpha':>9} {'s_t':>11} {'cum_s_t':>9}")
    # print a sparse view: every step if <=20, else ~20 rows
    stride = max(1, n // 20)
    for i in range(n):
        if i % stride and i not in (0, n - 1):
            continue
        print(f"{i:>4} {i / n * 100:>5.0f}% {sigma[i]:>9.4f} {alpha[i]:>9.4f} "
              f"{s_t[i].item():>11.4e} {cum[i].item():>9.3f}")

    def first_idx(mask):
        nz = torch.nonzero(mask)
        return int(nz[0]) if len(nz) else None

    def pct(i):
        return f"{i / n * 100:.0f}%" if i is not None else "n/a"

    a10 = first_idx(alpha >= 0.10)
    a50 = first_idx(alpha >= 0.50)
    a90 = first_idx(alpha >= 0.90)
    # central guidance mass: where cumulative s_t crosses 5% / 50% / 95%
    c05 = first_idx(cum >= 0.05)
    c50 = first_idx(cum >= 0.50)
    c95 = first_idx(cum >= 0.95)

    print(f"  alpha>=0.10 at step {a10} ({pct(a10)});  "
          f">=0.50 at {a50} ({pct(a50)});  >=0.90 at {a90} ({pct(a90)})")
    print(f"  guidance mass (cum s_t): 5% by step {c05} ({pct(c05)}), "
          f"50% by {c50} ({pct(c50)}), 95% by {c95} ({pct(c95)})")
    print(f"  => 90% of all guidance power lives in steps "
          f"[{c05}..{c95}]  (pct [{pct(c05)}..{pct(c95)}])")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", default=None,
                    help="rectified_flow | rf_denoiser | v (default: sweep all)")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--sigma-max", type=float, default=1.0)
    ap.add_argument("--seq-len", type=int, default=256, help="latent frames (Small=256)")
    ap.add_argument("--dist-shift", action="store_true",
                    help="apply DistributionShift (v-objective only)")
    ap.add_argument("--use-sine", action="store_true")
    args = ap.parse_args()

    if args.objective:
        configs = [(args.objective, args.dist_shift)]
    else:
        # default sweep: Small (RF) and V1-style (v, with/without dist_shift)
        configs = [
            ("rectified_flow", False),
            ("v", False),
            ("v", True),
        ]

    for objective, ds_flag in configs:
        ds = None
        if ds_flag:
            ds = DistributionShift(min_length=args.seq_len, use_sine=args.use_sine)
        sigma, alpha = build_schedule(
            objective, args.steps, args.sigma_max, args.seq_len, ds, args.use_sine
        )
        label = (f"objective={objective}  steps={args.steps}  "
                 f"sigma_max={args.sigma_max}  seq_len={args.seq_len}  "
                 f"dist_shift={'on' if ds_flag else 'off'}")
        summarize(label, sigma, alpha)


if __name__ == "__main__":
    main()
