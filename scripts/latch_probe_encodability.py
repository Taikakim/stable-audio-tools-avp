#!/usr/bin/env python3
"""Controllability probe: how linearly does the VAE latent encode each scalar feature?

For every musical scalar field in the crops' companion .json, fit a cheap ridge
probe from the (time-pooled mean+std of the) latent to the feature value and report
held-out R^2. High R^2 => the latent strongly encodes it => good LatCH/conditioner
candidate. Near-zero R^2 => weakly encoded => a control head will likely be
dead-for-guidance (as the sensitivity screener confirmed for flatness/tonic). Cheap,
CPU-only, no training — run before investing GPU on a head.

Usage: python scripts/latch_probe_encodability.py [N_crops]
"""

import glob
import json
import os
import sys

import numpy as np

LATENT_DIR = "/run/media/kim/Lehto/latents"
STEMS = ("_bass", "_drums", "_other", "_vocals")
# non-musical bookkeeping fields to skip
SKIP = {"seconds_total", "position", "start_time", "end_time", "start_sample",
        "end_sample", "duration", "samples", "has_stems", "track_metadata_year"}


def collect(n):
    X, feats = [], {}
    paths = []
    for td in sorted(glob.glob(LATENT_DIR + "/*")):
        if not os.path.isdir(td):
            continue
        for npy in sorted(glob.glob(td + "/*.npy")):
            if any(npy.endswith(s + ".npy") for s in STEMS):
                continue
            paths.append(npy)
            if len(paths) >= n:
                break
        if len(paths) >= n:
            break
    for npy in paths:
        jp = npy[:-4] + ".json"
        if not os.path.exists(jp):
            continue
        z = np.load(npy).astype(np.float32)              # [64, T]
        x = np.concatenate([z.mean(1), z.std(1)])        # [128] time-pooled mean+std
        d = json.load(open(jp))
        d = d.get("original_features", d)
        X.append(x)
        for k, v in d.items():
            if k in SKIP or not isinstance(v, (int, float)) or isinstance(v, bool):
                continue
            feats.setdefault(k, []).append((len(X) - 1, float(v)))
    return np.asarray(X), feats


def ridge_r2(X, y, lam=10.0):
    n = len(y)
    idx = np.random.RandomState(0).permutation(n)
    tr, va = idx[: int(0.8 * n)], idx[int(0.8 * n):]
    Xt, yt = X[tr], y[tr]
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-8
    Xt = (Xt - mu) / sd
    Xv = (X[va] - mu) / sd
    ym = yt.mean()
    w = np.linalg.solve(Xt.T @ Xt + lam * np.eye(Xt.shape[1]), Xt.T @ (yt - ym))
    pred = Xv @ w + ym
    ss_res = ((y[va] - pred) ** 2).sum()
    ss_tot = ((y[va] - y[va].mean()) ** 2).sum() + 1e-12
    return 1.0 - ss_res / ss_tot


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    X, feats = collect(n)
    print(f"probed {len(X)} crops, {len(feats)} scalar features\n")
    rows = []
    for k, pairs in feats.items():
        if len(pairs) < 50:
            continue
        ix = np.array([i for i, _ in pairs]); yv = np.array([v for _, v in pairs])
        if yv.std() < 1e-6:
            rows.append((k, float("nan"), 0.0)); continue
        rows.append((k, ridge_r2(X[ix], yv), float(yv.std())))
    rows.sort(key=lambda r: (-(r[1] if r[1] == r[1] else -1)))
    print(f"{'feature':28} {'R^2':>7} {'std':>10}  encodability")
    for k, r2, sd in rows:
        tag = "STRONG" if r2 > 0.4 else ("moderate" if r2 > 0.15 else ("weak" if r2 > 0.05 else "≈none"))
        print(f"{k:28} {r2:7.3f} {sd:10.3f}  {tag}")


if __name__ == "__main__":
    main()
