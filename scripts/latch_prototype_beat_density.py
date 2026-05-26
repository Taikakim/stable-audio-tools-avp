#!/usr/bin/env python3
"""Prototype + encodability probe for a CONTINUOUS beat-density LatCH feature.

Hypothesis (from the beat_activations post-mortem, 2026-05-26): the binary beat
marker train is a poor guidance target because it's sparse — the loss is flat
between spikes, so the gradient has nothing to push. A *smoothed* version (each
marker → a Gaussian bump) is dense and continuous, so it should be both more
linearly decodable from the latent AND a stronger guidance signal.

This script tests the first half cheaply (CPU, no GPU training): on a subset of
crops it Gaussian-smooths the existing `beat_activations_ts` (and `onsets_…`) at a
few widths and runs a FRAME-WISE ridge probe — predict the per-frame envelope from
a local window of latent frames — reporting held-out R^2 (split BY CLIP, no frame
leakage). If smoothed R^2 >> raw R^2, the continuous feature is worth training.

Usage: python scripts/latch_prototype_beat_density.py [N_crops]
"""

import glob
import os
import sys

import numpy as np

sys.path.insert(0, "/home/kim/Projects/mir/src")
from core.timeseries_db import TimeseriesDB, DEFAULT_DB_PATH

LATENT_DIR = "/run/media/kim/Lehto/latents"
STEMS = ("_bass", "_drums", "_other", "_vocals")
T = 256
FEATURES = ["beat_activations_ts", "onsets_activations_ts"]
SIGMAS = [0.0, 1.5, 3.0, 6.0]   # frames; 0 = raw binary markers (baseline)
CONTEXT_K = [0, 4]              # ± latent frames given to the linear probe


def gauss_kernel(sigma):
    if sigma <= 0:
        return None
    r = int(np.ceil(3 * sigma))
    x = np.arange(-r, r + 1)
    k = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def smooth(env, kern):
    if kern is None:
        return env.astype(np.float32)
    return np.convolve(env, kern, mode="same").astype(np.float32)


def context_matrix(z, k):
    """z:[C,T] -> [T, C*(2k+1)] local-window features per frame."""
    if k == 0:
        return z.T.copy()
    padded = np.pad(z, ((0, 0), (k, k)))
    parts = [padded[:, i:i + z.shape[1]] for i in range(2 * k + 1)]
    return np.concatenate(parts, axis=0).T  # [T, C*(2k+1)]


def ridge_r2(Xtr, ytr, Xva, yva, lam=10.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr = (Xtr - mu) / sd
    Xva = (Xva - mu) / sd
    ym = ytr.mean()
    w = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]), Xtr.T @ (ytr - ym))
    pred = Xva @ w + ym
    ss_res = ((yva - pred) ** 2).sum()
    ss_tot = ((yva - yva.mean()) ** 2).sum() + 1e-12
    return 1.0 - ss_res / ss_tot


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 800
    db = TimeseriesDB.open(DEFAULT_DB_PATH)
    print(f"DB: {DEFAULT_DB_PATH} ({db.count():,} entries); target subset {n} crops\n")

    # gather (latent, {feature: raw_envelope}) for crops present in the DB
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

    clips = []  # (z[C,T], {feat: env[T]})
    miss = 0
    for npy in paths:
        key = os.path.splitext(os.path.basename(npy))[0]
        arrs = db.get(key)
        if arrs is None:
            miss += 1
            continue
        envs = {}
        for f in FEATURES:
            a = arrs.get(f)
            if a is None:
                continue
            a = np.asarray(a, dtype=np.float32).reshape(-1)[:T]
            if a.shape[0] < T:
                a = np.pad(a, (0, T - a.shape[0]))
            envs[f] = a
        if not envs:
            continue
        z = np.load(npy).astype(np.float32)[:, :T]
        if z.shape[1] < T:
            z = np.pad(z, ((0, 0), (0, T - z.shape[1])))
        clips.append((z, envs))
    print(f"usable clips: {len(clips)}  (missing-in-DB: {miss})")
    if not clips:
        print("No clips with these features in the DB — check FEATURES names.")
        return

    # report marker density for context
    for f in FEATURES:
        vals = [c[1][f] for c in clips if f in c[1]]
        if vals:
            stacked = np.stack(vals)
            print(f"  {f:24} clips={len(vals)}  mean_activation={stacked.mean():.4f} "
                  f"frac>0.5={np.mean(stacked > 0.5):.4f}")
    print()

    # clip-level split (no frame leakage)
    rs = np.random.RandomState(0)
    idx = rs.permutation(len(clips))
    cut = int(0.8 * len(clips))
    tr_clips = [clips[i] for i in idx[:cut]]
    va_clips = [clips[i] for i in idx[cut:]]

    print(f"{'feature':22} {'sigma':>6} {'ctx±k':>6} {'R^2':>8}  encodability")
    for f in FEATURES:
        present_tr = [c for c in tr_clips if f in c[1]]
        present_va = [c for c in va_clips if f in c[1]]
        if not present_tr or not present_va:
            continue
        for sigma in SIGMAS:
            kern = gauss_kernel(sigma)
            for k in CONTEXT_K:
                Xtr = np.concatenate([context_matrix(z, k) for z, _ in present_tr])
                ytr = np.concatenate([smooth(e[f], kern) for _, e in present_tr])
                Xva = np.concatenate([context_matrix(z, k) for z, _ in present_va])
                yva = np.concatenate([smooth(e[f], kern) for _, e in present_va])
                r2 = ridge_r2(Xtr, ytr, Xva, yva)
                tag = ("STRONG" if r2 > 0.4 else "moderate" if r2 > 0.15
                       else "weak" if r2 > 0.05 else "≈none")
                lbl = f"{f} (raw)" if sigma == 0 else f
                print(f"{lbl:22} {sigma:6.1f} {k:6d} {r2:8.3f}  {tag}")
        print()
    db.close()


if __name__ == "__main__":
    main()
