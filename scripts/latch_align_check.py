#!/usr/bin/env python3
"""Verify LatCH target↔latent temporal alignment.

Loads a few real latents and their raw MIR feature arrays straight from the
TimeseriesDB (BEFORE _prepare_target trims/pads), and reports frame counts.
If a feature's raw length != the latent frame count, _prepare_target's
trim/pad silently time-warps/truncates the target for time-series features.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
sys.path.insert(0, "/home/kim/Projects/mir/src")

from core.timeseries_db import TimeseriesDB, DEFAULT_DB_PATH  # noqa: E402

LATENT_DIR = Path("/run/media/kim/Lehto/latents")
FEATURES = [
    "tonic_ts", "rms_energy_bass_ts", "beat_activations_ts",
    "spectral_flatness_ts", "hpcp_ts",
]


def find_latents(n=5):
    out = []
    stems = {"_bass", "_drums", "_other", "_vocals"}
    for track_dir in sorted(LATENT_DIR.iterdir()):
        if not track_dir.is_dir():
            continue
        for npy in sorted(track_dir.glob("*.npy")):
            if any(npy.stem.endswith(s) for s in stems):
                continue
            out.append(npy)
            if len(out) >= n:
                return out
    return out


def main():
    db = TimeseriesDB.open(DEFAULT_DB_PATH)
    print(f"DB: {DEFAULT_DB_PATH}  ({db.count():,} entries)")

    latents = find_latents(5)
    if not latents:
        print("No latents found.")
        return

    for npy in latents:
        key = npy.stem
        lat = np.load(str(npy))
        T_lat = lat.shape[-1]
        print(f"\n--- {key} ---")
        print(f"latent shape: {tuple(lat.shape)}  -> T_latent={T_lat}")
        arrays = db.get(key)
        if arrays is None:
            print("  NOT in DB")
            continue
        for feat in FEATURES:
            arr = arrays.get(feat)
            if arr is None:
                print(f"  {feat:<24} : (absent)")
                continue
            arr = np.asarray(arr)
            T_feat = arr.shape[0] if arr.ndim == 1 else max(arr.shape)
            ratio = T_feat / T_lat if T_lat else float("nan")
            flag = "  OK (matches latent)" if T_feat == T_lat else \
                   f"  MISALIGNED  (raw is {ratio:.2f}x latent frames)"
            print(f"  {feat:<24} : shape={tuple(arr.shape)}  T_feat={T_feat}{flag}")

    db.close()


if __name__ == "__main__":
    main()
