"""
WholeTrackTargetSource — LatCH targets sliced from whole-track .TIMESERIES.npz.

The MIR producer (mir/src/spectral/whole_track_timeseries.py) writes one
<track>.TIMESERIES.npz per track at a canonical frame rate (default 100 Hz),
covering the ENTIRE track, with full-mix + per-stem fields:

    beat_activation_ts, downbeat_activation_ts    (raw madmom soft probs)
    onset_envelope_ts, onset_envelope_<stem>_ts   (raw librosa onset strength)
    rms_<stem>_ts, rms_energy_<band>_ts, spectral_*_ts, hpcp_ts (n,12)

A LatCH crop covers the track interval [start_time, end_time] (read from the
crop's companion .json) and is encoded to a fixed-length latent. To build the
training target we slice the whole-track array at that interval and resample to
the latent's frame count. This matches the legacy per-crop TimeseriesDB
convention (content spans all latent frames) and generalises to variable crop
lengths (Stable Audio 3) since we always resample to `n_frames`.

Latent frame rates this serves (for reference; the loader is rate-agnostic —
it just resamples to whatever n_frames the latent has):
    SA Open Small / SA1 : 44100 / 2048 = 21.533 Hz   (256 frames / 11.888 s)
    SA3 medium          : 44100 / 4096 = 10.767 Hz

Track name is recovered from the crop key by stripping the trailing _<n>:
    "Artist - Title_12" -> "Artist - Title"
"""

import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

_KEY_SUFFIX = re.compile(r"_\d+$")


def track_from_key(crop_key: str) -> str:
    """'Artist - Title_12' -> 'Artist - Title' (inverse of f'{track}_{n}')."""
    return _KEY_SUFFIX.sub("", crop_key)


def _read_npz(path: Path) -> Tuple[Dict[str, np.ndarray], dict]:
    with np.load(str(path), allow_pickle=False) as z:
        meta = json.loads(str(z["__meta__"]))
        data = {k: z[k] for k in z.files if k != "__meta__"}
    return data, meta


def resample_axis0(arr: np.ndarray, n: int) -> np.ndarray:
    """Resample *arr* along axis 0 to length *n*.

    Downsampling uses fractional-bin mean pooling (anti-aliasing, and the right
    semantics for density/energy envelopes); upsampling uses linear interp.
    Handles 1-D (T,) and 2-D (T, C).
    """
    arr = np.asarray(arr, dtype=np.float32)
    T = arr.shape[0]
    if T == n:
        return arr
    if T == 0:
        return np.zeros((n,) + arr.shape[1:], dtype=np.float32)
    if T > n:
        edges = np.linspace(0, T, n + 1)
        out = np.empty((n,) + arr.shape[1:], dtype=np.float32)
        for i in range(n):
            s = int(np.floor(edges[i]))
            e = max(s + 1, int(np.ceil(edges[i + 1])))
            out[i] = arr[s:min(e, T)].mean(axis=0)
        return out
    # upsample
    xp = np.linspace(0.0, 1.0, T)
    x = np.linspace(0.0, 1.0, n)
    if arr.ndim == 1:
        return np.interp(x, xp, arr).astype(np.float32)
    return np.stack([np.interp(x, xp, arr[:, c]) for c in range(arr.shape[1])],
                    axis=1).astype(np.float32)


class WholeTrackTargetSource:
    """Serves channel-first (C, n_frames) targets from whole-track npz sidecars.

    Args:
        npz_root:   Dir holding <track>.TIMESERIES.npz (flat, portable — the MIR
                    output dir, e.g. /run/media/kim/Lehto/timeseries). A nested
                    <track>/<track>.TIMESERIES.npz layout is also accepted.
        cache:      Number of recently-used tracks to keep decoded in memory.
                    Consecutive crops share a track, so a small LRU avoids
                    re-reading/decoding the same npz hundreds of times.
    """

    def __init__(self, npz_root: str, cache: int = 8):
        self.npz_root = Path(npz_root)
        self._cache: "OrderedDict[str, Optional[Tuple[Dict[str, np.ndarray], dict]]]" = OrderedDict()
        self._cache_max = max(1, cache)

    def npz_path(self, track: str) -> Path:
        # Flat layout (portable, self-contained dir): npz_root/<track>.TIMESERIES.npz.
        # Falls back to nested (npz_root/<track>/<track>.TIMESERIES.npz) if pointed at
        # a source/latent tree instead of the dedicated timeseries dir.
        flat = self.npz_root / f"{track}.TIMESERIES.npz"
        if flat.exists():
            return flat
        return self.npz_root / track / f"{track}.TIMESERIES.npz"

    def _load(self, track: str):
        if track in self._cache:
            self._cache.move_to_end(track)
            return self._cache[track]
        p = self.npz_path(track)
        loaded = _read_npz(p) if p.exists() else None
        self._cache[track] = loaded
        if len(self._cache) > self._cache_max:
            self._cache.popitem(last=False)
        return loaded

    def has(self, track: str) -> bool:
        return self._load(track) is not None

    def fields(self, track: str):
        loaded = self._load(track)
        return sorted(loaded[0].keys()) if loaded else []

    def get(self, crop_key: str, feature: str, start_time: float, end_time: float,
            n_frames: int) -> Optional[np.ndarray]:
        """Return (C, n_frames) float32 target for *crop_key*, or None if absent.

        feature may be given with or without the trailing '_ts'.
        """
        track = track_from_key(crop_key)
        loaded = self._load(track)
        if loaded is None:
            return None
        data, meta = loaded
        field = feature if feature.endswith("_ts") else feature + "_ts"
        arr = data.get(field)
        if arr is None:
            return None

        fr = float(meta.get("frame_rate", 100))
        total = arr.shape[0]
        s = max(0, int(round(start_time * fr)))
        e = min(total, int(round(end_time * fr)))
        if e <= s:
            return None

        win = resample_axis0(arr[s:e], n_frames)        # (n,) or (n, C)
        if win.ndim == 1:
            win = win[None, :]                            # (1, n)
        else:
            win = win.T                                   # (C, n)
        return np.ascontiguousarray(win, dtype=np.float32)


# ---------------------------------------------------------------------------
# Self-test / sanity check
# ---------------------------------------------------------------------------

def _selftest():
    import argparse
    parser = argparse.ArgumentParser(
        description="Sanity-check WholeTrackTargetSource against real latents + npz.")
    parser.add_argument("--npz-root", default="/run/media/kim/Lehto/timeseries")
    parser.add_argument("--latent-root", default="/run/media/kim/Lehto/latents")
    parser.add_argument("--n-frames", type=int, default=256)
    parser.add_argument("--features", nargs="+",
                        default=["beat_activation", "downbeat_activation",
                                 "onset_envelope", "onset_envelope_drums",
                                 "rms_drums", "hpcp"])
    args = parser.parse_args()

    src = WholeTrackTargetSource(args.npz_root)
    latent_root = Path(args.latent_root)

    # Find a track that has BOTH latents and an existing npz.
    chosen = None
    for td in sorted(latent_root.iterdir()):
        if not td.is_dir():
            continue
        if src.has(td.name):
            npys = sorted(td.glob("*.npy"))
            npys = [p for p in npys if not any(p.stem.endswith(s)
                    for s in ("_bass", "_drums", "_other", "_vocals"))]
            if npys:
                chosen = (td, npys)
                break
    if chosen is None:
        print("No track found with both a latent dir and an existing .npz "
              "(batch may still be running). Try again later.")
        return

    track_dir, npys = chosen
    npy = npys[len(npys) // 2]            # a mid-track crop
    crop_key = npy.stem
    info = json.loads(npy.with_suffix(".json").read_text())
    info = info.get("original_features", info)
    st, et = float(info["start_time"]), float(info["end_time"])
    lat = np.load(str(npy))

    print(f"Track     : {track_dir.name}")
    print(f"Crop      : {crop_key}  [{st:.3f}, {et:.3f}] s  ({et-st:.3f}s)")
    print(f"Latent    : {lat.shape}  (resampling targets to n_frames={args.n_frames})")
    print(f"npz fields: {len(src.fields(track_dir.name))}")
    print("-" * 60)
    for feat in args.features:
        t = src.get(crop_key, feat, st, et, args.n_frames)
        if t is None:
            print(f"  {feat:24s} -> MISSING")
        else:
            print(f"  {feat:24s} -> shape={str(t.shape):10s} "
                  f"min={t.min():.3f} max={t.max():.3f} mean={t.mean():.3f}")


if __name__ == "__main__":
    _selftest()
