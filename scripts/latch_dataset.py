"""
LatCH Dataset — reads (latent, target_feature) pairs for training.

Time-series features (rms_energy_*_ts, beat_activations_ts, hpcp_ts, …) are
stored in the MIR TimeseriesDB (SQLite, WAL mode) at
    /home/kim/Projects/mir/data/timeseries.db

Scalar features (bpm, lufs, …) are still in the companion .INFO / .json files
next to each .npy latent.

DB lookup key = npy_path.stem  (e.g. "Artist - Title_0")
"""

import json
import sys
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# MIR TimeseriesDB — optional but required for _ts features
# ---------------------------------------------------------------------------
sys.path.insert(0, "/home/kim/Projects/mir/src")

try:
    from core.timeseries_db import TimeseriesDB, DEFAULT_DB_PATH
    _db_path = DEFAULT_DB_PATH
    HAS_TSDB = True
except ImportError:
    HAS_TSDB = False
    _db_path = None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# All feature keys that live in TimeseriesDB (not in .INFO)
_TIMESERIES_KEYS = {
    "rms_energy_bass_ts", "rms_energy_body_ts", "rms_energy_mid_ts", "rms_energy_air_ts",
    "spectral_flatness_ts", "spectral_flux_ts", "spectral_skewness_ts", "spectral_kurtosis_ts",
    "beat_activations_ts", "downbeat_activations_ts", "onsets_activations_ts",
    "hpcp_ts", "tonic_ts", "tonic_strength_ts",
    "brightness_ts", "roughness_ts", "hardness_ts", "depth_ts", "reverb_ts",
}

def _is_ts_feature(feature: str) -> bool:
    """Return True for features stored in TimeseriesDB."""
    return feature in _TIMESERIES_KEYS or feature.endswith("_ts")


def _load_ts_from_db(db: "TimeseriesDB", crop_key: str, feature: str) -> Optional[np.ndarray]:
    """Fetch one time-series array for *crop_key* from the database.

    Returns None if the key or field is absent.
    """
    arrays = db.get(crop_key)
    if arrays is None:
        return None
    arr = arrays.get(feature)
    return arr  # may be None


def _prepare_target(arr: np.ndarray, max_frames: int) -> np.ndarray:
    """
    Normalise any target array to shape (C, max_frames):
     - 1-D [T]      → (1, max_frames)
     - 2-D [T, C]   → (C, max_frames)   (hpcp_ts is [256, 12])
     - 2-D [C, T]   → (C, max_frames)   (already channel-first)

    Pads or trims along the T dimension.
    """
    arr = np.asarray(arr, dtype=np.float32)

    if arr.ndim == 1:
        # (T,) → (1, T)
        arr = arr[np.newaxis, :]
    elif arr.ndim == 2:
        # Decide channel-first vs channel-last:
        # For hpcp_ts the natural storage is [T, 12].
        # We treat the *smaller* dimension as channels.
        if arr.shape[0] > arr.shape[1]:
            arr = arr.T  # (C, T)
    else:
        raise ValueError(f"Unexpected array ndim={arr.ndim}")

    C, T = arr.shape
    if T > max_frames:
        arr = arr[:, :max_frames]
    elif T < max_frames:
        pad = max_frames - T
        arr = np.pad(arr, ((0, 0), (0, pad)))

    return arr  # (C, max_frames)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LatCHDataset(Dataset):
    """
    Dataset of (latent, target) pairs for LatCH training.

    Args:
        latent_dir:     Root directory containing per-track sub-dirs of .npy files.
        info_dir:       Companion .INFO file root (same structure as latent_dir).
                        Only needed for scalar features.  May be None if all
                        requested features are time-series.
        target_feature: Feature name to use as the training target.
                        Time-series features (ending in _ts) are read from the
                        TimeseriesDB.  Scalars are read from .INFO files.
        max_frames:     Latent / target sequence length (default 256).
        db_path:        Override path to timeseries.db.
        stem_suffixes:  Filename suffixes to *skip* (stems).
    """

    STEM_SUFFIXES = {"_bass", "_drums", "_other", "_vocals"}

    def __init__(
        self,
        latent_dir: str,
        info_dir: Optional[str] = None,
        target_feature: str = "rms_energy_bass",
        max_frames: int = 256,
        db_path: Optional[str] = None,
        stem_suffixes=None,
        clamp_min: Optional[float] = None,
        clamp_max: Optional[float] = None,
    ):
        self.latent_dir      = Path(latent_dir)
        self.info_dir        = Path(info_dir) if info_dir else None
        self.max_frames      = max_frames
        # Optional target clamp (e.g. dB-floor for rms_energy_* to tame near-silent
        # outliers). Applied to every returned target; keep train & inference consistent.
        self.clamp_min       = clamp_min
        self.clamp_max       = clamp_max

        # Canonical name is bare (no _ts).  We always append _ts for DB lookups.
        self.bare_feature    = target_feature.removesuffix("_ts")
        self.ts_feature      = self.bare_feature + "_ts"
        # is_ts_feature is True for all known DB-backed features
        self.is_ts_feature   = _is_ts_feature(self.ts_feature)

        # Open TimeseriesDB (one connection per Dataset instance)
        self._db: Optional["TimeseriesDB"] = None
        if self.is_ts_feature:
            if not HAS_TSDB:
                raise ImportError(
                    "TimeseriesDB not importable — add /home/kim/Projects/mir/src "
                    "to PYTHONPATH or fix the MIR import."
                )
            resolved_db = Path(db_path) if db_path else _db_path
            self._db = TimeseriesDB.open(resolved_db)
            print(f"TimeseriesDB opened: {resolved_db}  ({self._db.count():,} entries)")
            print(f"Target feature      : '{self.bare_feature}' → DB key '{self.ts_feature}'")

        if stem_suffixes is not None:
            self.STEM_SUFFIXES = set(stem_suffixes)

        # Discover all valid .npy files
        self.items: list[tuple[Path, str]] = []  # (npy_path, crop_key)
        print(f"Scanning {self.latent_dir} for latents …")
        if self.latent_dir.exists():
            for track_dir in sorted(self.latent_dir.iterdir()):
                if not track_dir.is_dir():
                    continue
                for npy_path in sorted(track_dir.glob("*.npy")):
                    # Skip stem files
                    if any(npy_path.stem.endswith(s) for s in self.STEM_SUFFIXES):
                        continue
                    crop_key = npy_path.stem  # e.g. "Artist - Title_0"
                    self.items.append((npy_path, crop_key))

        print(f"Found {len(self.items)} latent files.")

        # Validate a sample to fail fast with a helpful message
        if self.items and self.is_ts_feature and self._db is not None:
            _, key = self.items[0]
            sample = _load_ts_from_db(self._db, key, self.ts_feature)
            if sample is None:
                # Check if the bare scalar feature exists in the companion JSON
                try:
                    info = self._load_info(self.items[0][0], key)
                    if self.bare_feature in info:
                        print(
                            f"WARNING: '{self.bare_feature}' exists as a scalar in .INFO "
                            f"but '{self.ts_feature}' is NOT in the TimeseriesDB for '{key}'.\n"
                            f"         Run extract_timeseries() (skip_timeseries=False) on "
                            f"your crops to generate the time-series version."
                        )
                    else:
                        print(
                            f"WARNING: '{self.ts_feature}' not found in DB for '{key}'.  "
                            f"Make sure extract_timeseries() has been run."
                        )
                except Exception:
                    print(
                        f"WARNING: '{self.ts_feature}' missing in TimeseriesDB for '{key}'."
                    )

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        start_idx = idx
        while True:
            npy_path, crop_key = self.items[idx]
            try:
                # ---- Latent ----
                latent = np.load(str(npy_path)).astype(np.float32)  # [64, T]
                C_lat, T_lat = latent.shape
                if T_lat > self.max_frames:
                    latent = latent[:, :self.max_frames]
                elif T_lat < self.max_frames:
                    latent = np.pad(latent, ((0, 0), (0, self.max_frames - T_lat)))

                # ---- Target ----
                if self.is_ts_feature:
                    # --- read from TimeseriesDB ---
                    raw = _load_ts_from_db(self._db, crop_key, self.ts_feature)
                    if raw is None:
                        raise ValueError(
                            f"'{self.ts_feature}' missing in TimeseriesDB for key '{crop_key}'"
                        )
                    target = _prepare_target(raw, self.max_frames)  # (C, max_frames)

                else:
                    # --- scalar: broadcast to [1, max_frames] ---
                    info = self._load_info(npy_path, crop_key)
                    feat_val = info.get(self.bare_feature)
                    if feat_val is None:
                        raise ValueError(
                            f"'{self.bare_feature}' missing in .INFO for '{crop_key}'"
                        )
                    val = float(feat_val)
                    target = np.full((1, self.max_frames), val, dtype=np.float32)

                if self.clamp_min is not None or self.clamp_max is not None:
                    target = np.clip(target, self.clamp_min, self.clamp_max)

                latent_t = torch.from_numpy(latent)          # [64, T]
                target_t = torch.from_numpy(np.ascontiguousarray(target))  # [C, T]
                return latent_t, target_t

            except Exception:
                idx = (idx + 1) % len(self.items)
                if idx == start_idx:
                    raise RuntimeError("No valid items found in the entire dataset.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_info(self, npy_path: Path, crop_key: str) -> dict:
        """Load companion .INFO or .json for *npy_path*."""
        # 1. Try info_dir (separate tree)
        if self.info_dir is not None:
            info_path = self.info_dir / npy_path.parent.name / (crop_key + ".INFO")
            if info_path.exists():
                with open(info_path) as f:
                    data = json.load(f)
                return data.get("original_features", data)

        # 2. Try .json sibling (next to .npy)
        sibling = npy_path.with_suffix(".json")
        if sibling.exists():
            with open(sibling) as f:
                data = json.load(f)
            return data.get("original_features", data)

        raise FileNotFoundError(f"No .INFO / .json found for {npy_path}")

    def close(self) -> None:
        """Close the database connection. Call when done with the dataset."""
        if self._db is not None:
            self._db.close()
            self._db = None

    def __del__(self):
        self.close()
