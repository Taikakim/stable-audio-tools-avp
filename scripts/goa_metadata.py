"""Custom metadata for the Stable Audio Open Small Goa-Trance finetune.

Pre-encoded latents at /run/media/kim/Lehto/latents/<track>/<track>_<idx>.npy
each ship with a companion .json sidecar that already contains BPM,
syncopation, on_beat_ratio, rhythmic_complexity, rhythmic_evenness, and a
pile of other features. This module just plucks them out + assembles the
prompt + enforces the deterministic 25% / 5% track-level split.

Stable Audio's dataset machinery looks for a top-level callable named
`get_custom_metadata(info, audio)` here. Returning `__reject__: True` makes
the dataset skip the sample (used for tracks outside the 25% subset, or for
the val tracks when training).

To switch between train and val datasets without editing this file, set:
    GOA_SPLIT_MODE = "train"  # or "val" — read from env var

at the .ini's `defaults` or just edit the constant below for one-off runs.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import warnings
from pathlib import Path
from typing import Any, Optional

# -----------------------------------------------------------------------------
# Knobs — change these for a different split, then re-run from scratch.
# -----------------------------------------------------------------------------

GOA_SUBSET_FRAC      = 0.25      # fraction of all TRACKS to include in the dataset
GOA_VAL_OF_SUBSET    = 0.05      # of the 25%, what fraction goes to val (track-level)
GOA_SPLIT_SEED       = 12345     # deterministic across runs
GOA_SPLIT_MODE       = os.environ.get("GOA_SPLIT_MODE", "train").lower()  # "train" | "val"

# -----------------------------------------------------------------------------
# NumberConditioner ranges (must match model_config.json's number conditioner
# min_val/max_val so the conditioner can normalise to [-1, 1] cleanly).
# -----------------------------------------------------------------------------

BPM_MIN, BPM_MAX = 60.0, 200.0   # Goa is 130–160ish; bracket loosely
RHYTHM_MIN, RHYTHM_MAX = 0.0, 1.0

# Granite / music-flamingo captions live alongside the latents on Lehto.
# Override with GOA_GRANITE_DIR=/some/other/path if mounted elsewhere.
GRANITE_DIR = Path(os.environ.get("GOA_GRANITE_DIR", "/run/media/kim/Lehto"))
GRANITE_DATA_JS = GRANITE_DIR / "feature_explorer_data.js"
GRANITE_CAPTIONS_JS = GRANITE_DIR / "feature_explorer_captions.js"
GOA_USE_GRANITE = os.environ.get("GOA_USE_GRANITE", "1") != "0"

# Caption prompt length budget. T5 max_length=64 tokens ≈ ~256 chars, so leave
# headroom for the BPM/key suffix we tack on.
GRANITE_PROMPT_BUDGET = 220

# Substrings that betray a generation failure or the schema-prompt itself
# leaking into the caption. Captions matching these get rejected.
_GRANITE_REJECT_NEEDLES = (
    "Debuginfod has been disabled",
    "Insufficient metadata",
    "A general description of the track",
    "The mood and emotional character",
    "accel_Q8_0",  # the model-name string sometimes leaks into the technical field
    "I cannot",
    "I'm sorry",
)


def _granite_caption_ok(s: Optional[str]) -> bool:
    if not s or len(s) < 40:
        return False
    for needle in _GRANITE_REJECT_NEEDLES:
        if needle in s:
            return False
    return True


_granite_cache: Optional[dict[str, str]] = None


def _load_granite_captions() -> dict[str, str]:
    """Lazy-load and cache the {track_name: caption} map from the
    feature_explorer_*.js files. Returns an empty dict if the files are not
    reachable (e.g. Lehto unmounted) so callers can fall back gracefully."""
    global _granite_cache
    if _granite_cache is not None:
        return _granite_cache
    if not GOA_USE_GRANITE or not GRANITE_DATA_JS.exists() or not GRANITE_CAPTIONS_JS.exists():
        _granite_cache = {}
        return _granite_cache

    try:
        with open(GRANITE_DATA_JS) as f:
            dsrc = f.read()
        mt = re.search(r"const\s+TRACKS\s*=\s*(\[.*?\]);", dsrc, re.DOTALL)
        if not mt:
            warnings.warn("goa_metadata: TRACKS array not found in feature_explorer_data.js")
            _granite_cache = {}
            return _granite_cache
        tracks: list[str] = json.loads(mt.group(1))

        with open(GRANITE_CAPTIONS_JS) as f:
            csrc = f.read()
        # The file is `// comment\nconst CAPTIONS = { ... };` — slice the JSON
        # object out by locating the first `{` and the last `}`, then let the
        # JSON parser handle it. Regex-extracting individual arrays is unsafe
        # because captions contain quotes, newlines, and `]`.
        lo = csrc.find("{")
        hi = csrc.rfind("}")
        cap_obj = json.loads(csrc[lo:hi + 1])

        short_genre: list[Optional[str]] = cap_obj.get("music_flamingo_short_genre", [])
        short_mood: list[Optional[str]] = cap_obj.get("music_flamingo_short_mood", [])
        full_caption: list[Optional[str]] = cap_obj.get("music_flamingo_full", [])

        # Captions array (4494) is parallel-indexed to the FIRST 4494 entries
        # of TRACKS (4824); anything beyond falls through to synthesis. Verified
        # by spot-checking aligned indices in this repo's dev session.
        n = min(len(tracks), len(short_genre), len(short_mood), len(full_caption))
        cache: dict[str, str] = {}
        for i in range(n):
            sg, sm, fc = short_genre[i], short_mood[i], full_caption[i]
            # Prefer compact "<genre>. <mood>" form — fits T5 cleanly and stays
            # diverse across tracks. If those are junk, try the long caption.
            parts = []
            if _granite_caption_ok(sg):
                parts.append(sg.rstrip(". ").rstrip())
            if _granite_caption_ok(sm):
                parts.append(sm.rstrip(". ").rstrip())
            if parts:
                cap = ". ".join(parts) + "."
            elif _granite_caption_ok(fc):
                cap = fc.strip()
            else:
                continue
            if len(cap) > GRANITE_PROMPT_BUDGET:
                cap = cap[:GRANITE_PROMPT_BUDGET].rsplit(" ", 1)[0] + "…"
            cache[tracks[i]] = cap
        _granite_cache = cache
    except Exception as e:
        warnings.warn(f"goa_metadata: failed to load Granite captions ({e}); falling back to synthesis")
        _granite_cache = {}
    return _granite_cache


def _hash01(s: str, salt: str = "") -> float:
    """Deterministic hash of `s` (+ salt) into [0, 1). Stable across processes."""
    h = hashlib.blake2b(f"{salt}::{s}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") / 2**64


def _track_id_from_path(latent_filename: str) -> str:
    """Parent directory name = track identifier."""
    return Path(latent_filename).parent.name


def _track_split(track_id: str) -> str:
    """Return one of {"train", "val", "exclude"} for this track."""
    bucket = _hash01(track_id, salt=str(GOA_SPLIT_SEED))
    if bucket >= GOA_SUBSET_FRAC:
        return "exclude"
    # Re-hash inside the included subset so train/val split is independent of include/exclude.
    val_bucket = _hash01(track_id, salt=f"{GOA_SPLIT_SEED}_valhash")
    if val_bucket < GOA_VAL_OF_SUBSET:
        return "val"
    return "train"


def _build_prompt(info: dict, track_id: str = "") -> str:
    """Prefer a Granite music-flamingo caption when the track has one,
    falling back to a metadata-synthesised descriptor otherwise. The global
    conditioners (BPM, syncopation, ...) carry the structural information
    regardless, so a sparse text prompt is fine."""
    granite = _load_granite_captions().get(track_id) if track_id else None
    bpm = info.get("bpm") or info.get("bpm_essentia")
    tonic_scale = info.get("tonic_scale")  # "major" | "minor"

    if granite:
        tail = []
        if bpm:
            tail.append(f"{round(float(bpm))} BPM")
        if tonic_scale:
            tail.append(str(tonic_scale))
        return granite if not tail else f"{granite} ({', '.join(tail)})"

    # ---- synthesis fallback ----
    parts = ["Goa Trance"]
    title = info.get("track_metadata_title") or info.get("title")
    artist = info.get("track_metadata_artist") or info.get("artist")
    genre = info.get("track_metadata_genre") or info.get("essentia_genre")
    year = info.get("track_metadata_year") or info.get("release_year")

    if isinstance(genre, dict):
        # essentia_genre is sometimes a dict of {label: prob}; pick the top
        try:
            genre = max(genre.items(), key=lambda kv: kv[1])[0]
        except Exception:
            genre = None
    if genre and "Goa" not in str(genre):
        parts.append(str(genre))
    if artist:
        parts.append(f"by {artist}")
    if year:
        parts.append(str(int(year)))
    if bpm:
        parts.append(f"{round(float(bpm))} BPM")
    if tonic_scale:
        parts.append(str(tonic_scale))

    prompt = ", ".join(parts)
    # Some Goa releases have very long album/title strings — clamp to T5's 64-token max.
    return prompt[:200]


def _clip(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


def _safe_get(info: dict, key: str, default: float) -> float:
    v = info.get(key)
    if v is None:
        return float(default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


# -----------------------------------------------------------------------------
# The function Stable Audio Open's dataset machinery calls.
# -----------------------------------------------------------------------------

def get_custom_metadata(info: dict, audio: Any = None) -> dict:
    """Returns a metadata dict for one crop.

    Args:
        info: the raw .json sidecar content + `latent_filename` injected by
              PreEncodedDataset.
        audio: None for pre_encoded; would be the raw waveform tensor for
               on-the-fly encoding. Unused here.

    The dataset is split track-level: a track is in {train, val, exclude}
    based on GOA_SPLIT_SEED + the track's directory name. Tracks not matching
    GOA_SPLIT_MODE get `__reject__: True` so the dataset retries with another
    index.
    """
    latent_filename = info.get("latent_filename", "")
    track_id = _track_id_from_path(latent_filename)
    split = _track_split(track_id)
    if split != GOA_SPLIT_MODE:
        return {"__reject__": True}

    # Pull globals from the JSON. Each NumberConditioner expects a scalar.
    bpm = _safe_get(info, "bpm", _safe_get(info, "bpm_essentia", 130.0))
    syncopation = _safe_get(info, "syncopation", 0.5)
    on_beat_ratio = _safe_get(info, "on_beat_ratio", 0.5)
    rhythmic_complexity = _safe_get(info, "rhythmic_complexity", 0.5)
    rhythmic_evenness = _safe_get(info, "rhythmic_evenness", 0.5)
    seconds_total = _safe_get(info, "seconds_total", 11.36)

    return {
        "prompt": _build_prompt(info, track_id=track_id),
        "seconds_total": _clip(seconds_total, 0.0, 256.0),
        # Rhythmic globals — must match the IDs / ranges in the model_config conditioners.
        "bpm": _clip(bpm, BPM_MIN, BPM_MAX),
        "syncopation": _clip(syncopation, RHYTHM_MIN, RHYTHM_MAX),
        "on_beat_ratio": _clip(on_beat_ratio, RHYTHM_MIN, RHYTHM_MAX),
        "rhythmic_complexity": _clip(rhythmic_complexity, RHYTHM_MIN, RHYTHM_MAX),
        "rhythmic_evenness": _clip(rhythmic_evenness, RHYTHM_MIN, RHYTHM_MAX),
        # Carry the track id through for debugging / logging
        "track_id": track_id,
    }
