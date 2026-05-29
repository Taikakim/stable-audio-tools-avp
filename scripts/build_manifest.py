"""Scan renders/fusion_audition/ for .flac files and emit manifest.json.

Parses the universal-naming-scheme filenames produced by
scripts/render_audition.py and writes a manifest.json that the index.html
UI consumes. Re-run whenever new clips are added.

Filename grammar:
  <variant>__r<rho>_m<mu>_g<gamma>_w<weight>_s<start>_e<end>_v<value>.flac
  base.flac                       => unguided reference
  <variant>__noguidance.flac      => alternate unguided form (any variant)

Run:
  sat-venv/bin/python scripts/build_manifest.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO     = Path("/home/kim/Projects/SAO/stable-audio-tools")
RENDERS  = REPO / "renders" / "fusion_audition"
MANIFEST = RENDERS / "manifest.json"

# Parameter prefix => spec name => post-parse cast
PARAM_GRAMMAR = [
    ("r", "rho",   int),
    ("m", "mu",    int),
    ("g", "gamma", lambda s: int(s) / 100.0),     # g50 => 0.5
    ("w", "weight", int),
    ("s", "start", lambda s: int(s) / 100.0),     # s33 => 0.33
    ("e", "end",   lambda s: int(s) / 100.0),     # e100 => 1.00
    # v-15 => -15, or v50k => 0.05 (the k suffix = divide by 1000 for sub-unit values).
    ("v", "value", lambda s: int(s[:-1]) / 1000.0 if s.endswith("k") else int(s)),
]


def parse_filename(name: str) -> dict | None:
    """Parse a single FLAC filename into its parameter dict."""
    stem = Path(name).stem
    if stem == "base":
        return {"file": name, "variant": "base", "guided": False}

    if "__" not in stem:
        return None
    variant, rest = stem.split("__", 1)
    if rest == "noguidance":
        return {"file": name, "variant": variant, "guided": False}

    out = {"file": name, "variant": variant, "guided": True}
    # Split on underscores between tokens (each token starts with a letter)
    # e.g.  r2_m1_g50_w4_s00_e100_v-15
    tokens = rest.split("_")
    for token in tokens:
        for prefix, key, caster in PARAM_GRAMMAR:
            if token.startswith(prefix):
                try:
                    out[key] = caster(token[len(prefix):])
                except (ValueError, TypeError):
                    pass
                break
    return out


def build():
    if not RENDERS.exists():
        RENDERS.mkdir(parents=True, exist_ok=True)
    entries = []
    for f in sorted(RENDERS.glob("*.flac")):
        entry = parse_filename(f.name)
        if entry is None:
            continue
        entry["size_bytes"] = f.stat().st_size
        entries.append(entry)

    # Group variants and collect axis values
    variants = sorted({e["variant"] for e in entries})
    axes = {key: sorted({e[key] for e in entries if key in e})
            for _, key, _ in PARAM_GRAMMAR}

    manifest = {
        "version": 1,
        "render_settings": {
            "prompt": "Goa trance upbeat song, melancholy, korg, Roland, analog synths",
            "steps": 40,
            "cfg_scale": 7.0,
            "model": "Stable Audio Open Small",
            "seed": 42,
        },
        "variants": variants,
        "axes": axes,
        "entries": entries,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {MANIFEST} with {len(entries)} entries across {len(variants)} variants.")
    print(f"Variants: {variants}")
    print(f"Axes: {axes}")


if __name__ == "__main__":
    build()
