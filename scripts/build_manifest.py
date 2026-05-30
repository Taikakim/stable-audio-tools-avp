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
    """Parse a single FLAC filename into its parameter dict.

    Supports two filename forms:
      legacy:  <variant>__<params>.flac                     (single prompt)
      new:     <variant>__<prompt>__<params>.flac           (multi-prompt)

    Plus the unguided special cases:
      base.flac                                              -> any variant, no LatCH
      <variant>__noguidance.flac                             -> per-variant no-LatCH
    """
    stem = Path(name).stem
    if stem == "base":
        return {"file": name, "variant": "base", "prompt": "goa", "guided": False}

    if "__" not in stem:
        return None

    parts = stem.split("__")
    if len(parts) == 2:
        # Legacy 2-token form: <variant>__<rest>. Treat as goa prompt by convention.
        variant, rest = parts
        prompt = "goa"
    elif len(parts) >= 3:
        variant, prompt, rest = parts[0], parts[1], "__".join(parts[2:])
    else:
        return None

    if rest == "noguidance":
        return {"file": name, "variant": variant, "prompt": prompt, "guided": False}

    out = {"file": name, "variant": variant, "prompt": prompt, "guided": True}
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
    prompts = sorted({e["prompt"] for e in entries})
    axes = {key: sorted({e[key] for e in entries if key in e})
            for _, key, _ in PARAM_GRAMMAR}

    # Optional prompt-tag -> full-text map for the UI's tooltip; written
    # iff scripts/render_audition.py left a PROMPTS dict alongside us.
    prompts_dir_index = RENDERS / "prompts.json"
    prompt_texts = {}
    if prompts_dir_index.exists():
        try:
            prompt_texts = json.loads(prompts_dir_index.read_text())
        except Exception:
            prompt_texts = {}

    manifest = {
        "version": 2,
        "render_settings": {
            "steps": 40,
            "cfg_scale": 7.0,
            "model": "Stable Audio Open Small",
            "seed": 42,
        },
        "variants": variants,
        "prompts": prompts,
        "prompt_texts": prompt_texts,
        "axes": axes,
        "entries": entries,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {MANIFEST} with {len(entries)} entries across {len(variants)} variants.")
    print(f"Variants: {variants}")
    print(f"Axes: {axes}")


if __name__ == "__main__":
    build()
