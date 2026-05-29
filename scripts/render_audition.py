"""Render audition clips with one base-model reference + LatCH-guided variants.

Universal filename scheme (parsed by the index.html UI):
  <variant>__r<rho>_m<mu>_g<gamma>_w<weight>_s<start>_e<end>_v<value>.flac
Special cases:
  base.flac                 = base model, no LatCH guidance, single reference

Where:
  variant   identifier of the LatCH head's optimiser config
            ("adamw_a1", "sfnormuon", "fusion_b2", etc.)
  rho       TFG ρ ("variance" in the user's vocabulary), e.g. r2 / r8
  mu        TFG μ ("mean"),                            e.g. m1 / m4
  gamma     TFG γ ("noise") × 100,                     e.g. g50 (=0.5)
  weight    LatCH config weight,                       e.g. w4 / w8 / w32
  start     window start as % × 1,                     e.g. s00 / s33
  end       window end   as % × 1,                     e.g. e70 / e100
  value     target feature value (rounded to int),     e.g. v-15

Universal => new renders can be dropped into renders/fusion_audition/
and the HTML UI picks them up via build_manifest.py.

Run:
  sat-venv/bin/python scripts/render_audition.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict
from stable_audio_tools.inference.generation import generate_diffusion_cond


# --- config -----------------------------------------------------------------

REPO     = Path("/home/kim/Projects/SAO/stable-audio-tools")
OUT_DIR  = REPO / "renders" / "fusion_audition"
BASE_CFG = REPO / "models" / "checkpoints" / "small" / "base_model_config.json"
BASE_CKP = REPO / "models" / "checkpoints" / "small" / "base_model.ckpt"

PROMPT = "Goa trance upbeat song, melancholy, korg, Roland, analog synths"
STEPS  = 40
CFG_SCALE = 7.0
SEED   = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Legacy single-feature TARGET_VALUE (kept for backward compat with Phase 1
# filenames). New code uses TARGET_VALUES[feature] which is defined below.
TARGET_VALUE = -15.0

# Variants — each entry is (name_for_filename_prefix, head_feature, ckpt_path).
# Feature is needed because the prompt is fixed but the head's target value
# scale changes per feature (rms_energy is dB, flatness is 0..1, etc.).
VARIANTS = [
    {
        "name": "adamw_a1",
        "feature": "rms_energy_bass",
        "ckpt": REPO / "latch_weights"
                     / "latch_rms_energy_bass_bakeoff_rms_energy_bass_A1_s1_best.pt",
    },
    {
        "name": "sfnormuon",
        "feature": "rms_energy_bass",
        "ckpt": REPO / "latch_weights"
                     / "latch_rms_energy_bass_components_SFNorMuon_s1_best.pt",
    },
    # Phase-2: production ship heads (§21 — SF-NorMuon + d256/dp4 + bf16)
    {
        "name": "ship_sfn_bass",
        "feature": "rms_energy_bass",
        "ckpt": REPO / "latch_weights"
                     / "latch_rms_energy_bass_ship_rms_energy_bass_sfn_s1_best.pt",
    },
    {
        "name": "ship_ff_bass",
        "feature": "rms_energy_bass",
        "ckpt": REPO / "latch_weights"
                     / "latch_rms_energy_bass_ship_rms_energy_bass_fullfusion_s1_best.pt",
    },
    {
        "name": "ship_sfn_flatness",
        "feature": "spectral_flatness",
        "ckpt": REPO / "latch_weights"
                     / "latch_spectral_flatness_ship_spectral_flatness_sfn_s1_best.pt",
    },
    {
        "name": "ship_sfn_flux",
        "feature": "spectral_flux",
        "ckpt": REPO / "latch_weights"
                     / "latch_spectral_flux_ship_spectral_flux_sfn_s1_best.pt",
    },
]

# Default target value per feature (mid-range, sane for the prompt).
# rms_energy_bass: dB scale [-60, 0]; flatness: [0, 1]; flux: feature units.
TARGET_VALUES = {
    "rms_energy_bass": -15.0,
    "spectral_flatness": 0.05,        # mid-range flatness
    "spectral_flux": 5.0,             # mid-range flux
}

# 8 hparam combos covering corners + window variations.
# Each entry is (rho, mu, weight, start, end). gamma = 0.5 is shared.
HPARAMS = [
    (2, 1,  4, 0.00, 1.00),
    (2, 1, 32, 0.00, 1.00),
    (8, 4,  4, 0.00, 1.00),
    (8, 4, 32, 0.00, 1.00),
    (2, 4,  8, 0.00, 1.00),
    (8, 1,  8, 0.00, 1.00),
    (8, 4,  8, 0.33, 0.70),
    (8, 4,  8, 0.00, 0.70),
]
GAMMA = 0.5
N_ITER = 4


def fname(variant: str, rho: int, mu: int, weight: int,
          start: float, end: float, value: float) -> str:
    """Build the universal-naming-scheme filename for a render.

    Target value is encoded as int when whole, otherwise as int(value*1000)
    with a "k" suffix (so 0.05 -> v50k, -15 -> v-15, 5 -> v5). This keeps
    sortable alpha-order while preserving precision for sub-unit features.
    """
    s = int(round(start * 100))
    e = int(round(end * 100))
    if abs(value - round(value)) < 1e-9:
        vtag = f"v{int(round(value))}"
    else:
        vtag = f"v{int(round(value * 1000))}k"
    return (f"{variant}__r{rho}_m{mu}_g{int(GAMMA*100):02d}_w{weight}"
            f"_s{s:02d}_e{e:03d}_{vtag}.flac")


def render(model, ckpt_path: Path | None, hp: tuple | None, out_path: Path,
           target_value: float = TARGET_VALUE):
    """Generate a single clip. ckpt_path / hp None => unguided base."""
    sr = model_meta["sample_rate"]
    ss = model_meta["sample_size"]
    seconds = round(ss / sr)
    cond = [{"prompt": PROMPT, "seconds_total": seconds}]

    latch_configs = None
    latch_hparams = None
    if ckpt_path is not None and hp is not None:
        rho, mu, weight, start, end = hp
        latch_configs = [{
            "model_path": str(ckpt_path),
            "kind": "constant",
            "value": float(target_value),
            "weight": float(weight),
            "start_pct": float(start),
            "end_pct":   float(end),
        }]
        latch_hparams = {
            "rho":   float(rho),
            "mu":    float(mu),
            "gamma": float(GAMMA),
            "n_iter": int(N_ITER),
            "log_norms": False,
        }

    audio = generate_diffusion_cond(
        model,
        steps=STEPS,
        cfg_scale=float(CFG_SCALE),
        conditioning=cond,
        sample_size=ss,
        seed=SEED,
        device=DEVICE,
        sigma_max=1.0,
        latch_configs=latch_configs,
        latch_hparams=latch_hparams,
    )
    # generate returns (1, C, N); squeeze to (C, N) then transpose for soundfile
    wav = audio.squeeze(0).cpu().float().numpy()
    sf.write(str(out_path), wav.T, sr, format="FLAC", subtype="PCM_16")


# --- main -------------------------------------------------------------------

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load base model once and reuse for every render
    with open(BASE_CFG) as f:
        mc = json.load(f)
    model = create_model_from_config(mc)
    model.load_state_dict(load_ckpt_state_dict(str(BASE_CKP)))
    model = model.half().to(DEVICE).eval()
    model_meta = {"sample_rate": mc["sample_rate"], "sample_size": mc["sample_size"]}
    print(f"Base model loaded ({mc['sample_rate']} Hz, "
          f"{mc['sample_size']} samples = ~{round(mc['sample_size']/mc['sample_rate'])}s)")
    print(f"Output dir: {OUT_DIR}")
    print()

    # 1. Single unguided base reference
    base_path = OUT_DIR / "base.flac"
    if not base_path.exists():
        print(f"[1/{len(VARIANTS)*len(HPARAMS)+1}] base (no LatCH guidance)")
        render(model, None, None, base_path)
    else:
        print(f"[skip] base.flac already exists")

    # 2. LatCH-guided renders
    total = len(VARIANTS) * len(HPARAMS)
    idx = 1
    for variant in VARIANTS:
        if not variant["ckpt"].exists():
            print(f"[skip] {variant['name']}: checkpoint not found at {variant['ckpt']}")
            idx += len(HPARAMS)
            continue
        # Per-feature target value (falls back to legacy TARGET_VALUE if missing)
        feature = variant.get("feature", "rms_energy_bass")
        target_value = TARGET_VALUES.get(feature, TARGET_VALUE)
        for hp in HPARAMS:
            rho, mu, weight, start, end = hp
            out_path = OUT_DIR / fname(variant["name"], rho, mu, weight, start, end,
                                       target_value)
            if out_path.exists():
                print(f"[skip] {out_path.name}")
                idx += 1
                continue
            print(f"[{idx}/{total}] {out_path.name}")
            try:
                render(model, variant["ckpt"], hp, out_path, target_value=target_value)
            except Exception as e:
                print(f"  FAILED: {e}")
            idx += 1

    print()
    print(f"Done. {len(list(OUT_DIR.glob('*.flac')))} FLAC files in {OUT_DIR}")
