#!/usr/bin/env python3
"""Closed-loop verification for the beat_activations LatCH head.

Generate with beat-grid guidance (target = BPM) across 5 BPM-free rhythmic prompts
and fixed seeds, decode, and measure the OUTPUT's rhythm with mir's tools (librosa
backend, CPU — no VRAM contention) plus an onset-envelope periodicity metric at the
target BPM lag (the real "does it lock to the grid" measure).

Tests: (a) tempo control — guide to 90 vs 150 BPM, does detected tempo / periodicity
shift? (b) dose-response — weight sweep on one prompt. 3 seeds each.

Stable Audio loads half-precision; beat/onset analysis is CPU.
Usage: python scripts/latch_verify_rhythm.py [--steps 20]
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, "/home/kim/Projects/mir/src")
from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict
from stable_audio_tools.inference.generation import generate_diffusion_cond
from rhythm.beat_grid import detect_beats
from rhythm.bpm import calculate_bpm_from_beats
from rhythm.onsets import analyze_onsets
import librosa

MODEL_CONFIG = "models/checkpoints/small/base_model_config.json"
CKPT = "models/checkpoints/small/base_model.ckpt"
HEAD = "latch_weights/latch_beat_activations_v2_best.pt"
OUT = "latch_rhythm_out"
PROMPTS = [
    "drum loop",
    "techno beat, snappy 909 kick, 909 open hihat, powerful bassline",
    "breakbeat drum loop, chopped amen break, jungle",
    "four on the floor house groove, punchy kick, shaker",
    "tribal percussion rhythm, congas, toms, hand drums",
]
PROBE_BPMS = [90, 120, 150]   # periodicity is measured at each of these lags


def periodicity(env, sr, hop, bpm):
    """Normalized onset-envelope autocorrelation at the beat lag for `bpm`."""
    lag = int(round((60.0 / bpm) * sr / hop))
    if lag < 1 or lag >= len(env):
        return 0.0
    e = env - env.mean()
    ac = np.correlate(e, e, mode="full")
    ac = ac[len(ac) // 2:]
    return float(ac[lag] / ac[0]) if ac[0] > 0 else 0.0


def analyze(wav_path, dur):
    out = {}
    try:
        bt, _ = detect_beats(wav_path, method="librosa")
        if len(bt) >= 2:
            bpm, st = calculate_bpm_from_beats(np.asarray(bt), dur)
            out["bpm"] = bpm
            out["tightness"] = max(0.0, 1 - st["std_interval"] / (st["mean_interval"] + 1e-9))
    except Exception as e:
        print(f"   [beat failed: {e}]")
    try:
        o = analyze_onsets(wav_path)
        out["onset_density"] = o["onset_density"]
        out["onset_str"] = o["onset_strength_mean"]
    except Exception as e:
        print(f"   [onset failed: {e}]")
    try:
        y, sr = librosa.load(wav_path, sr=None, mono=True)
        env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
        for b in PROBE_BPMS:
            out[f"per{b}"] = periodicity(env, sr, 512, b)
    except Exception as e:
        print(f"   [periodicity failed: {e}]")
    return out or None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--gain", type=float, default=8.0)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = json.load(open(MODEL_CONFIG))
    model = create_model_from_config(cfg)
    model.load_state_dict(load_ckpt_state_dict(CKPT))
    model = model.half().to(dev).eval()
    sr, ss = cfg["sample_rate"], cfg["sample_size"]
    dur = ss / sr
    if dev == "cuda":
        torch.cuda.synchronize()
        print(f"VRAM after Stable Audio (half): {torch.cuda.memory_allocated()/1e9:.2f} GB")

    seeds = [42, 123, 7]
    # (prompt_idx, seed, tag, weight, target_bpm)
    combos = []
    for pi in range(len(PROMPTS)):
        for s in seeds:
            combos += [(pi, s, "base", 0.0, None),
                       (pi, s, "g90", 30.0, 90),
                       (pi, s, "g150", 30.0, 150)]
    for s in seeds:                                  # weight sweep on prompt 0
        for w in (15.0, 30.0, 50.0):
            combos.append((0, s, f"w{int(w)}@120", w, 120))

    results = defaultdict(list)  # key=(prompt_idx, tag)
    for pi, seed, tag, w, bpm in combos:
        lc = None
        if w > 0 and bpm is not None:
            lc = [{"model_path": HEAD, "kind": "beat_grid", "value": float(bpm),
                   "weight": float(w), "start_pct": 0.4, "end_pct": 1.0}]
        z = generate_diffusion_cond(
            model, steps=args.steps, cfg_scale=3.0,
            conditioning=[{"prompt": PROMPTS[pi], "seconds_total": round(dur)}],
            sample_size=ss, seed=seed, device=dev, sigma_max=1.0, return_latents=True,
            latch_configs=lc,
            latch_hparams={"rho": args.gain, "mu": args.gain, "gamma": 0.3, "n_iter": 6} if lc else None,
        )
        audio = model.pretransform.decode(z).squeeze(0).float().cpu().numpy()
        wp = os.path.join(OUT, f"p{pi}_{tag}_s{seed}.wav")
        sf.write(wp, audio.T, sr)
        m = analyze(wp, dur)
        if m:
            results[(pi, tag)].append(m)

    def avg(key, k):
        vs = [m[k] for m in results.get(key, []) if k in m]
        return np.mean(vs) if vs else float("nan")

    print("\n===== TEMPO CONTROL (mean/3 seeds) — does guiding 90 vs 150 shift the output? =====")
    print(f"{'prompt':42} {'setting':6} {'det_bpm':>8} {'per90':>7} {'per150':>7} {'tight':>6} {'ons_str':>8}")
    for pi, p in enumerate(PROMPTS):
        for tag in ("base", "g90", "g150"):
            print(f"{p[:42]:42} {tag:6} {avg((pi,tag),'bpm'):8.1f} {avg((pi,tag),'per90'):7.3f} "
                  f"{avg((pi,tag),'per150'):7.3f} {avg((pi,tag),'tightness'):6.3f} {avg((pi,tag),'onset_str'):8.3f}")
    print("\n===== WEIGHT SWEEP (prompt 0 'drum loop', target 120) =====")
    print(f"{'setting':10} {'det_bpm':>8} {'per120':>7} {'onset_str':>9} {'onset_dens':>10}")
    for tag in ("base", "w15@120", "w30@120", "w50@120"):
        print(f"{tag:10} {avg((0,tag),'bpm'):8.1f} {avg((0,tag),'per120'):7.3f} "
              f"{avg((0,tag),'onset_str'):9.3f} {avg((0,tag),'onset_density'):10.2f}")


if __name__ == "__main__":
    main()
