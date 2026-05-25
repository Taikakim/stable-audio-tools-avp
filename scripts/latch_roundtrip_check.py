#!/usr/bin/env python3
"""Verify LatCH training latents live in the base model's VAE space.

Decodes a few /latents .npy through the base model's pretransform, then
re-encodes. If these latents were produced by the SAME VAE, the round-trip
relative error ||encode(decode(z)) - z|| / ||z|| is small. A large error
(or noise-like decoded audio) means the latents came from a different encoder
and every LatCH head is querying out-of-distribution inputs at inference.

Also writes the decoded audio so it can be auditioned.
"""

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict

MODEL_CONFIG = "models/checkpoints/small/base_model_config.json"
CKPT = "models/checkpoints/small/base_model.ckpt"
LATENT_DIR = Path("/run/media/kim/Lehto/latents")
OUT = Path("latch_roundtrip_out")


def find_latents(n=3):
    out, stems = [], {"_bass", "_drums", "_other", "_vocals"}
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


def spectral_flatness(x):
    # x: 1-D float; ~1.0 => white noise, low => tonal/music
    X = np.abs(np.fft.rfft(x)) + 1e-10
    return float(np.exp(np.log(X).mean()) / X.mean())


def main():
    OUT.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(MODEL_CONFIG) as f:
        cfg = json.load(f)
    model = create_model_from_config(cfg)
    model.load_state_dict(load_ckpt_state_dict(CKPT))
    model = model.half().to(device).eval()
    pt = model.pretransform
    sr = cfg["sample_rate"]
    print(f"Model loaded. sr={sr}  device={device}")

    for npy in find_latents(3):
        z = torch.from_numpy(np.load(str(npy)).astype(np.float32))[None].to(device).half()
        with torch.no_grad():
            audio = pt.decode(z)
            z2 = pt.encode(audio)
        rel = (z2.float() - z.float()).norm() / (z.float().norm() + 1e-12)
        wav = audio.squeeze(0).cpu().float().numpy()  # [C, N]
        flat = spectral_flatness(wav[0])
        peak, rms = float(np.abs(wav).max()), float(np.sqrt((wav ** 2).mean()))
        sf.write(str(OUT / f"{npy.stem}.wav"), wav.T, sr)
        verdict = "SAME VAE (round-trips)" if rel < 0.35 else "MISMATCH? large round-trip error"
        print(f"\n{npy.stem}")
        print(f"  z shape={tuple(z.shape)}  ||z||={z.float().norm():.2f}")
        print(f"  latent round-trip rel err = {rel:.4f}   -> {verdict}")
        print(f"  decoded audio: peak={peak:.3f} rms={rms:.3f} spectral_flatness={flat:.4f} "
              f"({'noise-like' if flat > 0.5 else 'music-like'})")
    print(f"\nDecoded WAVs in {OUT}/ — listen to confirm they sound like real music.")


if __name__ == "__main__":
    main()
