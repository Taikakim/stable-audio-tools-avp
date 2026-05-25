#!/usr/bin/env python3
"""Closed-loop verification that an rms_energy_bass LatCH head controls bass RMS.

For each target dB level: generate (guidance, constant target) -> return latent ->
decode (torch default, or onnx) -> run mir's REAL rms_energy_bass_ts extractor on
the decoded audio -> report measured mean bass RMS. If the head controls the
feature, measured bass RMS should rise monotonically with the requested level
(and differ from the no-guidance baseline). Measures control, not just change.

Also benchmarks PT vs ONNX decode (absolute ms/clip + relative speedup, plus the
numerical difference) on the generated latents.

Run:
    python scripts/latch_verify_rms.py --head latch_weights/latch_rms_energy_bass_ep5.pt
    python scripts/latch_verify_rms.py --decoder onnx
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "scripts")
sys.path.insert(0, "/home/kim/Projects/mir/src")

from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict
from stable_audio_tools.inference.generation import generate_diffusion_cond
from latch_decode import make_decoder

from spectral.timeseries_features import _compute_multiband_rms_ts  # mir's real extractor

MODEL_CONFIG = "models/checkpoints/small/base_model_config.json"
CKPT = "models/checkpoints/small/base_model.ckpt"


def measure_bass_rms(audio_ct, sr):
    """audio_ct: [C, N] float -> mean bass RMS (dB) and the 256-frame curve."""
    mono = audio_ct.mean(axis=0).astype(np.float64)
    ts = _compute_multiband_rms_ts(mono, sr, 256)["rms_energy_bass_ts"]
    ts = np.asarray(ts, dtype=np.float64)
    return float(ts.mean()), ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default="latch_weights/latch_rms_energy_bass_ep5.pt")
    ap.add_argument("--decoder", choices=["torch", "onnx"], default="torch")
    ap.add_argument("--levels", default="-50,-30,-10",
                    help="constant target dB levels to request")
    ap.add_argument("--gain", type=float, default=5.0, help="rho=mu")
    ap.add_argument("--window", default="0.5,1.0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=7.0)
    ap.add_argument("--prompt", default="goa trance, hypnotic, driving")
    ap.add_argument("--benchmark", action="store_true",
                    help="also benchmark PT vs ONNX decode speed")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    s0, s1 = [float(x) for x in args.window.split(",")]
    levels = [float(x) for x in args.levels.split(",")]

    with open(MODEL_CONFIG) as f:
        cfg = json.load(f)
    model = create_model_from_config(cfg)
    model.load_state_dict(load_ckpt_state_dict(CKPT))
    model = model.half().to(device).eval()
    sr = cfg["sample_rate"]
    sample_size = cfg["sample_size"]
    seconds = round(sample_size / sr)
    print(f"Model loaded. sr={sr} ~{seconds}s  head={Path(args.head).name}  "
          f"decoder={args.decoder}  gain={args.gain}  window=[{s0},{s1}]")

    if args.decoder == "onnx":
        decode = make_decoder("onnx", sample_rate=sr)
    else:
        @torch.no_grad()
        def decode(z):  # reuse the already-loaded pretransform (no 2nd model in VRAM)
            if not torch.is_tensor(z):
                z = torch.from_numpy(np.asarray(z))
            return model.pretransform.decode(z.to(device).half()).float().cpu()

    def gen_latent(latch_configs):
        z = generate_diffusion_cond(
            model, steps=args.steps, cfg_scale=args.cfg,
            conditioning=[{"prompt": args.prompt, "seconds_total": seconds}],
            sample_size=sample_size, seed=args.seed, device=device, sigma_max=1.0,
            return_latents=True, latch_configs=latch_configs,
            latch_hparams={"rho": args.gain, "mu": args.gain, "gamma": 0.3,
                           "n_iter": 4, "log_norms": False} if latch_configs else None,
        )
        return z

    rows = []
    # baseline (no guidance)
    z_base = gen_latent(None)
    a = decode(z_base).squeeze(0).numpy()
    m, _ = measure_bass_rms(a, sr)
    rows.append(("baseline", None, m))
    print(f"\n[baseline] measured bass RMS = {m:.2f} dB")

    latents_for_bench = [z_base]
    for lvl in levels:
        cfgs = [{"model_path": args.head, "kind": "constant", "value": lvl,
                 "weight": 1.0, "start_pct": s0, "end_pct": s1}]
        z = gen_latent(cfgs)
        latents_for_bench.append(z)
        a = decode(z).squeeze(0).numpy()
        m, _ = measure_bass_rms(a, sr)
        rows.append(("guided", lvl, m))
        print(f"[target={lvl:+.0f} dB] measured bass RMS = {m:.2f} dB")

    # verdict: does measured bass RMS track the requested level?
    g = [(lvl, m) for k, lvl, m in rows if k == "guided"]
    xs = np.array([lvl for lvl, _ in g])
    ys = np.array([m for _, m in g])
    base_m = rows[0][2]
    print("\n===== rms control summary =====")
    print(f"{'requested dB':>13} {'measured dB':>12} {'Δ vs baseline':>14}")
    print(f"{'baseline':>13} {base_m:>12.2f} {0.0:>14.2f}")
    for lvl, m in g:
        print(f"{lvl:>13.0f} {m:>12.2f} {m - base_m:>14.2f}")
    if len(xs) >= 2:
        corr = float(np.corrcoef(xs, ys)[0, 1])
        mono = bool(np.all(np.diff(ys) > 0))
        print(f"\ncorrelation(requested, measured) = {corr:+.3f}   "
              f"monotonic-increasing = {mono}")
        if corr > 0.8 and ys.max() - ys.min() > 1.0:
            print("VERDICT: head controls bass RMS (measured tracks requested).")
        elif corr > 0.3:
            print("VERDICT: weak control — measured partially tracks requested.")
        else:
            print("VERDICT: NO control — measured does not track requested target.")

    if args.benchmark:
        benchmark_decoders(latents_for_bench, sr)


def benchmark_decoders(latents, sr, n_iter=10):
    print("\n===== PT vs ONNX decode benchmark =====")
    td = make_decoder("torch")
    try:
        od = make_decoder("onnx", sample_rate=sr)
    except Exception as e:
        print(f"onnx decoder unavailable: {e}")
        return

    def timeit(dec, name, sync):
        # warmup (compile / cudnn / migraphx)
        _ = dec(latents[0])
        if sync:
            torch.cuda.synchronize()
        diffs = []
        t0 = time.perf_counter()
        for _ in range(n_iter):
            for z in latents:
                _ = dec(z)
        if sync:
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / (n_iter * len(latents))
        return dt

    t_torch = timeit(td, "torch", sync=True)
    t_onnx = timeit(od, "onnx", sync=False)
    # numerical agreement on one latent
    a_t = td(latents[0]).squeeze(0).numpy()
    a_o = od(latents[0]).squeeze(0).numpy()
    m = min(a_t.shape[-1], a_o.shape[-1])
    rel = np.linalg.norm(a_t[..., :m] - a_o[..., :m]) / (np.linalg.norm(a_t[..., :m]) + 1e-12)

    print(f"torch decode: {t_torch*1000:8.2f} ms/clip")
    print(f"onnx  decode: {t_onnx*1000:8.2f} ms/clip")
    faster, slower = ("onnx", "torch") if t_onnx < t_torch else ("torch", "onnx")
    ratio = max(t_torch, t_onnx) / max(1e-9, min(t_torch, t_onnx))
    print(f"relative: {faster} is {ratio:.2f}x faster than {slower}  "
          f"(abs Δ {abs(t_torch - t_onnx)*1000:.2f} ms/clip)")
    print(f"numerical agreement: rel diff = {rel:.4f}")


if __name__ == "__main__":
    main()
