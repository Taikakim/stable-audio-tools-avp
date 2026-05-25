#!/usr/bin/env python3
"""Pluggable VAE decoder: torch (default, reliable) or onnx (optional).

The torch backend uses the base model's own pretransform — guaranteed to match
the VAE space the LatCH latents live in. The onnx backend runs
models/exported_model/vae_decoder.onnx via onnxruntime; it's optional ("just in
case it breaks things") and only valid if that ONNX was exported from the SAME
autoencoder as the base model. Use --compare to check agreement before relying
on it.

CLI:
    python scripts/latch_decode.py --compare            # torch vs onnx on real latents
    python scripts/latch_decode.py --decoder onnx --npy <file.npy> --out out.wav
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

MODEL_CONFIG = "models/checkpoints/small/base_model_config.json"
CKPT = "models/checkpoints/small/base_model.ckpt"
ONNX_DECODER = "models/exported_model/vae_decoder.onnx"


def make_torch_decoder(model_config=MODEL_CONFIG, ckpt=CKPT, device="cuda", half=True):
    from stable_audio_tools.models.factory import create_model_from_config
    from stable_audio_tools.models.utils import load_ckpt_state_dict
    with open(model_config) as f:
        cfg = json.load(f)
    model = create_model_from_config(cfg)
    model.load_state_dict(load_ckpt_state_dict(ckpt))
    model = model.to(device).eval()
    if half:
        model = model.half()
    pt = model.pretransform
    dtype = torch.float16 if half else torch.float32

    @torch.no_grad()
    def decode(z):
        if not torch.is_tensor(z):
            z = torch.from_numpy(np.asarray(z))
        z = z.to(device=device, dtype=dtype)
        return pt.decode(z).float().cpu()

    decode.sample_rate = cfg["sample_rate"]
    return decode


def make_onnx_decoder(onnx_path=ONNX_DECODER, sample_rate=44100):
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise RuntimeError(
            "onnx decoder requested but onnxruntime is not installed in this venv"
        ) from e
    if not Path(onnx_path).exists():
        raise FileNotFoundError(
            f"{onnx_path} not found — export it first with export_vae_onnx.py "
            f"(pass the base model config/ckpt so it matches the latent space)."
        )
    providers = [p for p in ("MIGraphXExecutionProvider", "CPUExecutionProvider")
                 if p in ort.get_available_providers()]
    sess = ort.InferenceSession(onnx_path, providers=providers)
    in_name = sess.get_inputs()[0].name

    def decode(z):
        if torch.is_tensor(z):
            z = z.detach().cpu().numpy()
        z = np.asarray(z, dtype=np.float32)
        out = sess.run(None, {in_name: z})[0]
        return torch.from_numpy(np.asarray(out)).float()

    decode.sample_rate = sample_rate
    return decode


def make_decoder(backend="torch", **kw):
    """backend: 'torch' (default) or 'onnx'."""
    if backend == "onnx":
        return make_onnx_decoder(
            onnx_path=kw.get("onnx_path", ONNX_DECODER),
            sample_rate=kw.get("sample_rate", 44100),
        )
    return make_torch_decoder(
        model_config=kw.get("model_config", MODEL_CONFIG),
        ckpt=kw.get("ckpt", CKPT),
        device=kw.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
        half=kw.get("half", True),
    )


def _find_latents(n=3):
    root = Path("/run/media/kim/Lehto/latents")
    out, stems = [], {"_bass", "_drums", "_other", "_vocals"}
    for td in sorted(root.iterdir()):
        if not td.is_dir():
            continue
        for npy in sorted(td.glob("*.npy")):
            if any(npy.stem.endswith(s) for s in stems):
                continue
            out.append(npy)
            if len(out) >= n:
                return out
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoder", choices=["torch", "onnx"], default="torch")
    ap.add_argument("--npy", default=None, help="decode this latent .npy")
    ap.add_argument("--out", default="decoded.wav")
    ap.add_argument("--compare", action="store_true",
                    help="decode real latents with BOTH backends, report agreement")
    args = ap.parse_args()

    if args.compare:
        import soundfile as sf
        td = make_torch_decoder()
        od = make_onnx_decoder(sample_rate=td.sample_rate)
        print(f"{'latent':<34} {'rel_diff(onnx vs torch)':>24}")
        for npy in _find_latents(3):
            z = np.load(str(npy)).astype(np.float32)[None]
            a_t = td(z).squeeze(0).numpy()
            a_o = od(z).squeeze(0).numpy()
            m = min(a_t.shape[-1], a_o.shape[-1])
            rel = np.linalg.norm(a_t[..., :m] - a_o[..., :m]) / (np.linalg.norm(a_t[..., :m]) + 1e-12)
            verdict = "MATCH (same VAE)" if rel < 0.05 else "DIVERGES — different VAE / not safe"
            print(f"{npy.stem:<34} {rel:>24.4f}  {verdict}")
        return

    dec = make_decoder(args.decoder)
    if args.npy is None:
        args.npy = str(_find_latents(1)[0])
    import soundfile as sf
    z = np.load(args.npy).astype(np.float32)[None]
    audio = dec(z).squeeze(0).numpy()
    sf.write(args.out, audio.T, dec.sample_rate)
    print(f"[{args.decoder}] decoded {Path(args.npy).name} -> {args.out}  "
          f"(sr={dec.sample_rate}, shape={audio.shape})")


if __name__ == "__main__":
    main()
