"""
Inference comparison: Local edits vs Fresh upstream stable-audio-tools.
Uses the Small model checkpoint at models/checkpoints/small/

Run with: sat-venv/bin/python inference_comparison.py
"""
import matplotlib
matplotlib.use('Agg')

import sys
import os
import json
import time
import stable_audio_tools.rocm_env  # set HIP/MIOpen/TunableOp env before torch
import torch
import torchaudio
import numpy as np

# ─── Configuration ────────────────────────────────────────────────────
MODEL_CONFIG_PATH = "models/checkpoints/small/model_config.json"
MODEL_CKPT_PATH   = "models/checkpoints/small/model.ckpt"
OUTPUT_DIR         = "inference_comparison_outputs"
UPSTREAM_DIR       = "upstream_fresh"

PROMPTS = [
    {"prompt": "drum breaks 174 BPM",                                       "seconds_total": 10},
    {"prompt": "A short, beautiful piano riff in C minor",                  "seconds_total": 10},
    {"prompt": "Synth pluck arp with reverb and delay, 128 BPM",           "seconds_total": 10},
    {"prompt": "Birds singing in the forest",                               "seconds_total": 10},
]

SEED = 42
STEPS = 8  # rf_denoiser uses few steps
CFG_SCALE = 1.0
SAMPLER_TYPE = "pingpong"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_model(config_path, ckpt_path, create_model_fn, load_ckpt_fn, copy_state_dict_fn):
    """Load model from config + checkpoint using the provided library functions."""
    with open(config_path) as f:
        model_config = json.load(f)

    print(f"  Creating model from config...")
    model = create_model_fn(model_config)

    print(f"  Loading checkpoint: {ckpt_path}")
    state_dict = load_ckpt_fn(ckpt_path)
    copy_state_dict_fn(model, state_dict)
    del state_dict

    model = model.cuda().eval()
    model.model.half()
    return model, model_config


def run_inference(model, model_config, generate_fn, label, prompts):
    """Run inference for all prompts and return timing info."""
    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]
    device = next(model.parameters()).device
    results = []

    for i, cond in enumerate(prompts):
        prompt_text = cond["prompt"]
        print(f"\n  [{label}] Generating #{i}: '{prompt_text}' ({cond['seconds_total']}s)")

        conditioning = [cond]

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad(), torch.cuda.amp.autocast():
            audio = generate_fn(
                model=model,
                conditioning=conditioning,
                steps=STEPS,
                cfg_scale=CFG_SCALE,
                sample_size=sample_size,
                seed=SEED + i,
                device=device,
                batch_size=1,
                sampler_type=SAMPLER_TYPE,
            )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        # Post-process
        audio = audio.squeeze(0).to(torch.float32)
        peak = torch.max(torch.abs(audio))
        if peak > 0:
            audio = audio / peak
        audio = audio.clamp(-1, 1).cpu()

        # Save
        safe_prompt = prompt_text[:40].replace(' ', '_').replace(',', '')
        filename = os.path.join(OUTPUT_DIR, f"{label}_{i}_{safe_prompt}.wav")
        torchaudio.save(filename, audio, sample_rate)

        # Stats
        rms = torch.sqrt(torch.mean(audio ** 2)).item()
        duration_samples = audio.shape[-1]
        duration_sec = duration_samples / sample_rate

        results.append({
            "prompt": prompt_text,
            "file": filename,
            "elapsed_sec": elapsed,
            "rms": rms,
            "duration_sec": duration_sec,
            "audio_shape": list(audio.shape),
        })
        print(f"    -> Saved: {filename}  ({elapsed:.2f}s, RMS={rms:.4f})")

    return results


def main():
    print("=" * 70)
    print("INFERENCE COMPARISON: Local Edits vs Fresh Upstream")
    print("=" * 70)
    print(f"Model config: {MODEL_CONFIG_PATH}")
    print(f"Checkpoint:   {MODEL_CKPT_PATH}")
    print(f"Steps: {STEPS}, CFG: {CFG_SCALE}, Sampler: {SAMPLER_TYPE}, Seed: {SEED}")
    print(f"Prompts: {len(PROMPTS)}")
    print()

    # ─── Phase 1: LOCAL (edited) version ──────────────────────────────
    print("=" * 70)
    print("PHASE 1: LOCAL (edited) stable_audio_tools")
    print("=" * 70)

    # The local package is already importable (it's in site-packages or cwd)
    from stable_audio_tools.models import create_model_from_config
    from stable_audio_tools.models.utils import load_ckpt_state_dict, copy_state_dict
    from stable_audio_tools.inference.generation import generate_diffusion_cond

    print(f"  Library path: {os.path.dirname(os.path.abspath(create_model_from_config.__module__.replace('.', '/') + '.py'))}")

    model_local, config_local = load_model(
        MODEL_CONFIG_PATH, MODEL_CKPT_PATH,
        create_model_from_config, load_ckpt_state_dict, copy_state_dict,
    )

    local_results = run_inference(model_local, config_local, generate_diffusion_cond, "local", PROMPTS)

    # Free GPU memory
    del model_local
    torch.cuda.empty_cache()

    # ─── Phase 2: UPSTREAM (fresh) version ────────────────────────────
    print()
    print("=" * 70)
    print("PHASE 2: UPSTREAM (fresh) stable_audio_tools")
    print("=" * 70)

    # We need to import from the upstream clone.
    # Remove the local package from sys.modules so we can load the upstream one.
    mods_to_remove = [k for k in sys.modules if k.startswith("stable_audio_tools")]
    for k in mods_to_remove:
        del sys.modules[k]

    # Prepend upstream dir to sys.path so it gets priority
    upstream_abs = os.path.abspath(UPSTREAM_DIR)
    sys.path.insert(0, upstream_abs)

    from stable_audio_tools.models import create_model_from_config as create_model_upstream
    from stable_audio_tools.models.utils import load_ckpt_state_dict as load_ckpt_upstream, copy_state_dict as copy_upstream
    from stable_audio_tools.inference.generation import generate_diffusion_cond as generate_upstream

    # Verify we loaded from upstream
    import stable_audio_tools as _sat
    print(f"  Library path: {os.path.dirname(os.path.abspath(_sat.__file__))}")

    model_upstream, config_upstream = load_model(
        MODEL_CONFIG_PATH, MODEL_CKPT_PATH,
        create_model_upstream, load_ckpt_upstream, copy_upstream,
    )

    upstream_results = run_inference(model_upstream, config_upstream, generate_upstream, "upstream", PROMPTS)

    del model_upstream
    torch.cuda.empty_cache()

    # ─── Summary ──────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Prompt':<50} {'Local (s)':>10} {'Upstream (s)':>12} {'ΔRMS':>8}")
    print("-" * 82)
    for lr, ur in zip(local_results, upstream_results):
        delta_rms = lr["rms"] - ur["rms"]
        print(f"{lr['prompt'][:50]:<50} {lr['elapsed_sec']:>10.2f} {ur['elapsed_sec']:>12.2f} {delta_rms:>+8.4f}")

    total_local = sum(r["elapsed_sec"] for r in local_results)
    total_upstream = sum(r["elapsed_sec"] for r in upstream_results)
    print("-" * 82)
    print(f"{'TOTAL':<50} {total_local:>10.2f} {total_upstream:>12.2f}")
    print()
    print(f"Output files saved to: {OUTPUT_DIR}/")
    print("Done!")


if __name__ == "__main__":
    main()
