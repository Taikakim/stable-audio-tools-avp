#!/home/kim/Projects/SAO/stable-audio-tools/sat-venv/bin/python
"""
encode_dataset.py — Encode audio crops to VAE latents for SAT training / latent player.

By default encodes full-mix crops to --output-dir.  Pass --stem-dir to also (or
instead) encode the separated stems (bass/drums/other/vocals) to a second location.

Usage — full-mix only (SAT training dataset):
    ./encode_dataset.py \
        --source-dir /run/media/kim/Mantu/ai-music/Goa_Separated_crops \
        --output-dir /run/media/kim/Lehto/goa-small

Usage — full-mix + stems in one pass:
    ./encode_dataset.py \
        --source-dir /run/media/kim/Mantu/ai-music/Goa_Separated_crops \
        --output-dir /run/media/kim/Lehto/goa-small \
        --stem-dir   /run/media/kim/Lehto/goa-stems

Usage — stems only (skip full-mix encoding):
    ./encode_dataset.py \
        --source-dir /run/media/kim/Mantu/ai-music/Goa_Separated_crops \
        --stem-dir   /run/media/kim/Lehto/goa-stems

Re-runs are incremental: existing .npy files are skipped.  Companion .json files are
refreshed (without re-encoding) when the source .INFO sidecar is newer than the .json.
Use --force to re-encode everything.

Output structure
  Full-mix:  output-dir / <track_folder> / <crop_name>.npy + .json
  Stems:     stem-dir   / <track_folder> / <crop_name>_<stem>.npy + .json
"""

import os

# ROCm env (HIP/MIOpen/TunableOp) is applied here, before torch is imported.
import stable_audio_tools.rocm_env  # noqa: F401

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.utils import copy_state_dict, load_ckpt_state_dict

STEM_SUFFIXES = {"_bass", "_drums", "_other", "_vocals"}
AUDIO_EXTENSIONS = {".flac", ".wav", ".mp3", ".ogg"}

REPO_ROOT = Path(__file__).parent
DEFAULT_MODEL_CONFIG = REPO_ROOT / "models/checkpoints/small/base_model_config.json"
DEFAULT_CKPT_PATH    = REPO_ROOT / "models/checkpoints/small/base_model.ckpt"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_autoencoder(model_config_path: Path, ckpt_path: Path,
                     model_half: bool, device: str):
    with open(model_config_path) as f:
        model_config = json.load(f)

    sample_rate        = model_config["sample_rate"]
    sample_size        = model_config["sample_size"]
    downsampling_ratio = model_config["model"]["pretransform"]["config"]["downsampling_ratio"]

    print(f"Creating model from config ({model_config_path.name})")
    model = create_model_from_config(model_config)

    print(f"Loading checkpoint from {ckpt_path.name} ...")
    copy_state_dict(model, load_ckpt_state_dict(str(ckpt_path)))

    if not hasattr(model, "pretransform") or model.pretransform is None:
        raise ValueError("Model has no pretransform — expected a diffusion_cond model")
    autoencoder = model.pretransform
    del model

    autoencoder.eval().requires_grad_(False)
    if model_half:
        autoencoder.to(torch.float16)
    autoencoder.to(device)

    print(f"Autoencoder ready — sr={sample_rate}, sample_size={sample_size}, "
          f"downsampling_ratio={downsampling_ratio}")
    return autoencoder, sample_rate, sample_size, downsampling_ratio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_stem(path: Path) -> bool:
    return any(path.stem.endswith(s) for s in STEM_SUFFIXES)


def get_info_sidecar(audio_path: Path) -> Path:
    """Return the .INFO path for any audio file, stripping stem suffixes if needed."""
    stem = audio_path.stem
    for suffix in STEM_SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return audio_path.with_name(stem + ".INFO")


def build_padding_mask(actual_frames: int, sample_size: int,
                       downsampling_ratio: int) -> list:
    content_latent = actual_frames // downsampling_ratio
    total_latent   = sample_size   // downsampling_ratio
    return [1] * content_latent + [0] * (total_latent - content_latent)


def collect_tasks(source_dir: Path, output_dir: Path | None,
                  stem_dir: Path | None) -> tuple[list, list]:
    """
    Returns (encode_tasks, refresh_tasks).

    encode_tasks:  list of (audio_path, npy_out) to encode
    refresh_tasks: list of (audio_path, npy_out) where .npy exists but .INFO is newer
    """
    encode_tasks  = []
    refresh_tasks = []

    track_dirs = sorted(d for d in source_dir.iterdir() if d.is_dir())
    for track_dir in tqdm(track_dirs, desc="Scanning", unit="track", dynamic_ncols=True):
        for ext in AUDIO_EXTENSIONS:
            for audio_path in sorted(track_dir.glob(f"*{ext}")):
                stem_file = is_stem(audio_path)

                if stem_file and stem_dir is not None:
                    npy_out = stem_dir / track_dir.name / audio_path.with_suffix(".npy").name
                elif not stem_file and output_dir is not None:
                    npy_out = output_dir / track_dir.name / audio_path.with_suffix(".npy").name
                else:
                    continue  # stems with no stem_dir, or full-mix with no output_dir

                json_out  = npy_out.with_suffix(".json")
                info_path = get_info_sidecar(audio_path)

                if not npy_out.exists():
                    encode_tasks.append((audio_path, npy_out))
                elif (info_path.exists() and json_out.exists() and
                        info_path.stat().st_mtime > json_out.stat().st_mtime):
                    refresh_tasks.append((audio_path, npy_out))

    return encode_tasks, refresh_tasks


def refresh_json(audio_path: Path, npy_out: Path) -> bool:
    """Rewrite companion .json from fresh .INFO, preserving seconds_total/padding_mask."""
    json_out  = npy_out.with_suffix(".json")
    info_path = get_info_sidecar(audio_path)
    try:
        existing = {}
        try:
            with open(json_out) as f:
                existing = json.load(f)
        except Exception:
            pass
        info_data = {}
        try:
            info_data = json.loads(info_path.read_text())
        except Exception:
            pass
        companion = {k: existing[k] for k in ("seconds_total", "padding_mask") if k in existing}
        companion.update({k: v for k, v in info_data.items()
                          if not isinstance(v, list) or k == "padding_mask"})
        with open(json_out, "w") as f:
            json.dump(companion, f)
        return True
    except Exception as e:
        print(f"  JSON refresh error: {audio_path.name}: {e}")
        return False


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AudioCropDataset(Dataset):
    """Loads, resamples, and pads audio crops in DataLoader worker processes."""

    def __init__(self, tasks: list, sample_rate: int, sample_size: int):
        # tasks: list of (audio_path, npy_out)
        self.tasks = tasks
        self.sample_rate = sample_rate
        self.sample_size = sample_size

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        audio_path, npy_out = self.tasks[idx]
        path_str = str(audio_path)
        npy_str  = str(npy_out)
        try:
            audio, sr = torchaudio.load(path_str)

            if sr != self.sample_rate:
                audio = torchaudio.functional.resample(audio, sr, self.sample_rate)

            if audio.shape[0] == 1:
                audio = audio.repeat(2, 1)
            elif audio.shape[0] > 2:
                audio = audio[:2]

            original_frames = audio.shape[1]
            actual_frames   = min(original_frames, self.sample_size)
            seconds_total   = original_frames / self.sample_rate

            if original_frames < self.sample_size:
                audio = F.pad(audio, (0, self.sample_size - original_frames))
            else:
                audio = audio[:, :self.sample_size]

            info_path = get_info_sidecar(Path(path_str))
            info_str  = "{}"
            if info_path.exists():
                try:
                    info_str = info_path.read_text()
                except Exception:
                    pass

            return {
                "audio":         audio,
                "path":          path_str,
                "npy_out":       npy_str,
                "actual_frames": actual_frames,
                "info_str":      info_str,
                "seconds_total": seconds_total,
                "error":         "",
            }

        except Exception as e:
            return {
                "audio":         torch.zeros(2, self.sample_size),
                "path":          path_str,
                "npy_out":       npy_str,
                "actual_frames": 0,
                "info_str":      "{}",
                "seconds_total": 0.0,
                "error":         str(e),
            }


def collate_fn(samples: list) -> dict:
    valid  = [s for s in samples if not s["error"]]
    failed = [s for s in samples if s["error"]]
    return {
        "audio":         torch.stack([s["audio"] for s in valid]) if valid else torch.empty(0),
        "paths":         [s["path"]          for s in valid],
        "npy_outs":      [s["npy_out"]       for s in valid],
        "actual_frames": [s["actual_frames"] for s in valid],
        "info_strs":     [s["info_str"]      for s in valid],
        "seconds_total": [s["seconds_total"] for s in valid],
        "failed":        [(s["path"], s["error"]) for s in failed],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Encode audio crops to VAE latents for SAT training / latent player"
    )
    parser.add_argument("--source-dir",   required=True,
                        help="Root crops directory (Goa_Separated_crops)")
    parser.add_argument("--output-dir",   default=None,
                        help="Output directory for full-mix latents (omit to skip full-mix encoding)")
    parser.add_argument("--stem-dir",     default=None,
                        help="Output directory for stem latents (omit to skip stem encoding)")
    parser.add_argument("--model-config", default=str(DEFAULT_MODEL_CONFIG))
    parser.add_argument("--ckpt-path",    default=str(DEFAULT_CKPT_PATH))
    parser.add_argument("--batch-size",   type=int, default=8)
    parser.add_argument("--num-workers",  type=int, default=4)
    parser.add_argument("--model-half",    action="store_true", default=True)
    parser.add_argument("--no-model-half", dest="model_half", action="store_false")
    parser.add_argument("--force",  action="store_true",
                        help="Re-encode even if .npy already exists")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if not args.output_dir and not args.stem_dir:
        print("Error: provide --output-dir and/or --stem-dir", file=sys.stderr)
        sys.exit(1)

    source_dir        = Path(args.source_dir)
    output_dir        = Path(args.output_dir)  if args.output_dir else None
    stem_dir          = Path(args.stem_dir)    if args.stem_dir   else None
    model_config_path = Path(args.model_config)
    ckpt_path         = Path(args.ckpt_path)

    for p, label in [(source_dir, "source-dir"), (model_config_path, "model-config"),
                     (ckpt_path, "ckpt-path")]:
        if not p.exists():
            print(f"Error: {label} not found: {p}", file=sys.stderr)
            sys.exit(1)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    if stem_dir:
        stem_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    autoencoder, sample_rate, sample_size, downsampling_ratio = load_autoencoder(
        model_config_path, ckpt_path, args.model_half, device
    )
    dtype = torch.float16 if args.model_half else torch.float32

    # Collect work
    print(f"Scanning {source_dir} ...")
    if args.force:
        # In force mode: collect all files, skip the incremental checks
        encode_tasks  = []
        refresh_tasks = []
        for track_dir in tqdm(sorted(d for d in source_dir.iterdir() if d.is_dir()),
                              desc="Scanning", unit="track", dynamic_ncols=True):
            for ext in AUDIO_EXTENSIONS:
                for audio_path in sorted(track_dir.glob(f"*{ext}")):
                    stem_file = is_stem(audio_path)
                    if stem_file and stem_dir is not None:
                        npy_out = stem_dir / track_dir.name / audio_path.with_suffix(".npy").name
                    elif not stem_file and output_dir is not None:
                        npy_out = output_dir / track_dir.name / audio_path.with_suffix(".npy").name
                    else:
                        continue
                    encode_tasks.append((audio_path, npy_out))
    else:
        encode_tasks, refresh_tasks = collect_tasks(source_dir, output_dir, stem_dir)

    n_full  = sum(1 for _, npy in encode_tasks if output_dir and str(npy).startswith(str(output_dir)))
    n_stems = sum(1 for _, npy in encode_tasks if stem_dir   and str(npy).startswith(str(stem_dir)))
    print(f"  To encode:   {n_full} full-mix,  {n_stems} stems")
    if refresh_tasks:
        print(f"  To refresh:  {len(refresh_tasks)} .json files (source .INFO is newer)")

    # JSON-only refreshes (no GPU needed)
    refreshed = sum(1 for ap, npy in refresh_tasks if refresh_json(ap, npy))
    if refreshed:
        print(f"  Refreshed {refreshed} .json files")

    if not encode_tasks:
        print("Nothing to encode.")
        return

    print(f"\nEncoding {len(encode_tasks)} files — "
          f"batch_size={args.batch_size}, num_workers={args.num_workers}, "
          f"half={args.model_half}, device={device}")

    dataset = AudioCropDataset(encode_tasks, sample_rate, sample_size)
    loader  = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_fn,
        pin_memory=(device == "cuda"),
    )

    encoded   = 0
    errors    = 0
    error_log = source_dir / "_encode_errors.txt"

    for batch in tqdm(loader, desc="Encoding", unit="batch"):
        for path_str, err in batch["failed"]:
            tqdm.write(f"  Load error: {Path(path_str).name}: {err}")
            errors += 1
            with open(error_log, "a") as ef:
                ef.write(f"LOAD\t{path_str}\t{err}\n")

        if not batch["paths"]:
            continue

        audio_batch = batch["audio"].to(device=device, dtype=dtype)

        try:
            with torch.no_grad():
                latents = autoencoder.encode(audio_batch)
            latents_np = latents.cpu().float().numpy()
        except Exception as e:
            tqdm.write(f"  Encode error at {Path(batch['paths'][0]).name}: {e}")
            errors += len(batch["paths"])
            with open(error_log, "a") as ef:
                for p in batch["paths"]:
                    ef.write(f"ENCODE\t{p}\t{e}\n")
            continue

        for i, (path_str, npy_str, actual_frames, info_str, seconds_total) in enumerate(zip(
            batch["paths"], batch["npy_outs"], batch["actual_frames"],
            batch["info_strs"], batch["seconds_total"]
        )):
            npy_out  = Path(npy_str)
            json_out = npy_out.with_suffix(".json")
            npy_out.parent.mkdir(parents=True, exist_ok=True)

            try:
                info_data = json.loads(info_str)
            except Exception:
                info_data = {}

            padding_mask = build_padding_mask(actual_frames, sample_size, downsampling_ratio)
            companion = {"seconds_total": seconds_total, "padding_mask": padding_mask}
            companion.update({k: v for k, v in info_data.items()
                              if not isinstance(v, list) or k == "padding_mask"})

            np.save(str(npy_out), latents_np[i])
            with open(json_out, "w") as f:
                json.dump(companion, f)

            encoded += 1

    print(f"\nDone. Encoded: {encoded}  Errors: {errors}")
    if errors:
        print(f"Error log: {error_log}")
    if encoded > 0:
        sample = np.load(str(npy_out))
        print(f"Latent shape: {sample.shape}  "
              f"(expected [{autoencoder.encoded_channels}, {sample_size // downsampling_ratio}])")


if __name__ == "__main__":
    main()
