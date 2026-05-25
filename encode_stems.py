#!/home/kim/Projects/SAO/stable-audio-tools/sat-venv/bin/python
"""
encode_stems.py — Encode separated stem crops to latents for the latent crossfader.

Finds all stem audio files (bass/drums/other/vocals) under the given source track
directory, encodes them through the frozen VAE pretransform, and writes .npy + .json
pairs into the stem latent directory (separate from the SAT training dataset).

Usage — encode all tracks (recommended):
    ./encode_stems.py \\
        --source-dir /run/media/kim/Mantu/ai-music/Goa_Separated_crops \\
        --output-dir /run/media/kim/Lehto/goa-stems

Usage — encode one track:
    ./encode_stems.py \\
        --track-dir  "/run/media/kim/Mantu/ai-music/Goa_Separated_crops/0900 X - Dream - Radio" \\
        --output-dir /run/media/kim/Lehto/goa-stems

--source-dir processes every subdirectory as a track folder, loading the model once.
--track-dir targets a single track folder (same crop directory as --source-dir subfolders).

Re-runs are incremental: existing .npy files are skipped. Companion .json files are
refreshed (without re-encoding) when the source .INFO sidecar is newer than the .json.

The companion .json for each stem crop mirrors the full-mix companion format:
  - seconds_total, padding_mask (for silence trimming)
  - position and all other fields from the track's .INFO sidecar

The output directory structure is:
    stem_dir / <track_folder_name> / <stem_crop_name>.npy + .json
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
# Model loading (same as encode_dataset.py)
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
    """Return the .INFO path for a stem file, stripping the stem suffix."""
    stem = audio_path.stem
    for suffix in STEM_SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return audio_path.with_name(stem + ".INFO")


def find_stem_files(track_dir: Path) -> list:
    """Find all stem audio files in a track directory."""
    files = []
    for ext in AUDIO_EXTENSIONS:
        for p in sorted(track_dir.glob(f"*{ext}")):
            if is_stem(p):
                files.append(p)
    return sorted(files)


def build_padding_mask(actual_frames: int, sample_size: int,
                       downsampling_ratio: int) -> list:
    content_latent = actual_frames // downsampling_ratio
    total_latent   = sample_size   // downsampling_ratio
    return [1] * content_latent + [0] * (total_latent - content_latent)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class StemCropDataset(Dataset):
    def __init__(self, files: list, sample_rate: int, sample_size: int):
        self.files = files
        self.sample_rate = sample_rate
        self.sample_size = sample_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio_path = self.files[idx]
        path_str = str(audio_path)
        try:
            audio, sr = torchaudio.load(path_str)

            if sr != self.sample_rate:
                audio = torchaudio.functional.resample(audio, sr, self.sample_rate)

            if audio.shape[0] == 1:
                audio = audio.repeat(2, 1)
            elif audio.shape[0] > 2:
                audio = audio[:2]

            original_frames = audio.shape[1]
            actual_frames = min(original_frames, self.sample_size)
            seconds_total = original_frames / self.sample_rate

            if original_frames < self.sample_size:
                audio = F.pad(audio, (0, self.sample_size - original_frames))
            else:
                audio = audio[:, :self.sample_size]

            info_path = get_info_sidecar(Path(path_str))
            info_str = "{}"
            if info_path.exists():
                try:
                    info_str = info_path.read_text()
                except Exception:
                    pass

            return {
                "audio": audio, "path": path_str,
                "actual_frames": actual_frames, "info_str": info_str,
                "seconds_total": seconds_total, "error": "",
            }
        except Exception as e:
            return {
                "audio": torch.zeros(2, self.sample_size), "path": path_str,
                "actual_frames": 0, "info_str": "{}", "seconds_total": 0.0,
                "error": str(e),
            }


def collate_fn(samples: list) -> dict:
    valid  = [s for s in samples if not s["error"]]
    failed = [s for s in samples if s["error"]]
    return {
        "audio":         torch.stack([s["audio"] for s in valid]) if valid else torch.empty(0),
        "paths":         [s["path"]          for s in valid],
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
        description="Encode separated stem crops to latents (for latent crossfader)"
    )
    parser.add_argument("--source-dir",   default=None,
                        help="Root crops directory — encode all track subfolders (model loaded once)")
    parser.add_argument("--track-dir",    default=None,
                        help="Path to a single track folder (alternative to --source-dir)")
    parser.add_argument("--output-dir",   required=True,
                        help="Root output directory for stem latents (e.g. /run/media/kim/Lehto/goa-stems)")
    parser.add_argument("--model-config", default=str(DEFAULT_MODEL_CONFIG))
    parser.add_argument("--ckpt-path",    default=str(DEFAULT_CKPT_PATH))
    parser.add_argument("--batch-size",   type=int, default=4)
    parser.add_argument("--num-workers",  type=int, default=4)
    parser.add_argument("--model-half",    action="store_true", default=True)
    parser.add_argument("--no-model-half", dest="model_half", action="store_false")
    parser.add_argument("--force",  action="store_true",
                        help="Re-encode even if .npy already exists")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if not args.source_dir and not args.track_dir:
        print("Error: provide --source-dir (all tracks) or --track-dir (one track)",
              file=sys.stderr)
        sys.exit(1)

    stem_dir          = Path(args.output_dir)
    model_config_path = Path(args.model_config)
    ckpt_path         = Path(args.ckpt_path)

    # Collect track directories to process
    if args.source_dir:
        source_dir = Path(args.source_dir)
        if not source_dir.exists():
            print(f"Error: source-dir not found: {source_dir}", file=sys.stderr)
            sys.exit(1)
        track_dirs = sorted([d for d in source_dir.iterdir() if d.is_dir()])
        print(f"Found {len(track_dirs)} track directories in {source_dir}")
    else:
        track_dir = Path(args.track_dir)
        if not track_dir.exists():
            print(f"Error: track-dir not found: {track_dir}", file=sys.stderr)
            sys.exit(1)
        track_dirs = [track_dir]

    for p, label in [(model_config_path, "model-config"), (ckpt_path, "ckpt-path")]:
        if not p.exists():
            print(f"Error: {label} not found: {p}", file=sys.stderr)
            sys.exit(1)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    autoencoder, sample_rate, sample_size, downsampling_ratio = load_autoencoder(
        model_config_path, ckpt_path, args.model_half, device
    )
    dtype = torch.float16 if args.model_half else torch.float32

    total_encoded = 0
    total_errors  = 0
    total_refreshed = 0

    for track_dir in track_dirs:
        out_track_dir = stem_dir / track_dir.name
        out_track_dir.mkdir(parents=True, exist_ok=True)

        all_files = find_stem_files(track_dir)
        if not all_files:
            continue

        # Separate: files to encode vs files needing only a JSON refresh
        json_refresh = []
        if not args.force:
            pending = []
            for f in all_files:
                npy_out  = out_track_dir / f.with_suffix(".npy").name
                json_out = npy_out.with_suffix(".json")
                if not npy_out.exists():
                    pending.append(f)
                else:
                    info_path = get_info_sidecar(f)
                    if (info_path.exists() and json_out.exists() and
                            info_path.stat().st_mtime > json_out.stat().st_mtime):
                        json_refresh.append((f, npy_out))
            skipped = len(all_files) - len(pending) - len(json_refresh)
            if skipped and len(track_dirs) == 1:
                print(f"Skipping {skipped} already-encoded files")
            all_files = pending
        else:
            pending = all_files

        # JSON-only refresh (no re-encoding)
        for audio_path, npy_out in json_refresh:
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
                companion = {}
                for key in ("seconds_total", "padding_mask"):
                    if key in existing:
                        companion[key] = existing[key]
                companion.update(info_data)
                with open(json_out, "w") as f:
                    json.dump(companion, f)
                total_refreshed += 1
            except Exception as e:
                print(f"  JSON refresh error: {audio_path.name}: {e}")

        if not all_files:
            continue

        desc = f"Encoding {track_dir.name[:40]}"
        dataset = StemCropDataset(all_files, sample_rate, sample_size)
        loader  = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            prefetch_factor=2 if args.num_workers > 0 else None,
            persistent_workers=args.num_workers > 0,
            collate_fn=collate_fn,
            pin_memory=(device == "cuda"),
        )

        error_log = out_track_dir / "_encode_errors.txt"

        for batch in tqdm(loader, desc=desc, unit="batch"):
            for path_str, err in batch["failed"]:
                tqdm.write(f"  Load error: {Path(path_str).name}: {err}")
                total_errors += 1
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
                total_errors += len(batch["paths"])
                with open(error_log, "a") as ef:
                    for p in batch["paths"]:
                        ef.write(f"ENCODE\t{p}\t{e}\n")
                continue

            for i, (path_str, actual_frames, info_str, seconds_total) in enumerate(zip(
                batch["paths"], batch["actual_frames"],
                batch["info_strs"], batch["seconds_total"]
            )):
                stem_name = Path(path_str).with_suffix(".npy").name
                npy_out  = out_track_dir / stem_name
                json_out = npy_out.with_suffix(".json")

                try:
                    info_data = json.loads(info_str)
                except Exception:
                    info_data = {}

                padding_mask = build_padding_mask(actual_frames, sample_size, downsampling_ratio)
                companion = {"seconds_total": seconds_total, "padding_mask": padding_mask}
                companion.update(info_data)

                np.save(str(npy_out), latents_np[i])
                with open(json_out, "w") as f:
                    json.dump(companion, f)

                total_encoded += 1

    print(f"\nDone. Encoded: {total_encoded}  JSON-refreshed: {total_refreshed}  Errors: {total_errors}")


if __name__ == "__main__":
    main()
